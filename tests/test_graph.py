"""Citation graph identity, direction, evidence and publication regression tests."""

import runpy
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from test_enrichment import FAKE_TAXONOMY, FakeChatCompletions, FakeCitationProvider, FakeOpenAIClient, _make_paper
from test_ingestion import FakeSparkSession
from test_run_storage import MemorySource

from autographrs.config import ProjectConfig
from autographrs.enrichment.bibliography import BibliographyStore
from autographrs.enrichment.tagger import PaperTagger
from autographrs.graph import GraphStore, build_citation_graph
from autographrs.models import BibliographicReference, Paper, PaperEnrichment
from autographrs.pipeline import run_ingestion


def _ref(
    target: str,
    position: int = 1,
    arxiv: str | None = None,
    doi: str | None = None,
) -> BibliographicReference:
    namespace = target.partition(":")[0]
    return BibliographicReference(
        reference_id=f"r{position}",
        position=position,
        target_id=target,
        match_method={
            "scholar": "scholar_id",
            "openalex": "openalex_id",
            "arxiv": "arxiv_id",
            "doi": "doi",
            "ref": "unresolved",
        }[namespace],
        text="A reference",
        title="An external paper",
        arxiv_ids=[arxiv] if arxiv else [],
        dois=[doi] if doi else [],
        evidence_url="https://example.org/bibliography",
    )


def _pair(
    pid: str,
    refs: list[BibliographicReference] | None = None,
    doi: str | None = None,
    provider_id: str | None = None,
    source: str = "arxiv",
) -> tuple[Paper, PaperEnrichment]:
    paper = _make_paper(pid, f"Paper {pid}").model_copy(update={"doi": doi, "source": source})
    return paper, PaperEnrichment(
        run_id="run-1",
        source=source,
        paper_id=pid,
        title=paper.title,
        bibliography=refs or [],
        citation_status="ok",
        citation_source="semantic_scholar",
        citation_paper_ids=[provider_id] if provider_id else [],
    )


def test_selected_external_and_unresolved_references_form_an_open_graph() -> None:
    """Direction comes only from bibliographies; missing identities remain separate."""
    data = [
        _pair(
            "2401.00001",
            [
                _ref("arxiv:2402.00002v2"),
                _ref("doi:10.1000/external", 2),
                _ref("ref:same", 3),
                _ref("ref:same", 4),
            ],
        ),
        _pair("2402.00002", [_ref("doi:10.1000/external")]),
    ]
    original = deepcopy(data)
    graph = build_citation_graph("run-1", data)
    assert data == original
    assert len(graph.nodes) == 5
    assert len(graph.edges) == 5
    assert sum(node.kind == "unresolved" for node in graph.nodes) == 2
    pairs = {(edge.source_id, edge.target_id) for edge in graph.edges}
    assert ("paper:arxiv:2401.00001", "paper:arxiv:2402.00002") in pairs
    assert ("paper:arxiv:2402.00002", "paper:arxiv:2401.00001") not in pairs
    assert sum(dst == "doi:10.1000/external" for _, dst in pairs) == 2
    assert all(edge.evidence[0].provider == "semantic_scholar" for edge in graph.edges)
    assert graph == build_citation_graph("run-1", list(reversed(data)))


def test_cross_provider_aliases_merge_transitively_without_title_matching() -> None:
    """A bridge reference can reconcile Scholar, DOI and OpenAlex across selected works."""
    data = [
        _pair(
            "local-a",
            [
                _ref("scholar:s1", doi="https://doi.org/10.1000/ABC"),
                _ref("openalex:W2", 2, doi="10.1000/abc"),
                _ref("scholar:unrelated", 3),
            ],
            source="other",
        ),
        _pair("local-b", provider_id="https://openalex.org/W2", source="other"),
    ]
    graph = build_citation_graph("run-1", data)
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    edge = next(edge for edge in graph.edges if edge.target_id == "paper:other:local-b")
    assert len(edge.evidence) == 2
    assert {item.position for item in edge.evidence} == {1, 2}


def test_conflicting_components_are_quarantined_including_bridge_references() -> None:
    """Do not fuse different selected records or choose one side of conflicting evidence."""
    data = [
        _pair("2401.00001", [_ref("doi:10.1000/shared")]),
        _pair("2402.00002", doi="10.1000/shared"),
        _pair("2403.00003", doi="10.1000/shared"),
    ]
    graph = build_citation_graph("run-1", data)
    assert len(graph.nodes) == 4
    edge = graph.edges[0]
    target = next(node for node in graph.nodes if node.node_id == edge.target_id)
    assert target.kind == "unresolved"
    assert target.identity_status == "ambiguous"
    assert graph.warnings
    assert graph == build_citation_graph("run-1", list(reversed(data)))
    contradictory = [_pair("2401.00001", [_ref("scholar:s1", doi="10.1000/a"), _ref("scholar:s1", 2, doi="10.1000/b")])]
    result = build_citation_graph("run-1", contradictory)
    assert len(result.nodes) == 3
    assert all(node.identity_status == "ambiguous" for node in result.nodes if node.kind == "unresolved")


def test_doi_arxiv_alias_and_self_citations() -> None:
    """An arXiv DOI and versioned arXiv ID identify one work, without an inverse citation."""
    data = [
        _pair("2401.00001", [_ref("doi:10.48550/arXiv.2402.00002"), _ref("arxiv:2401.00001v2", 2)]),
        _pair("2402.00002"),
    ]
    graph = build_citation_graph("run-1", data)
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0].target_id == "paper:arxiv:2402.00002"
    assert any("Self-citation omitted" in warning for warning in graph.warnings)


def test_missing_partial_and_empty_bibliographies_are_explicit() -> None:
    """Empty and legacy bibliographies preserve selected nodes and coverage warnings."""
    pair = _pair("2401.00001")
    pair[1].bibliography = None
    pair[1].citation_status = "rate_limited"
    graph = build_citation_graph("run-1", [pair])
    assert len(graph.nodes) == 1 and not graph.edges
    assert "rate_limited" in graph.warnings[0]
    assert "available=False" in graph.warnings[0]
    assert build_citation_graph("run-1", []).nodes == []


def test_invalid_input_and_dangling_edges_are_rejected() -> None:
    """Run isolation, exact pairing and unique selections are mandatory."""
    pair = _pair("2401.00001")
    for data in ([pair, pair], [(pair[0], pair[1].model_copy(update={"run_id": "another"}))]):
        with pytest.raises(ValueError):
            build_citation_graph("run-1", data)
    graph = build_citation_graph("run-1", [_pair("2401.00001", [_ref("scholar:s1")])])
    graph.nodes = [node for node in graph.nodes if node.kind == "selected"]
    with pytest.raises(ValueError, match="dangling"):
        graph.validate_integrity()


@pytest.fixture
def graph_store(tmp_path: Path) -> GraphStore:
    """Publish a minimal enrichment run in an isolated SQL-backed test double."""
    config = ProjectConfig.from_yaml(str(Path(__file__).resolve().parents[1] / "project_config.yml"), "dev")
    return GraphStore(config, spark=FakeSparkSession(), volume_root_override=tmp_path)  # type: ignore[arg-type]


def _publish_run(store: GraphStore) -> str:
    with patch("autographrs.enrichment.tagger.fetch_arxiv_category_taxonomy", return_value=FAKE_TAXONOMY):
        result = run_ingestion(
            "graph",
            config=store.config,
            store=store,
            source=MemorySource([_make_paper("2401.00001", "A")]),
            download_pdfs=False,
            citation_provider=FakeCitationProvider({}),
            tagger=PaperTagger(store.config, client=FakeOpenAIClient(FakeChatCompletions(["rag"], "Summary"))),
        )
    return result.run_id


def test_graph_build_publication_is_isolated_and_rebuildable(graph_store: GraphStore) -> None:
    """Each build is an independent snapshot; published ingestion data is never modified."""
    store = graph_store
    run_id = _publish_run(store)
    row = store.spark.tables[store.full_enrichment_table_name][0]  # type: ignore[attr-defined]
    row["bibliography"] = [_ref("doi:10.1000/external").model_dump()]
    graph_queries_start = len(store.spark.sql_queries)  # type: ignore[attr-defined]
    before = deepcopy(store.spark.tables[store.full_runs_table_name])  # type: ignore[attr-defined]
    first = store.build_graph(run_id)
    second = store.build_graph(run_id)
    assert first.graph_build_id != second.graph_build_id
    assert first.status == second.status == "COMPLETED"
    loaded = store.load_graph(run_id, first.graph_build_id)
    assert len(loaded.nodes) == 2 and len(loaded.edges) == 1
    assert loaded.edges[0].evidence[0].text == "A reference"
    assert store.load_graph(run_id, first.graph_build_id) == store.load_graph(run_id, second.graph_build_id)
    assert before == store.spark.tables[store.full_runs_table_name]  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="not published"):
        store.load_graph(run_id, "missing")
    with pytest.raises(ValueError):
        store.load_graph("different-run", first.graph_build_id)
    assert all(
        not query.startswith("DELETE")
        for query in store.spark.sql_queries[graph_queries_start:]  # type: ignore[attr-defined]
    )


def test_failed_build_is_unpublished_and_retry_uses_a_new_snapshot(graph_store: GraphStore) -> None:
    """A failure between writes cannot expose a partial graph or damage an earlier build."""
    store = graph_store
    run_id = _publish_run(store)
    good = store.build_graph(run_id)
    original = store._ensure_table

    def fail_before_edges(table: str, schema: str) -> None:
        if table == store.full_edges_table_name:
            raise RuntimeError("Simulated graph publication failure")
        original(table, schema)

    with patch.object(store, "_ensure_table", side_effect=fail_before_edges), pytest.raises(RuntimeError):
        store.build_graph(run_id)
    rows = store.spark.tables[store.full_builds_table_name]  # type: ignore[attr-defined]
    failed = next(row for row in rows if row["status"] == "FAILED")
    with pytest.raises(ValueError, match="not published"):
        store.load_graph(run_id, failed["graph_build_id"])
    assert store.load_graph(run_id, good.graph_build_id).nodes
    retry = store.build_graph(run_id)
    assert retry.status == "COMPLETED"
    assert retry.graph_build_id != failed["graph_build_id"]


def test_empty_graph_and_stored_corruption_are_handled(graph_store: GraphStore) -> None:
    """Empty edge arrays use explicit schemas, and publication counts are checked on reads."""
    store = graph_store
    run_id = _publish_run(store)
    build = store.build_graph(run_id)
    assert store.load_graph(run_id, build.graph_build_id).edges == []
    store.spark.tables[store.full_nodes_table_name].clear()  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="counts"):
        store.load_graph(run_id, build.graph_build_id)


def test_empty_selection_publishes_zero_counts(graph_store: GraphStore) -> None:
    """Zero-node graphs still have explicit tables and a valid publication marker."""
    store = graph_store
    result = run_ingestion(
        "empty",
        config=store.config,
        store=store,
        source=MemorySource([]),
        download_pdfs=False,
        citation_provider=FakeCitationProvider({}),
    )
    build = store.build_graph(result.run_id)
    graph = store.load_graph(result.run_id, build.graph_build_id)
    assert build.node_count == build.edge_count == 0
    assert not graph.nodes and not graph.edges


def test_graph_rejects_skipped_enrichment(graph_store: GraphStore) -> None:
    """A completed metadata-only run is not a completed bibliography snapshot."""
    store = graph_store
    result = run_ingestion(
        "no-enrichment",
        config=store.config,
        store=store,
        source=MemorySource([_make_paper("2401.00001", "A")]),
        download_pdfs=False,
        enrich=False,
    )
    with pytest.raises(ValueError, match="skipped"):
        store.build_graph(result.run_id)


def test_hundreds_of_documents_have_exact_node_and_edge_counts() -> None:
    """Three hundred selected works share one external node and form a directed chain."""
    data = []
    for index in range(300):
        refs = [_ref("doi:10.1000/common")]
        if index:
            refs.append(_ref(f"arxiv:2401.{index - 1:05d}", 2))
        data.append(_pair(f"2401.{index:05d}", refs))
    graph = build_citation_graph("run-1", data)
    assert len(graph.nodes) == 301
    assert len(graph.edges) == 599
    assert len(graph.warnings) == 0


def test_job_and_notebooks_pass_the_actual_published_run_id(graph_store: GraphStore) -> None:
    """The graph task consumes the generated run ID, not the empty new-search job parameter."""
    root = Path(__file__).resolve().parents[1]
    job = yaml.safe_load((root / "resources" / "autograph_pipeline.job.yml").read_text())["resources"]["jobs"][
        "ingest_papers"
    ]
    assert job["name"] == "job-autograph-pipeline"
    tasks = {task["task_key"]: task for task in job["tasks"]}
    graph_task = tasks["build_graph"]
    assert graph_task["depends_on"] == [{"task_key": "enrich_bibliography"}]
    assert tasks["enrich_bibliography"]["depends_on"] == [{"task_key": "ingest_papers"}]
    assert graph_task["notebook_task"]["base_parameters"]["bibliography_snapshot_id"] == (
        "{{tasks.enrich_bibliography.values.bibliography_snapshot_id}}"
    )
    assert graph_task["notebook_task"]["base_parameters"]["run_id"] == "{{tasks.ingest_papers.values.run_id}}"
    assert graph_task["environment_key"] == tasks["ingest_papers"]["environment_key"]
    assert job["environments"][0]["spec"]["environment_version"] == "5"

    store = graph_store
    run_id = _publish_run(store)
    dataset = store.load_run_dataset(run_id)
    result = SimpleNamespace(
        run_id=run_id,
        papers=[paper for paper, _ in dataset],
        enrichments=[item for _, item in dataset],
        delta_table=store.full_table_name,
        enrichment_table=store.full_enrichment_table_name,
        volume_dir="pdfs",
    )
    parameters = {
        "env": "dev",
        "topic": "graph",
        "num_papers": "1",
        "ranking_criteria": "relevance",
        "criteria_date": "",
        "categories": "",
        "run_id": "",
    }
    dbutils = MagicMock()
    dbutils.widgets.get.side_effect = parameters.__getitem__
    runtime = ModuleType("databricks.sdk.runtime")
    runtime.dbutils = dbutils
    with (
        patch.dict(sys.modules, {"databricks.sdk.runtime": runtime}),
        patch("autographrs.config.load_config", return_value=store.config),
        patch("autographrs.pipeline.run_ingestion", return_value=result) as ingestion,
        patch("autographrs.storage.delta.DeltaPaperStore", return_value=store),
    ):
        runpy.run_path(str(root / "resources" / "deployment_notebooks" / "ingest_papers.py"))
    assert ingestion.call_args.kwargs["run_id"] is None
    dbutils.jobs.taskValues.set.assert_called_once_with(key="run_id", value=run_id)
    parameters["run_id"] = run_id
    dbutils.jobs.taskValues.set.reset_mock()
    metadata_store = BibliographyStore(store.config, spark=store.spark)
    with (
        patch.dict(sys.modules, {"databricks.sdk.runtime": runtime}),
        patch("autographrs.config.load_config", return_value=store.config),
        patch("autographrs.enrichment.bibliography.BibliographyStore", return_value=metadata_store),
    ):
        runpy.run_path(str(root / "resources" / "deployment_notebooks" / "enrich_bibliography.py"))
    snapshot_id = dbutils.jobs.taskValues.set.call_args.kwargs["value"]
    metadata_store.load_metadata(run_id, snapshot_id)
    parameters["bibliography_snapshot_id"] = snapshot_id
    dbutils.jobs.taskValues.set.reset_mock()
    with (
        patch.dict(sys.modules, {"databricks.sdk.runtime": runtime}),
        patch("autographrs.config.load_config", return_value=store.config),
        patch("autographrs.graph.GraphStore", return_value=store),
    ):
        runpy.run_path(str(root / "resources" / "deployment_notebooks" / "build_graph.py"))
    values = {call.kwargs["key"]: call.kwargs["value"] for call in dbutils.jobs.taskValues.set.call_args_list}
    assert values["run_id"] == run_id
    graph = store.load_graph(run_id, values["graph_build_id"])
    assert graph.nodes
    assert graph.bibliography_snapshot_id == snapshot_id
