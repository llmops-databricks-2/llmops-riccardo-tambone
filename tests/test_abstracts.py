"""Batch bibliography abstract retrieval, text classification and snapshot regression tests."""

import io
import json
import urllib.error
from copy import deepcopy
from datetime import UTC, datetime, timedelta, timezone
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import arxiv
import pytest
from test_graph import _pair, _publish_run, _ref
from test_ingestion import FakeSparkSession

from autographrs.config import ProjectConfig
from autographrs.enrichment.abstracts import AbstractMetadata, AbstractResolver, reconstruct_abstract
from autographrs.enrichment.bibliography import BibliographyStore
from autographrs.enrichment.citations import SemanticScholarCitationResolver
from autographrs.enrichment.openalex import OpenAlexCitationResolver
from autographrs.graph import GraphStore, build_citation_graph
from autographrs.graph.models import GraphNode
from autographrs.models import CitationProviderReport


def _nodes() -> list[GraphNode]:
    graph = build_citation_graph(
        "run-1",
        [
            _pair(
                "2401.00001",
                [
                    _ref("scholar:s1"),
                    _ref("scholar:s1", 2),
                    _ref("doi:10.1000/a", 3),
                    _ref("ref:unknown", 4),
                    _ref("openalex:W2", 5),
                ],
            )
        ],
    )
    return [node for node in graph.nodes if node.kind != "selected"]


def test_scholar_batch_and_openalex_fallback_preserve_provenance_and_variants() -> None:
    """Only exactly identified works are queried, once per deduplicated identity."""
    resolver = AbstractResolver(
        scholar=SemanticScholarCitationResolver(min_interval=0),
        openalex=OpenAlexCitationResolver(min_interval=0),
    )
    with (
        patch.object(
            resolver.scholar,
            "_request_json",
            return_value=[
                {"paperId": "doi-a", "title": "Article A", "abstract": "   "},
                {"paperId": "s1", "title": "Article S", "abstract": "Abstract S"},
            ],
        ) as scholar,
        patch.object(resolver.openalex, "_request") as openalex,
    ):
        from autographrs.enrichment.openalex import _Page

        openalex.side_effect = [
            _Page.model_validate(
                {
                    "results": [
                        {
                            "id": "https://openalex.org/W2",
                            "title": "Article W",
                            "abstract_inverted_index": {"world": [1], "Hello": [0]},
                        },
                    ]
                }
            ),
            _Page.model_validate(
                {
                    "results": [
                        {"id": "https://openalex.org/W3", "doi": "https://doi.org/10.1000/a", "title": "Article A"},
                    ]
                }
            ),
        ]
        original = _nodes()
        before = deepcopy(original)
        results = {item.node_id: item for item in resolver.fetch(original)}
    assert original == before
    assert scholar.call_count == 1
    assert scholar.call_args.kwargs["body"]["ids"] == ["DOI:10.1000/a", "s1"]
    assert "references" not in scholar.call_args.args[0]
    assert openalex.call_count == 2
    assert results["scholar:s1"].text_variant == "title_abstract"
    assert results["openalex:W2"].abstract == "Hello world"
    assert results["openalex:W2"].provider == "openalex"
    assert results["doi:10.1000/a"].text_variant == "title_only"
    assert len(results["doi:10.1000/a"].attempts) == 2
    unresolved = next(item for item in results.values() if item.status == "unresolved")
    assert unresolved.attempts == []


@pytest.mark.parametrize("index", [{"A": [1]}, {"A": [-1]}, {"A": [0], "B": [0]}])
def test_invalid_inverted_indexes_are_rejected(index: dict[str, list[int]]) -> None:
    """Malformed positions must not create fabricated or reordered abstracts."""
    with pytest.raises(ValueError, match="positions"):
        reconstruct_abstract(index)


def test_abstract_reconstruction_handles_repetitions_and_absence() -> None:
    """Positions, not dictionary order, determine the reconstructed text."""
    assert reconstruct_abstract({"B": [1], "A": [0, 2]}) == "A B A"
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None


def test_rate_limit_stops_remaining_scholar_chunks() -> None:
    """An exhausted provider is not retried afresh for every work."""
    resolver = AbstractResolver(scholar=SemanticScholarCitationResolver(min_interval=0, batch_size=1))
    limited = urllib.error.HTTPError("https://example.org", 429, "", Message(), None)
    graph = build_citation_graph("run-1", [_pair("2401.00001", [_ref("scholar:s1"), _ref("scholar:s2", 2)])])
    with (
        patch.object(resolver.scholar, "_request_json", side_effect=limited) as request,
        patch.object(resolver.openalex, "_request") as fallback,
    ):
        results = resolver.fetch([node for node in graph.nodes if node.kind != "selected"])
    assert request.call_count == 1
    fallback.assert_not_called()
    assert all(item.status == "rate_limited" and item.text_variant == "title_only" for item in results)


def test_missing_title_is_not_replaced_by_identifier() -> None:
    """An API missing title cannot produce a title-only embedding from a placeholder."""
    resolver = AbstractResolver()
    reference = _ref("scholar:s1")
    reference.title = None
    graph = build_citation_graph("run-1", [_pair("2401.00001", [reference])])
    with patch.object(resolver.scholar, "_request_json", return_value=[None]):
        result = resolver.fetch([node for node in graph.nodes if node.kind != "selected"])[0]
    assert result.title is None and result.text_variant is None
    assert result.status == "missing"


def test_hundreds_of_cited_works_use_bounded_batches() -> None:
    """Two hundred and one unique works are queried in 100/100/1 batches, not per paper."""
    resolver = AbstractResolver(scholar=SemanticScholarCitationResolver(min_interval=0))
    nodes = [
        GraphNode(node_id=f"scholar:s{i}", kind="external", identifiers=[f"scholar:s{i}"], title=f"Paper {i}")
        for i in range(201)
    ]

    def respond(url: str, body: dict[str, list[str]]) -> list[dict[str, str]]:
        return [{"paperId": pid, "title": f"Paper {pid}", "abstract": "Abstract"} for pid in body["ids"]]

    with patch.object(resolver.scholar, "_request_json", side_effect=respond) as request:
        results = resolver.fetch(nodes)
    assert [len(call.kwargs["body"]["ids"]) for call in request.call_args_list] == [100, 100, 1]
    assert len(results) == 201 and all(item.text_variant == "title_abstract" for item in results)


def test_malformed_batch_and_unrelated_openalex_results_are_not_assumed_matches() -> None:
    """Report malformed responses and never attach metadata from an unrelated DOI."""
    from autographrs.enrichment.openalex import _Page

    resolver = AbstractResolver()
    graph = build_citation_graph("run-1", [_pair("2401.00001", [_ref("doi:10.1000/a")])])
    with (
        patch.object(resolver.scholar, "_request_json", return_value=[]),
        patch.object(
            resolver.openalex,
            "_request",
            return_value=_Page.model_validate(
                {
                    "results": [
                        {
                            "id": "https://openalex.org/W99",
                            "doi": "https://doi.org/10.1000/unrelated",
                            "title": "Wrong title",
                            "abstract_inverted_index": {"Wrong": [0]},
                        }
                    ]
                }
            ),
        ),
    ):
        result = resolver.fetch([node for node in graph.nodes if node.kind != "selected"])[0]
    assert result.abstract is None and result.title == "An external paper"
    assert result.status == "error"
    assert result.attempts[0].status == "error"


def test_http_wire_fields_and_openalex_abstract_parsing() -> None:
    """Existing provider HTTP helpers validate real JSON shapes without fetching references."""
    resolver = AbstractResolver(
        scholar=SemanticScholarCitationResolver(min_interval=0),
        openalex=OpenAlexCitationResolver(min_interval=0),
    )
    graph = build_citation_graph("run-1", [_pair("2401.00001", [_ref("openalex:W2")])])
    response = io.BytesIO(
        json.dumps(
            {"results": [{"id": "https://openalex.org/W2", "title": "Title", "abstract_inverted_index": {"Text": [0]}}]}
        ).encode()
    )
    with patch("urllib.request.urlopen", return_value=response) as request:
        result = resolver.fetch([node for node in graph.nodes if node.kind != "selected"])[0]
    assert request.call_count == 1
    assert result.abstract == "Text" and result.text_variant == "title_abstract"


def test_snapshots_and_graph_consume_exact_metadata_without_rewriting_originals(tmp_path: Path) -> None:
    """Published abstract snapshots are run-scoped, immutable, and referenced by graph builds."""
    config = ProjectConfig.from_yaml(str(Path(__file__).resolve().parents[1] / "project_config.yml"), "dev")
    spark = FakeSparkSession()
    graph_store = GraphStore(config, spark=spark, volume_root_override=tmp_path)  # type: ignore[arg-type]
    store = BibliographyStore(config, spark=spark, volume_root_override=tmp_path)  # type: ignore[arg-type]
    run_id = _publish_run(graph_store)
    row = spark.tables[store.full_enrichment_table_name][0]
    row["bibliography"] = [_ref("scholar:s1").model_dump()]
    before = deepcopy(row)
    resolver = AbstractResolver(scholar=SemanticScholarCitationResolver(min_interval=0))
    with patch.object(
        resolver.scholar,
        "_request_json",
        return_value=[{"paperId": "s1", "title": "Cited article", "abstract": "An abstract"}],
    ):
        snapshot = store.enrich_bibliography(run_id, resolver)
    assert row == before
    metadata = store.load_metadata(run_id, snapshot.snapshot_id)
    assert metadata[0].abstract == "An abstract"
    build = graph_store.build_graph(run_id, snapshot.snapshot_id)
    graph = graph_store.load_graph(run_id, build.graph_build_id)
    assert graph.bibliography_snapshot_id == snapshot.snapshot_id
    node = next(node for node in graph.nodes if node.kind == "external")
    assert node.abstract == "An abstract" and node.text_variant == "title_abstract"
    with pytest.raises(ValueError, match="published"):
        store.load_metadata("another-run", snapshot.snapshot_id)
    with (
        patch.object(store.config, "bibliography_cache_days", 0),
        patch.object(resolver, "fetch", side_effect=RuntimeError("test failure")),
        pytest.raises(RuntimeError),
    ):
        store.enrich_bibliography(run_id, resolver)
    failed = next(row for row in spark.tables[store.full_snapshots_table_name] if row["status"] == "FAILED")
    with pytest.raises(ValueError, match="published"):
        store.load_metadata(run_id, failed["snapshot_id"])
    assert store.load_metadata(run_id, snapshot.snapshot_id) == metadata
    spark.tables[store.full_metadata_table_name][0]["abstract"] = None
    spark.tables[store.full_metadata_table_name].clear()
    with pytest.raises(ValueError, match="counts"):
        store.load_metadata(run_id, snapshot.snapshot_id)


@pytest.mark.parametrize("corrupt", [False, True])
def test_spark_naive_timestamp_readback_preserves_integrity(tmp_path: Path, corrupt: bool) -> None:
    """Spark TIMESTAMP collect drops timezone information, including inside nested provider reports."""
    config = ProjectConfig.from_yaml(str(Path(__file__).resolve().parents[1] / "project_config.yml"), "dev")
    spark = FakeSparkSession()
    graph_store = GraphStore(config, spark=spark, volume_root_override=tmp_path)  # type: ignore[arg-type]
    run_id = _publish_run(graph_store)
    spark.tables[graph_store.full_enrichment_table_name][0]["bibliography"] = [_ref("scholar:s1").model_dump()]
    store = BibliographyStore(config, spark=spark)  # type: ignore[arg-type]
    original_read = store._read_rows
    timestamp = datetime(2026, 10, 29, 10, 30, 0, 123456, tzinfo=UTC)
    node_id = next(
        node.node_id
        for node in build_citation_graph(run_id, store.load_run_dataset(run_id)).nodes
        if node.kind == "external"
    )
    resolver = AbstractResolver()
    metadata = AbstractMetadata(
        node_id=node_id,
        title="External title",
        text_variant="title_only",
        status="rate_limited",
        fetched_at=timestamp,
        attempts=[
            CitationProviderReport(
                source="semantic_scholar",
                status="rate_limited",
                error="HTTP 429",
                fetched_at=timestamp.astimezone(timezone(timedelta(hours=2))),
            )
        ],
    )

    def spark_read(table: str, run: str) -> list[dict]:
        rows = original_read(table, run)
        if table == store.full_metadata_table_name:
            for row in rows:
                row["fetched_at"] = row["fetched_at"].astimezone().replace(tzinfo=None)
                for attempt in row["attempts"]:
                    attempt["fetched_at"] = attempt["fetched_at"].astimezone().replace(tzinfo=None)
                if corrupt:
                    row["title"] = "Unexpected modified title"
        return rows

    with (
        patch.object(resolver, "fetch", return_value=[metadata]),
        patch.object(store, "_read_rows", side_effect=spark_read),
    ):
        if corrupt:
            with pytest.raises(ValueError, match="differs"):
                store.enrich_bibliography(run_id, resolver)
        else:
            snapshot = store.enrich_bibliography(run_id, resolver)
            loaded = store.load_metadata(run_id, snapshot.snapshot_id)[0]
            assert loaded == metadata
            assert loaded.fetched_at.tzinfo == UTC
            assert loaded.attempts[0].fetched_at.tzinfo == UTC


def test_arxiv_fallback_exact_ids_and_preserved_provider_failures() -> None:
    """Known arXiv identifiers can recover abstracts after Scholar's quota is exhausted."""
    client = MagicMock()
    client.results.return_value = iter(
        [SimpleNamespace(entry_id="https://arxiv.org/abs/1810.04805v2", title="BERT", summary="An abstract")]
    )
    resolver = AbstractResolver(arxiv_client=client)
    node = GraphNode(
        node_id="arxiv:1810.04805",
        kind="external",
        identifiers=["arxiv:1810.04805"],
        title="BERT",
        text_variant="title_only",
    )
    with patch.object(
        resolver.scholar, "_request_json", side_effect=urllib.error.HTTPError("url", 429, "quota", {}, None)
    ):
        item = resolver.fetch([node])[0]
    assert item.status == "ok" and item.provider == "arxiv" and item.abstract == "An abstract"
    assert [attempt.status for attempt in item.attempts] == ["rate_limited", "ok"]
    search = client.results.call_args.args[0]
    assert search.id_list == ["1810.04805"]


@pytest.mark.parametrize("failure", ["missing", "foreign", "quota"])
def test_arxiv_fallback_reports_missing_and_failed_lookups(failure: str) -> None:
    """Do not match by title or convert arXiv failures into fabricated successful metadata."""
    client = MagicMock()
    if failure == "quota":
        client.results.side_effect = arxiv.HTTPError("url", 2, 429)
    elif failure == "foreign":
        client.results.return_value = iter(
            [SimpleNamespace(entry_id="https://arxiv.org/abs/9999.99999", title="Same title", summary="Wrong work")]
        )
    else:
        client.results.return_value = iter([])
    resolver = AbstractResolver(arxiv_client=client)
    node = GraphNode(node_id="a", kind="external", identifiers=["arxiv:1810.04805"], title="BERT")
    with patch.object(resolver.scholar, "_request_json", return_value=[None]):
        item = resolver.fetch([node])[0]
    assert item.abstract is None
    assert item.attempts[-1].source == "arxiv"
    assert item.attempts[-1].status == {"quota": "rate_limited", "foreign": "error", "missing": "not_found"}[failure]


def test_successful_abstract_cache_reuses_across_node_ids_and_expires() -> None:
    """Persistent cache is identity-based, positive-only, versioned and independent of run-local IDs."""
    config = ProjectConfig.from_yaml(str(Path(__file__).resolve().parents[1] / "project_config.yml"), "dev")
    spark = FakeSparkSession()
    store = BibliographyStore(config, spark=spark)  # type: ignore[arg-type]
    node = GraphNode(
        node_id="first-run-node", kind="external", identifiers=["doi:10.1000/a", "scholar:s1"], title="Title"
    )
    item = AbstractMetadata(
        node_id=node.node_id,
        title="Title",
        abstract="Abstract",
        text_variant="title_abstract",
        status="ok",
        provider="semantic_scholar",
        evidence_url="https://example.org",
    )
    resolver = MagicMock()
    resolver.fetch.return_value = [item]
    assert store._fetch_cached([node], resolver) == [item]
    other = node.model_copy(update={"node_id": "second-run-node", "identifiers": list(reversed(node.identifiers))})
    new_store = BibliographyStore(config, spark=spark)  # type: ignore[arg-type]
    hit = new_store._fetch_cached([other], resolver)[0]
    assert hit.node_id == other.node_id and hit.fetched_at == item.fetched_at
    assert hit.provider == item.provider
    assert resolver.fetch.call_count == 1
    row = spark.tables[store.full_cache_table_name][0]
    expired = item.model_copy(update={"fetched_at": datetime.now(UTC) - timedelta(days=31)})
    row["metadata_json"] = expired.model_dump_json()
    resolver.fetch.return_value = [item.model_copy(update={"node_id": other.node_id})]
    store._fetch_cached([other], resolver)
    assert resolver.fetch.call_count == 2
    changed_aliases = node.model_copy(update={"identifiers": ["doi:10.1000/different", "scholar:s1"]})
    resolver.fetch.return_value = [item]
    store._fetch_cached([changed_aliases], resolver)
    assert resolver.fetch.call_count == 3


def test_abstract_cache_does_not_store_errors_or_unresolved_identities() -> None:
    """Rate limiting must not poison future attempts; unresolved occurrences are not global identities."""
    config = ProjectConfig.from_yaml(str(Path(__file__).resolve().parents[1] / "project_config.yml"), "dev")
    spark = FakeSparkSession()
    store = BibliographyStore(config, spark=spark)  # type: ignore[arg-type]
    node = GraphNode(node_id="a", kind="external", identifiers=["scholar:s1"], title="Title")
    resolver = MagicMock()
    resolver.fetch.return_value = [AbstractMetadata(node_id="a", status="rate_limited")]
    store._fetch_cached([node], resolver)
    store._fetch_cached([node], resolver)
    assert resolver.fetch.call_count == 2 and not spark.tables[store.full_cache_table_name]
    node.kind = "unresolved"
    resolver.fetch.return_value = [
        AbstractMetadata(node_id="a", title="Title", abstract="Text", text_variant="title_abstract", status="ok")
    ]
    store._fetch_cached([node], resolver)
    assert not spark.tables[store.full_cache_table_name]


def test_arxiv_batches_and_no_calls_when_scholar_has_abstract() -> None:
    """Large known-ID sets use bounded batches; successful Scholar results need no arXiv request."""
    client = MagicMock()

    def results(search: arxiv.Search) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(entry_id=f"https://arxiv.org/abs/{key}v1", title=key, summary="Abstract")
            for key in search.id_list
        ]

    client.results.side_effect = results
    resolver = AbstractResolver(arxiv_client=client)
    nodes = [
        GraphNode(node_id=f"node-{index}", kind="external", identifiers=[f"arxiv:2401.{index:05d}"], title="Title")
        for index in range(201)
    ]
    with patch.object(resolver.scholar, "_request_json", side_effect=lambda url, body: [None] * len(body["ids"])):
        items = resolver.fetch(nodes)
    assert all(item.status == "ok" for item in items)
    assert [len(call.args[0].id_list) for call in client.results.call_args_list] == [100, 100, 1]
    client.results.reset_mock()
    with patch.object(
        resolver.scholar,
        "_request_json",
        return_value=[{"paperId": "s1", "title": "Title", "abstract": "Scholar abstract"}],
    ):
        resolver.fetch(nodes[:1])
    client.results.assert_not_called()


def test_cached_success_survives_snapshot_write_failure(tmp_path: Path) -> None:
    """A failed publication does not discard already acquired positive metadata."""
    config = ProjectConfig.from_yaml(str(Path(__file__).resolve().parents[1] / "project_config.yml"), "dev")
    spark = FakeSparkSession()
    graph_store = GraphStore(config, spark=spark, volume_root_override=tmp_path)  # type: ignore[arg-type]
    run_id = _publish_run(graph_store)
    spark.tables[graph_store.full_enrichment_table_name][0]["bibliography"] = [_ref("scholar:s1").model_dump()]
    store = BibliographyStore(config, spark=spark)  # type: ignore[arg-type]
    node = next(
        node for node in build_citation_graph(run_id, store.load_run_dataset(run_id)).nodes if node.kind == "external"
    )
    resolver = MagicMock()
    resolver.fetch.return_value = [
        AbstractMetadata(
            node_id=node.node_id, title="Title", abstract="Abstract", text_variant="title_abstract", status="ok"
        )
    ]
    original = store._merge_records

    def merge(records: list[dict], table: str, *args: object, **kwargs: object) -> int:
        if table == store.full_metadata_table_name:
            raise RuntimeError("Snapshot write failed")
        return original(records, table, *args, **kwargs)

    with patch.object(store, "_merge_records", side_effect=merge), pytest.raises(RuntimeError, match="write failed"):
        store.enrich_bibliography(run_id, resolver)
    snapshot = store.enrich_bibliography(run_id, resolver)
    assert snapshot.status == "COMPLETED"
    assert resolver.fetch.call_count == 1
