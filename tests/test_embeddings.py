"""Gateway, immutable embedding snapshots and Lakebase publication regression tests."""

import runpy
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import pytest
import yaml
from openai.types import CreateEmbeddingResponse, Embedding
from test_graph import _publish_run, _ref
from test_ingestion import FakeSparkSession

from autographrs.config import ProjectConfig
from autographrs.embeddings.gateway import GatewayEmbedder
from autographrs.embeddings.models import SearchPublication
from autographrs.embeddings.search import SearchStore
from autographrs.embeddings.storage import EmbeddingStore
from autographrs.enrichment.abstracts import AbstractMetadata
from autographrs.enrichment.bibliography import BibliographyStore
from autographrs.graph import GraphStore, build_citation_graph
from autographrs.graph.models import CitationGraph

ROOT = Path(__file__).resolve().parents[1]


def _config(**updates: object) -> ProjectConfig:
    config = ProjectConfig.from_yaml(str(ROOT / "project_config.yml"), "dev")
    return ProjectConfig.model_validate({**config.model_dump(by_alias=True), **updates})


def _engine(config: ProjectConfig) -> GatewayEmbedder:
    client = MagicMock()

    def create(*, input: list[str], model: str, encoding_format: str) -> CreateEmbeddingResponse:
        return CreateEmbeddingResponse(
            data=[
                Embedding(index=index, embedding=[0.1, 0.2, float(len(text))], object="embedding")
                for index, text in reversed(list(enumerate(input)))
            ],
            model="course-embedding-model",
            object="list",
            usage={"prompt_tokens": 1, "total_tokens": 1},
        )

    client.with_options.return_value.embeddings.create.side_effect = create
    return GatewayEmbedder(config, client)


@pytest.fixture
def prepared_input(tmp_path: Path) -> tuple[SearchStore, str, str]:
    """Provide selected, external title-only and title-less unresolved documents."""
    config = _config()
    spark = FakeSparkSession()
    graph_store = GraphStore(config, spark=spark, volume_root_override=tmp_path)  # type: ignore[arg-type]
    run_id = _publish_run(graph_store)
    row = spark.tables[graph_store.full_enrichment_table_name][0]
    row["bibliography"] = [_ref("doi:10.1000/external").model_dump(), _ref("ref:missing", 2).model_dump()]
    row["bibliography"][1]["title"] = None
    graph = build_citation_graph(run_id, graph_store.load_run_dataset(run_id))
    resolver = MagicMock()
    resolver.fetch.return_value = [
        AbstractMetadata(
            node_id=node.node_id,
            title=node.title if node.text_variant else None,
            text_variant=node.text_variant,
            status="missing" if node.kind == "external" else "unresolved",
        )
        for node in graph.nodes
        if node.kind != "selected"
    ]
    metadata = BibliographyStore(config, spark=spark)  # type: ignore[arg-type]
    bibliography = metadata.enrich_bibliography(run_id, resolver)
    return SearchStore(config, spark=spark), run_id, bibliography.snapshot_id  # type: ignore[arg-type]


def test_gateway_batches_order_indices_and_never_truncates() -> None:
    """Use the configured Gateway, not an assumed backing model, and preserve the entire text."""
    config = _config(embedding_max_chars=12)
    engine = _engine(config)
    vectors, model = engine.embed(["first", "second"])
    assert vectors == [[0.1, 0.2, 5.0], [0.1, 0.2, 6.0]]
    assert model == "course-embedding-model"
    assert engine.client.with_options.return_value.embeddings.create.call_args.kwargs == {
        "model": "course_ops.gateway.embed",
        "input": ["first", "second"],
        "encoding_format": "float",
    }
    for texts in ([], [""], ["a" * 13], ["a"] * 33):
        with pytest.raises(ValueError):
            engine.embed(texts)


@pytest.mark.parametrize(
    "vectors,indices",
    [
        ([[1.0], [2.0]], [0, 0]),
        ([[0.0]], [0]),
        ([[float("nan")]], [0]),
        ([[float("inf")]], [0]),
        ([[]], [0]),
        ([[1.0], [1.0, 2.0]], [0, 1]),
    ],
)
def test_gateway_rejects_malformed_vectors(vectors: list[list[float]], indices: list[int]) -> None:
    """Reject duplicate indices, zero/nonfinite vectors and inconsistent dimensions."""
    engine = _engine(_config())
    engine.client.with_options.return_value.embeddings.create.side_effect = None
    engine.client.with_options.return_value.embeddings.create.return_value = CreateEmbeddingResponse(
        data=[
            Embedding(index=index, embedding=vector, object="embedding")
            for index, vector in zip(indices, vectors, strict=True)
        ],
        model="model",
        object="list",
        usage={"prompt_tokens": 1, "total_tokens": 1},
    )
    with pytest.raises(ValueError):
        engine.embed(["text"] * len(vectors))


def test_embedding_snapshots_reuse_vectors_and_preserve_sources(prepared_input: tuple[SearchStore, str, str]) -> None:
    """Changing retrieval backend does not invalidate existing embedding snapshots or mutate source data."""
    store, run_id, bib = prepared_input
    before = deepcopy(store.spark.tables[store.full_enrichment_table_name])
    engine = _engine(store.config)
    first = store.prepare(run_id, bib, engine)
    second = store.prepare(run_id, bib, engine)
    assert engine.client.with_options.return_value.embeddings.create.call_count == 1
    assert first.snapshot_id != second.snapshot_id
    assert first.document_count == second.document_count == 2 and first.skipped_count == 1
    assert store.spark.tables[store.full_enrichment_table_name] == before
    _, docs = store.load_snapshot(run_id, first.snapshot_id)
    assert {doc.text_variant for doc in docs} == {"title_only", "title_abstract"}
    assert not any("enableChangeDataFeed" in query for query in store.spark.sql_queries)
    with pytest.raises(ValueError, match="published"):
        store.load_snapshot("other-run", first.snapshot_id)
    changed = EmbeddingStore(_config(embedding_revision="v2"), spark=store.spark)
    with pytest.raises(ValueError, match="incompatible"):
        changed.load_snapshot(run_id, first.snapshot_id)
    row = next(
        row for row in store.spark.tables[store.full_documents_table_name] if row["snapshot_id"] == first.snapshot_id
    )
    row["text"] = "corruption"
    with pytest.raises(ValueError, match="integrity"):
        store.load_snapshot(run_id, first.snapshot_id)


def test_failed_embedding_batches_are_unpublished_and_reusable(prepared_input: tuple[SearchStore, str, str]) -> None:
    """Repair reuses persisted successful batches without publishing failed snapshots."""
    store, run_id, bib = prepared_input
    store.config = _config(embedding_batch_size=1)
    engine = _engine(store.config)
    original = engine.embed
    calls = 0

    def fail_second(texts: list[str]) -> tuple[list[list[float]], str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Gateway unavailable")
        return original(texts)

    with patch.object(engine, "embed", side_effect=fail_second), pytest.raises(RuntimeError, match="Gateway"):
        store.prepare(run_id, bib, engine)
    failed = next(row for row in store.spark.tables[store.full_snapshots_table_name] if row["status"] == "FAILED")
    with pytest.raises(ValueError, match="published"):
        store.load_snapshot(run_id, failed["snapshot_id"])
    assert len(store.spark.tables[store.full_documents_table_name]) == 1
    assert store.prepare(run_id, bib, engine).document_count == 2
    assert engine.client.with_options.return_value.embeddings.create.call_count == 2


def test_model_drift_and_overlong_text_fail_explicitly(prepared_input: tuple[SearchStore, str, str]) -> None:
    """Model changes cannot silently mix vector spaces, and text guards do not truncate."""
    store, run_id, bib = prepared_input
    store.config = _config(embedding_batch_size=1)
    engine = _engine(store.config)
    original = engine.embed
    calls = 0

    def drifting(texts: list[str]) -> tuple[list[list[float]], str]:
        nonlocal calls
        calls += 1
        vectors, model = original(texts)
        return vectors, model if calls == 1 else "different-model"

    with patch.object(engine, "embed", side_effect=drifting), pytest.raises(ValueError, match="changed"):
        store.prepare(run_id, bib, engine)
    store.config = _config(embedding_max_chars=1)
    with pytest.raises(ValueError, match="no truncation"):
        store.prepare(run_id, bib, engine)


def test_identical_text_reuses_vectors_without_merging_nodes(prepared_input: tuple[SearchStore, str, str]) -> None:
    """Content caching must not turn title equality into identity equality."""
    store, run_id, bib = prepared_input
    graph = BibliographyStore(store.config, spark=store.spark).load_graph_input(run_id, bib)
    selected = next(node for node in graph.nodes if node.kind == "selected")
    for node in graph.nodes:
        node.title, node.abstract, node.text_variant = selected.title, selected.abstract, selected.text_variant
    engine = _engine(store.config)
    with patch("autographrs.enrichment.bibliography.BibliographyStore.load_graph_input", return_value=graph):
        snapshot = store.prepare(run_id, bib, engine)
    _, docs = store.load_snapshot(run_id, snapshot.snapshot_id)
    assert len(docs) == 3 and len({doc.node_id for doc in docs}) == 3
    assert len({doc.embedding_key for doc in docs}) == 1
    assert len(engine.client.with_options.return_value.embeddings.create.call_args.kwargs["input"]) == 1


def test_empty_snapshot_needs_no_gateway_or_postgres(prepared_input: tuple[SearchStore, str, str]) -> None:
    """Empty snapshots never create zero-dimensional indexes."""
    store, run_id, bib = prepared_input
    engine = _engine(store.config)
    with patch(
        "autographrs.enrichment.bibliography.BibliographyStore.load_graph_input",
        return_value=CitationGraph(run_id=run_id),
    ):
        snapshot = store.prepare(run_id, bib, engine)
    with patch("autographrs.embeddings.search.LakebasePgVectorManager") as manager:
        publication = store.publish(run_id, snapshot.snapshot_id)
        assert store.search("anything", run_id, snapshot.snapshot_id, publication.publication_id, embedder=engine) == []
        manager.assert_not_called()
    assert snapshot.document_count == snapshot.dimension == 0
    engine.client.with_options.assert_not_called()


def test_shared_database_and_separate_postgres_schemas() -> None:
    """Share the existing project/compute without sharing application tables between environments."""
    configs = [ProjectConfig.from_yaml(str(ROOT / "project_config.yml"), env) for env in ("dev", "acc", "prd")]
    assert len({config.lakebase_host for config in configs}) == 1
    assert {config.lakebase_schema for config in configs} == {"autographrs_dev", "autographrs_acc", "autographrs_prd"}
    assert len({SearchStore(config).full_documents_table_name for config in configs}) == 3


def test_publication_search_and_graph_lineage(prepared_input: tuple[SearchStore, str, str]) -> None:
    """New pgvector publications preserve lineage; old Vector Search publications cannot be reinterpreted."""
    store, run_id, bib = prepared_input
    engine = _engine(store.config)
    snapshot = store.prepare(run_id, bib, engine)
    _, docs = store.load_snapshot(run_id, snapshot.snapshot_id)
    with patch("autographrs.embeddings.search.LakebasePgVectorManager", autospec=True) as manager:
        manager.return_value.search.return_value = [{"document_id": docs[0].document_id, "cosine_similarity": 0.9}]
        publication = store.publish(run_id, snapshot.snapshot_id)
        assert publication.backend == "lakebase_pgvector"
        assert manager.return_value.method_calls[:2] == [call.setup(), call.publish(snapshot, docs)]
        manager.return_value.publish.assert_called_once_with(snapshot, docs)
        results = store.search("memory", run_id, snapshot.snapshot_id, publication.publication_id, embedder=engine)
        assert results[0]["cosine_similarity"] == 0.9
        with pytest.raises(ValueError, match="published"):
            store.search("memory", run_id, snapshot.snapshot_id, "missing", embedder=engine)
        manager.return_value.search.return_value = [{"document_id": "foreign"}]
        with pytest.raises(ValueError, match="unknown"):
            store.search("memory", run_id, snapshot.snapshot_id, publication.publication_id, embedder=engine)
    graph_store = GraphStore(store.config, spark=store.spark)
    build = graph_store.build_graph(run_id, bib, snapshot.snapshot_id, publication.publication_id)
    graph = graph_store.load_graph(run_id, build.graph_build_id)
    assert graph.embedding_snapshot_id == snapshot.snapshot_id
    assert all(edge.relation == "CITES" for edge in graph.edges)
    store.spark.tables[store.full_publications_table_name][0]["backend"] = "vector_search"
    with pytest.raises(ValueError, match="backend"):
        store.load_publication(run_id, snapshot.snapshot_id, publication.publication_id)


@pytest.mark.parametrize("failing_step", ["setup", "publish"])
def test_postgres_failure_does_not_publish_and_retry_does_not_embed(
    prepared_input: tuple[SearchStore, str, str],
    failing_step: str,
) -> None:
    """PostgreSQL and Delta failures are explicit, and successful earlier publications remain usable."""
    store, run_id, bib = prepared_input
    engine = _engine(store.config)
    snapshot = store.prepare(run_id, bib, engine)
    with patch("autographrs.embeddings.search.LakebasePgVectorManager", autospec=True) as manager:
        first = store.publish(run_id, snapshot.snapshot_id)
        manager.return_value.reset_mock()
        getattr(manager.return_value, failing_step).side_effect = RuntimeError("Database unavailable")
        with pytest.raises(RuntimeError, match="Database"):
            store.publish(run_id, snapshot.snapshot_id)
        if failing_step == "setup":
            manager.return_value.publish.assert_not_called()
        failed = next(
            row for row in store.spark.tables[store.full_publications_table_name] if row["status"] == "FAILED"
        )
        with pytest.raises(ValueError, match="published"):
            store.load_publication(run_id, snapshot.snapshot_id, failed["publication_id"])
        store.load_publication(run_id, snapshot.snapshot_id, first.publication_id)
        getattr(manager.return_value, failing_step).side_effect = None
        original_save = store._save_publication

        def fail_delta(publication: SearchPublication) -> None:
            if publication.status == "COMPLETED":
                raise RuntimeError("Delta unavailable after PostgreSQL commit")
            original_save(publication)

        with (
            patch.object(store, "_save_publication", side_effect=fail_delta),
            pytest.raises(RuntimeError, match="Delta"),
        ):
            store.publish(run_id, snapshot.snapshot_id)
        retry = store.publish(run_id, snapshot.snapshot_id)
    assert retry.status == "COMPLETED"
    assert engine.client.with_options.return_value.embeddings.create.call_count == 1


def test_notebooks_and_bundle_wire_explicit_snapshot_values(prepared_input: tuple[SearchStore, str, str]) -> None:
    """Reused job tasks now set up PostgreSQL and publish pgvector indexes, not AI Search."""
    store, run_id, bib = prepared_input
    tasks = yaml.safe_load((ROOT / "resources" / "autograph_pipeline.job.yml").read_text())["resources"]["jobs"][
        "ingest_papers"
    ]["tasks"]
    keys = [task["task_key"] for task in tasks]
    assert keys == [
        "ingest_papers",
        "enrich_bibliography",
        "prepare_embeddings",
        "publish_search_index",
        "build_graph",
    ]
    for previous, task in zip(keys[:-1], tasks[1:], strict=True):
        assert task["depends_on"] == [{"task_key": previous}]
    assert not (ROOT / "resources" / "setup_search.job.yml").exists()
    assert not (ROOT / "resources" / "deployment_notebooks" / "setup_search.py").exists()
    parameters = {"env": "dev", "run_id": run_id, "bibliography_snapshot_id": bib}
    dbutils = MagicMock()
    dbutils.widgets.get.side_effect = parameters.__getitem__
    runtime = ModuleType("databricks.sdk.runtime")
    runtime.dbutils = dbutils
    original = store.prepare
    with (
        patch.dict(sys.modules, {"databricks.sdk.runtime": runtime}),
        patch("autographrs.config.load_config", return_value=store.config),
        patch("autographrs.embeddings.storage.EmbeddingStore", return_value=store),
        patch.object(store, "prepare", side_effect=lambda run, bib: original(run, bib, _engine(store.config))),
    ):
        runpy.run_path(str(ROOT / "resources" / "deployment_notebooks" / "prepare_embeddings.py"))
    snapshot_id = dbutils.jobs.taskValues.set.call_args.kwargs["value"]
    parameters["embedding_snapshot_id"] = snapshot_id
    with (
        patch.dict(sys.modules, {"databricks.sdk.runtime": runtime}),
        patch("autographrs.config.load_config", return_value=store.config),
        patch("autographrs.embeddings.search.SearchStore", return_value=store),
        patch("autographrs.embeddings.search.LakebasePgVectorManager", autospec=True) as manager,
    ):
        runpy.run_path(str(ROOT / "resources" / "deployment_notebooks" / "publish_search_index.py"))
        manager.return_value.setup.assert_called_once()
    publication_id = dbutils.jobs.taskValues.set.call_args.kwargs["value"]
    store.load_publication(run_id, snapshot_id, publication_id)
    parameters["search_publication_id"] = publication_id
    with (
        patch.dict(sys.modules, {"databricks.sdk.runtime": runtime}),
        patch("autographrs.config.load_config", return_value=store.config),
        patch("autographrs.graph.GraphStore", return_value=GraphStore(store.config, spark=store.spark)),
    ):
        runpy.run_path(str(ROOT / "resources" / "deployment_notebooks" / "build_graph.py"))
    values = {call.kwargs["key"]: call.kwargs["value"] for call in dbutils.jobs.taskValues.set.call_args_list}
    graph = GraphStore(store.config, spark=store.spark).load_graph(run_id, values["graph_build_id"])
    assert graph.embedding_snapshot_id == snapshot_id and graph.search_publication_id == publication_id
    from autographrs.assistant.orchestration import CollectionOutput

    output = CollectionOutput.model_validate_json(dbutils.notebook.exit.call_args.args[0])
    assert output.run_id == run_id and output.graph_build_id == values["graph_build_id"] and output.env == "dev"
