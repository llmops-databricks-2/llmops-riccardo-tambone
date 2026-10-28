"""Regression tests for isolated, idempotent and explicitly published run snapshots."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
from test_arxiv_html import HTML, HtmlResponse
from test_enrichment import FAKE_TAXONOMY, FakeChatCompletions, FakeCitationProvider, FakeOpenAIClient, _make_paper
from test_ingestion import FakeSparkSession

from autographrs.config import ProjectConfig
from autographrs.enrichment.arxiv_html import ArxivHtmlCitationResolver
from autographrs.enrichment.tagger import PaperTagger
from autographrs.models import Paper, PaperEnrichment, SearchRequest
from autographrs.pipeline import run_ingestion
from autographrs.storage.delta import DeltaPaperStore

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "project_config.yml"


class MemorySource:
    """Source double that detects unwanted re-searches during a retry."""

    source_name = "arxiv"

    def __init__(self, papers: list[Paper]) -> None:
        self.papers = papers
        self.search_calls = 0
        self.download_calls = 0
        self.fail_download = False

    def search(self, request: SearchRequest) -> list[Paper]:
        """Return the selected snapshot in source order."""
        self.search_calls += 1
        return self.papers[: request.num_papers]

    def download_pdfs(self, papers: list[Paper], destination_dir: str | Path) -> list[Paper]:
        """Write fixture PDFs unless a failed download is requested."""
        self.download_calls += 1
        if self.fail_download:
            return papers
        directory = Path(destination_dir)
        directory.mkdir(parents=True, exist_ok=True)
        result = []
        for paper in papers:
            path = directory / f"{paper.safe_filename}.pdf"
            path.write_bytes(b"%PDF-1.4 fixture")
            result.append(paper.model_copy(update={"pdf_volume_path": str(path)}))
        return result


@pytest.fixture
def store(tmp_path: Path) -> DeltaPaperStore:
    """Create an isolated SQL-backed test double, not a remote Spark connection."""
    config = ProjectConfig.from_yaml(str(PROJECT_CONFIG), "dev")
    return DeltaPaperStore(config, spark=FakeSparkSession(), volume_root_override=tmp_path)  # type: ignore[arg-type]


def _tagger(store: DeltaPaperStore) -> PaperTagger:
    return PaperTagger(store.config, client=FakeOpenAIClient(FakeChatCompletions(["rag"], "Summary.")))


def test_new_runs_do_not_overwrite_or_cross_join(store: DeltaPaperStore) -> None:
    """The same paper in two searches has two independent, consistently joined snapshots."""
    paper = _make_paper("2401.00001", "First title")
    source = MemorySource([paper])
    with patch("autographrs.enrichment.tagger.fetch_arxiv_category_taxonomy", return_value=FAKE_TAXONOMY):
        first = run_ingestion(
            "rag",
            config=store.config,
            store=store,
            source=source,
            download_pdfs=False,
            tagger=_tagger(store),
            citation_provider=FakeCitationProvider({}),
        )
        source.papers = [paper.model_copy(update={"title": "Second title"})]
        second = run_ingestion(
            "rag",
            config=store.config,
            store=store,
            source=source,
            download_pdfs=False,
            tagger=_tagger(store),
            citation_provider=FakeCitationProvider({}),
        )
    assert first.run_id != second.run_id
    for result, title in ((first, "First title"), (second, "Second title")):
        dataset = store.load_run_dataset(result.run_id)
        assert [(p.title, e.title) for p, e in dataset] == [(title, title)]
        assert store.load_run(result.run_id).status == "COMPLETED"
    assert len(store.spark.tables[store.full_table_name]) == 2  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="only FAILED"):
        run_ingestion(
            "rag",
            config=store.config,
            store=store,
            source=source,
            download_pdfs=False,
            run_id=first.run_id,
        )


def test_retry_reuses_selection_and_does_not_duplicate_partially_written_tables(store: DeltaPaperStore) -> None:
    """A failure between table writes is hidden; retry updates the same keys and reuses PDFs."""
    first_paper = _make_paper("2401.00001", "Saved title")
    source = MemorySource([first_paper])
    save = store.save_enrichments

    def write_then_fail(enrichments: list[PaperEnrichment]) -> int:
        save(enrichments)
        raise RuntimeError("Failure after writing enrichment")

    with (
        patch.object(store, "save_enrichments", side_effect=write_then_fail),
        patch("autographrs.enrichment.tagger.fetch_arxiv_category_taxonomy", return_value=FAKE_TAXONOMY),
        pytest.raises(RuntimeError, match="Failure after writing"),
    ):
        run_ingestion(
            "rag",
            config=store.config,
            store=store,
            source=source,
            tagger=_tagger(store),
            citation_provider=FakeCitationProvider({}),
        )
    run_id = store.spark.tables[store.full_runs_table_name][0]["run_id"]  # type: ignore[attr-defined]
    failed = store.load_run(run_id)
    assert failed.status == "FAILED"
    assert failed.ingestion_status == "COMPLETED"
    assert failed.enrichment_status == "FAILED"
    for read in (store.load_papers, store.load_enrichments, store.load_run_dataset):
        with pytest.raises(ValueError, match="not ready"):
            read(run_id)
    source.papers = [_make_paper("2402.00002", "Would be a different search result")]
    with patch("autographrs.enrichment.tagger.fetch_arxiv_category_taxonomy", return_value=FAKE_TAXONOMY):
        result = run_ingestion(
            "rag",
            config=store.config,
            store=store,
            source=source,
            tagger=_tagger(store),
            citation_provider=FakeCitationProvider({}),
            run_id=run_id,
        )
    assert result.run_id == run_id
    assert source.search_calls == 1
    assert source.download_calls == 1
    assert result.papers[0].title == "Saved title"
    assert len(store.load_run_dataset(run_id)) == 1
    assert len(store.spark.tables[store.full_table_name]) == 1  # type: ignore[attr-defined]
    assert len(store.spark.tables[store.full_enrichment_table_name]) == 1  # type: ignore[attr-defined]
    assert store.load_run(run_id).attempt_id != failed.attempt_id


def test_missing_download_fails_run_and_can_be_retried(store: DeltaPaperStore) -> None:
    """A successful metadata write does not falsely publish an incomplete PDF download."""
    source = MemorySource([_make_paper("2401.00001", "Paper")])
    source.fail_download = True
    with pytest.raises(RuntimeError, match="0/1 PDFs"):
        run_ingestion("rag", config=store.config, store=store, source=source, enrich=False)
    run_id = store.spark.tables[store.full_runs_table_name][0]["run_id"]  # type: ignore[attr-defined]
    assert store.load_run(run_id).ingestion_status == "FAILED"
    source.fail_download = False
    result = run_ingestion("rag", config=store.config, store=store, source=source, enrich=False, run_id=run_id)
    assert source.search_calls == 1
    assert store.load_run(run_id).pdf_count == 1
    assert Path(result.papers[0].pdf_volume_path or "").is_file()
    assert store.load_run(run_id).enrichment_status == "SKIPPED"
    assert len(store.load_papers(run_id)) == 1
    with pytest.raises(ValueError, match="skipped"):
        store.load_enrichments(run_id)


def test_retry_rejects_changed_parameters_and_running_attempts(store: DeltaPaperStore) -> None:
    """No parallel claim, parameter change or implicit creation via a user-supplied retry ID."""
    request = SearchRequest(topic="rag")
    run = store.start_run("existing-run", request, "arxiv", False, False, retry=False)
    with pytest.raises(ValueError, match="only FAILED"):
        store.start_run(run.run_id, request, "arxiv", False, False, retry=True)
    with pytest.raises(ValueError, match="already exists"):
        store.start_run(run.run_id, request, "arxiv", False, False, retry=False)
    run.status = "FAILED"
    store.update_run(run)
    with pytest.raises(ValueError, match="parameters must match"):
        store.start_run(run.run_id, SearchRequest(topic="other"), "arxiv", False, False, retry=True)
    with pytest.raises(ValueError, match="registered run"):
        store.start_run("unknown-run", request, "arxiv", False, False, retry=True)
    active = store.start_run(run.run_id, request, "arxiv", False, False, retry=True)
    with pytest.raises(ValueError, match="not owned"):
        store.update_run(run)
    assert store.load_run(run.run_id).attempt_id == active.attempt_id


def test_upsert_deduplicates_batch_preserves_rank_and_protects_published_data(store: DeltaPaperStore) -> None:
    """Only one row per logical key is persisted; finalized runs are immutable."""
    request = SearchRequest(topic="rag")
    run = store.start_run("upsert-run", request, "arxiv", False, False, retry=False)
    a = _make_paper("2401.00001", "A")
    b = _make_paper("2402.00002", "B")
    store.save_papers([b, a, b], run.run_id, request)
    store.save_papers([b, a], run.run_id, request)
    run.paper_count = 2
    run.selection_complete = True
    run.ingestion_status = "COMPLETED"
    run.enrichment_status = "SKIPPED"
    run.status = "COMPLETED"
    store.update_run(run)
    assert [paper.paper_id for paper in store.load_papers(run.run_id)] == [b.paper_id, a.paper_id]
    assert len(store.spark.tables[store.full_table_name]) == 2  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="terminal"):
        store.save_papers([a], run.run_id, request)


def test_legacy_duplicates_are_rejected_not_silently_discarded(store: DeltaPaperStore) -> None:
    """Pre-existing duplicate keys must be repaired explicitly, not hidden by a dropDuplicates."""
    request = SearchRequest(topic="rag")
    run = store.start_run("duplicate-run", request, "arxiv", False, False, retry=False)
    paper = _make_paper("2401.00001", "A")
    store.save_papers([paper], run.run_id, request)
    rows = store.spark.tables[store.full_table_name]  # type: ignore[attr-defined]
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError, match="Duplicate historical"):
        store.save_papers([paper], run.run_id, request)


def test_empty_run_is_published_without_schema_inference(store: DeltaPaperStore) -> None:
    """No results is a valid, explicit zero-count run with empty published datasets."""
    result = run_ingestion("empty", config=store.config, store=store, source=MemorySource([]))
    assert store.load_run_dataset(result.run_id) == []
    assert store.load_run(result.run_id).paper_count == 0
    assert store.load_run(result.run_id).status == "COMPLETED"


def test_invalid_run_id_and_mixed_enrichment_writes_are_rejected(store: DeltaPaperStore) -> None:
    """Run IDs cannot escape the volume path or combine unrelated snapshots in a write."""
    with pytest.raises(ValueError, match="run_id must"):
        store.run_pdf_dir("../outside")
    enrichment = PaperEnrichment(paper_id="2401.00001", run_id="run-a", title="A")
    with pytest.raises(ValueError, match="exactly one"):
        store.save_enrichments([enrichment, enrichment.model_copy(update={"run_id": "run-b"})])


def test_failed_initial_selection_is_researched_without_old_rows(store: DeltaPaperStore) -> None:
    """If selection never reached its checkpoint, retry replaces only that unpublished selection."""
    request = SearchRequest(topic="rag")
    run = store.start_run("incomplete-selection", request, "arxiv", False, False, retry=False)
    store.save_papers([_make_paper("2401.00001", "Uncommitted selection")], run.run_id, request)
    run.status = "FAILED"
    store.update_run(run)
    source = MemorySource([_make_paper("2402.00002", "New selection")])
    result = run_ingestion(
        "rag", config=store.config, store=store, source=source, download_pdfs=False, enrich=False, run_id=run.run_id
    )
    assert [paper.paper_id for paper in store.load_papers(result.run_id)] == ["2402.00002"]


def test_legacy_enrichment_schema_gets_source_before_merge(store: DeltaPaperStore) -> None:
    """The new composite ON key must be added explicitly on pre-existing Delta tables."""
    spark = store.spark
    table = store.full_enrichment_table_name
    spark.tables[table] = [{"run_id": "legacy", "paper_id": "old-paper"}]  # type: ignore[attr-defined]
    spark.columns[table] = ["run_id", "paper_id"]  # type: ignore[attr-defined]
    with patch("autographrs.enrichment.tagger.fetch_arxiv_category_taxonomy", return_value=FAKE_TAXONOMY):
        result = run_ingestion(
            "rag",
            config=store.config,
            store=store,
            source=MemorySource([_make_paper("2401.00001", "A")]),
            download_pdfs=False,
            tagger=_tagger(store),
            citation_provider=FakeCitationProvider({}),
        )
    assert store.load_enrichments(result.run_id)[0].source == "arxiv"
    assert any(query.startswith("ALTER TABLE") for query in spark.sql_queries)  # type: ignore[attr-defined]
    assert spark.tables[table][0]["source"] is None  # type: ignore[attr-defined]
    assert spark.tables[table][0]["paper_id"] == "old-paper"  # type: ignore[attr-defined]


def test_readers_reject_missing_duplicate_or_misjoined_enrichment(store: DeltaPaperStore) -> None:
    """A completed status is necessary but does not bypass key/count integrity checks."""
    with patch("autographrs.enrichment.tagger.fetch_arxiv_category_taxonomy", return_value=FAKE_TAXONOMY):
        result = run_ingestion(
            "rag",
            config=store.config,
            store=store,
            source=MemorySource([_make_paper("2401.00001", "A")]),
            download_pdfs=False,
            tagger=_tagger(store),
            citation_provider=FakeCitationProvider({}),
        )
    table = store.full_enrichment_table_name
    rows = store.spark.tables[table]  # type: ignore[attr-defined]
    original = deepcopy(rows[0])
    rows[0]["paper_id"] = "different-paper"
    with pytest.raises(ValueError, match="keys do not match"):
        store.load_run_dataset(result.run_id)
    rows[0] = original
    rows.append(deepcopy(original))
    with pytest.raises(ValueError, match="Duplicate enrichment keys"):
        store.load_enrichments(result.run_id)
    rows.clear()
    with pytest.raises(ValueError, match="count is inconsistent"):
        store.load_enrichments(result.run_id)


def test_citation_failure_is_published_with_explicit_warning(store: DeltaPaperStore) -> None:
    """Unavailable citation enrichment is not confused with a pipeline storage failure."""
    from autographrs.enrichment.citations import CitationLookup

    provider = FakeCitationProvider({})
    with (
        patch.object(provider, "fetch_references_and_citations", return_value=CitationLookup(status="rate_limited")),
        patch("autographrs.enrichment.tagger.fetch_arxiv_category_taxonomy", return_value=FAKE_TAXONOMY),
    ):
        result = run_ingestion(
            "rag",
            config=store.config,
            store=store,
            source=MemorySource([_make_paper("2401.00001", "A")]),
            download_pdfs=False,
            tagger=_tagger(store),
            citation_provider=provider,
        )
    run = store.load_run(result.run_id)
    assert run.status == "COMPLETED"
    assert run.warnings == ["2401.00001: citation lookup rate_limited"]
    assert store.load_run_dataset(result.run_id)[0][1].citation_status == "rate_limited"


def test_html_bibliography_survives_storage_and_scoped_read(store: DeltaPaperStore) -> None:
    """The explicit HTML provider persists references outside arXiv/the run."""
    with (
        patch("urllib.request.urlopen", return_value=HtmlResponse(HTML.encode())),
        patch("autographrs.enrichment.tagger.fetch_arxiv_category_taxonomy", return_value=FAKE_TAXONOMY),
    ):
        result = run_ingestion(
            "rag",
            config=store.config,
            store=store,
            source=MemorySource([_make_paper("2401.00001", "A")]),
            download_pdfs=False,
            tagger=_tagger(store),
            citation_provider=ArxivHtmlCitationResolver(),
        )
    paper, enrichment = store.load_run_dataset(result.run_id)[0]
    assert enrichment.citation_source == "arxiv_html"
    assert enrichment.citation_status == "ok"
    assert enrichment.cites_in_set == []
    assert enrichment.bibliography is not None
    assert len(enrichment.bibliography) == 5
    assert enrichment.bibliography[1].target_id == "doi:10.1000/abc"
    assert enrichment.bibliography[2].match_method == "unresolved"
    assert paper.cited_paper_ids == ["2402.00002", "hep-th/9901001"]
    assert "bibliography ARRAY<STRUCT<" in store.spark.schemas[store.full_enrichment_table_name]  # type: ignore[attr-defined]
    assert store.load_run(result.run_id).enrichment_count == 1
    store.spark.tables[store.full_enrichment_table_name][0]["bibliography"] = None  # type: ignore[attr-defined]
    assert store.load_enrichments(result.run_id)[0].bibliography is None
