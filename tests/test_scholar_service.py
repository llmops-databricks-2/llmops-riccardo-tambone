"""Scholar open-graph references, bounded retry budgets and source-independent fallback."""

import json
import urllib.error
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch

from test_enrichment import FAKE_TAXONOMY, FakeChatCompletions, FakeOpenAIClient, _make_paper
from test_ingestion import FakeSparkSession
from test_run_storage import MemorySource

from autographrs.config import ProjectConfig
from autographrs.enrichment.citation_service import CitationService
from autographrs.enrichment.citations import CitationLookup, SemanticScholarCitationResolver, fetch_bibliographies
from autographrs.enrichment.openalex import OpenAlexCitationResolver
from autographrs.enrichment.tagger import PaperTagger
from autographrs.models import BibliographicReference
from autographrs.pipeline import run_ingestion
from autographrs.storage.delta import DeltaPaperStore


def _payload() -> list[dict[str, object]]:
    return [
        {
            "paperId": "source-a",
            "referenceCount": 4,
            "references": [
                {
                    "paperId": "outside",
                    "title": "External article",
                    "authors": [{"name": "Alice"}],
                    "year": 2020,
                    "externalIds": {"DOI": "10.1000/ABC"},
                },
                {"paperId": None, "title": "DOI-only article", "externalIds": {"DOI": "10.1000/other"}},
                {"paperId": None, "title": "Book with no identifier"},
                None,
            ],
        }
    ]


def _response(payload: object) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    return response


def test_scholar_retains_external_doi_and_unresolved_references() -> None:
    """Every returned reference survives, even without a Scholar ID, DOI or arXiv identifier."""
    resolver = SemanticScholarCitationResolver()
    with patch.object(resolver, "_request_json", return_value=_payload()):
        result = resolver.fetch_batch(["2410.12096"])["2410.12096"]
    assert len(result.bibliography) == 4
    article, doi_only, book, null = result.bibliography
    assert article.target_id == "scholar:outside"
    assert article.title == "External article"
    assert article.authors == ["Alice"]
    assert article.publication_year == 2020
    assert article.dois == ["10.1000/abc"]
    assert doi_only.target_id == "doi:10.1000/other"
    assert book.title == "Book with no identifier"
    assert book.match_method == null.match_method == "unresolved"
    assert result.status == "partial"
    assert result.references == set()


def test_batch_generic_sources_select_doi_pmid_and_url_without_arxiv_assumptions() -> None:
    """One HTTP batch covers multiple identifier types instead of individual retry loops."""
    papers = [
        _make_paper("local", "A").model_copy(update={"source": "other", "doi": "10.1000/a"}),
        _make_paper("12345", "B").model_copy(update={"source": "pubmed"}),
        _make_paper("another", "C").model_copy(
            update={"source": "repository", "html_url": "https://repository.example.org/item/another"}
        ),
    ]
    resolver = SemanticScholarCitationResolver()
    with patch.object(resolver, "_request_json", return_value=[None] * 3) as request:
        resolver.fetch_papers(papers)
    assert request.call_count == 1
    assert request.call_args.kwargs["body"] == {
        "ids": ["DOI:10.1000/a", "PMID:12345", "URL:https://repository.example.org/item/another"]
    }


def test_429_exponential_jitter_and_shared_wait_budget() -> None:
    """Without Retry-After use meaningful cooldowns; stop before exceeding the shared budget."""
    resolver = SemanticScholarCitationResolver(min_interval=0, max_attempts=5, retry_wait_budget=25)
    error = urllib.error.HTTPError("https://example.org", 429, "", Message(), None)
    with (
        patch("urllib.request.urlopen", side_effect=error) as request,
        patch("autographrs.enrichment.citations.random.uniform", return_value=0.5),
        patch("autographrs.enrichment.citations.time.sleep") as sleep,
    ):
        outcome = resolver.fetch_batch(["2410.12096"])["2410.12096"]
    assert outcome.status == "rate_limited"
    assert request.call_count == 2
    assert [call.args[0] for call in sleep.call_args_list if call.args[0] > 0] == [10.5]


def test_retry_budget_is_shared_across_pagination_requests() -> None:
    """A fresh page does not reset the total retry sleep budget."""
    resolver = SemanticScholarCitationResolver(min_interval=0, retry_wait_budget=15)
    error = urllib.error.HTTPError("https://example.org", 429, "", Message(), None)
    payload = [{"paperId": "a", "referenceCount": 2, "references": []}]
    page = {"data": [{"citedPaper": {"paperId": "b", "title": "B"}}], "next": 1}
    with (
        patch("urllib.request.urlopen", side_effect=[error, _response(payload), _response(page), error]) as request,
        patch("autographrs.enrichment.citations.random.uniform", return_value=0),
        patch("autographrs.enrichment.citations.time.sleep"),
    ):
        outcome = resolver.fetch_batch(["2410.12096"])["2410.12096"]
    assert request.call_count == 4
    assert outcome.status == "partial"
    assert outcome.bibliography[0].target_id == "scholar:b"


def test_exhausted_batch_does_not_retry_for_each_remaining_paper() -> None:
    """A run with 501 papers stops after one exhausted batch, not 501 retry sequences."""
    resolver = SemanticScholarCitationResolver(max_attempts=2, min_interval=0)
    error = urllib.error.HTTPError("https://example.org", 429, "", Message(), None)
    with (
        patch("urllib.request.urlopen", side_effect=error) as request,
        patch("autographrs.enrichment.citations.time.sleep"),
    ):
        outcomes = resolver.fetch_batch([f"2401.{index:05d}" for index in range(501)])
    assert request.call_count == 2
    assert len(outcomes) == 501
    assert all(outcome.status == "rate_limited" for outcome in outcomes.values())


def test_partial_pagination_preserves_inline_data_and_blocks_remaining_pages() -> None:
    """After one exhausted pagination failure the other paper keeps its inline references."""
    resolver = SemanticScholarCitationResolver()
    payload = [
        {"paperId": "a", "referenceCount": 2, "references": [{"paperId": "b", "title": "B"}]},
        {"paperId": "c", "referenceCount": 2, "references": [{"paperId": "d", "title": "D"}]},
    ]
    error = urllib.error.HTTPError("https://example.org", 429, "", Message(), None)
    with patch.object(resolver, "_request_json", side_effect=[payload, error]) as request:
        outcomes = resolver.fetch_batch(["2401.00001", "2402.00002"])
    assert request.call_count == 2
    assert outcomes["2401.00001"].bibliography[0].title == "B"
    assert outcomes["2402.00002"].bibliography[0].title == "D"
    assert all(outcome.status == "partial" for outcome in outcomes.values())


def _external_lookup() -> CitationLookup:
    return CitationLookup(
        source="openalex",
        provider_paper_id="https://openalex.org/W1",
        bibliography=[
            BibliographicReference(
                reference_id="W2",
                position=1,
                text="External",
                title="External",
                target_id="openalex:W2",
                match_method="openalex_id",
                evidence_url="https://openalex.org/W1",
            )
        ],
    )


def test_fallback_preserves_both_outcomes_and_never_restarts_scholar() -> None:
    """Fallback is per empty bibliography, not a second Scholar attempt per source paper."""
    papers = [_make_paper("2401.00001", "A"), _make_paper("2402.00002", "B")]
    service = CitationService()
    limited = CitationLookup(status="rate_limited", error="Semantic Scholar HTTP 429")
    with (
        patch.object(
            service.scholar, "fetch_papers", return_value={paper.paper_id: limited for paper in papers}
        ) as primary,
        patch.object(service.openalex, "fetch_for_paper", return_value=_external_lookup()) as secondary,
    ):
        results = service.fetch_papers(papers)
    assert primary.call_count == 1
    assert secondary.call_count == 2
    outcome = results[papers[0].paper_id]
    assert outcome.source == "openalex"
    assert [(report.source, report.status) for report in outcome.attempts] == [
        ("semantic_scholar", "rate_limited"),
        ("openalex", "ok"),
    ]
    assert outcome.bibliography[0].title == "External"


def test_useful_scholar_data_is_not_overwritten_or_queried_twice() -> None:
    """Even a partial but useful Scholar result remains authoritative in this fallback design."""
    service = CitationService()
    paper = _make_paper("2401.00001", "A")
    result = _external_lookup().model_copy(update={"source": "semantic_scholar", "status": "partial"})
    with (
        patch.object(service.scholar, "fetch_papers", return_value={paper.paper_id: result}),
        patch.object(service.openalex, "fetch_for_paper") as fallback,
    ):
        outcome = service.fetch_papers([paper])[paper.paper_id]
    fallback.assert_not_called()
    assert outcome.source == "semantic_scholar"
    assert len(outcome.attempts) == 1


def test_default_pipeline_publishes_scholar_bibliography_and_provider_diagnostics(tmp_path: Path) -> None:
    """The deployed default preserves structured data and reports failed primary lookups."""
    config = ProjectConfig.from_yaml(str(Path(__file__).resolve().parents[1] / "project_config.yml"), "dev")
    store = DeltaPaperStore(config, spark=FakeSparkSession(), volume_root_override=tmp_path)  # type: ignore[arg-type]
    paper = _make_paper("2401.00001", "A")
    with (
        patch.object(SemanticScholarCitationResolver, "_request_json", return_value=_payload()),
        patch.object(OpenAlexCitationResolver, "fetch_for_paper") as fallback,
        patch("autographrs.enrichment.tagger.fetch_arxiv_category_taxonomy", return_value=FAKE_TAXONOMY),
    ):
        result = run_ingestion(
            "topic",
            config=config,
            store=store,
            source=MemorySource([paper]),
            download_pdfs=False,
            tagger=PaperTagger(config, client=FakeOpenAIClient(FakeChatCompletions(["tag"], "Summary"))),
        )
    fallback.assert_not_called()
    enrichment = store.load_enrichments(result.run_id)[0]
    assert enrichment.citation_source == "semantic_scholar"
    assert enrichment.bibliography is not None
    assert len(enrichment.bibliography) == 4
    assert enrichment.citation_lookups is not None
    assert enrichment.citation_lookups[0].status == "partial"
    assert enrichment.citation_paper_ids == ["https://www.semanticscholar.org/paper/source-a"]
    assert store.load_run(result.run_id).warnings


def test_scholar_provider_ids_preserved_for_generic_sources() -> None:
    """Non-arXiv identities and references are preserved without treating them as arXiv IDs."""
    papers = [
        _make_paper("one", "A").model_copy(update={"source": "other", "doi": "10.1000/a"}),
        _make_paper("two", "B").model_copy(update={"source": "other", "doi": "10.1000/b"}),
    ]
    payload = [
        {"paperId": "a", "referenceCount": 1, "references": [{"paperId": "b"}]},
        {"paperId": "b", "referenceCount": 0, "references": []},
    ]
    with patch.object(SemanticScholarCitationResolver, "_request_json", return_value=payload):
        outcomes = fetch_bibliographies(papers, SemanticScholarCitationResolver())
    assert outcomes["one"].references == set()
    assert outcomes["one"].bibliography[0].target_id == "scholar:b"
    assert outcomes["two"].provider_paper_id == "https://www.semanticscholar.org/paper/b"


def test_retry_after_http_date_exceeds_budget_without_early_retry() -> None:
    """HTTP-date cooldowns cannot be shortened to fit a configured wait budget."""
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    headers = Message()
    headers["Retry-After"] = format_datetime(datetime.now(UTC) + timedelta(minutes=5), usegmt=True)
    error = urllib.error.HTTPError("https://example.org", 429, "", headers, None)
    with patch("urllib.request.urlopen", side_effect=error) as request:
        outcome = SemanticScholarCitationResolver().fetch_batch(["2401.00001"])["2401.00001"]
    assert request.call_count == 1
    assert outcome.status == "rate_limited"


def test_disabled_scholar_does_not_trigger_fallback() -> None:
    """Explicit opt-out must not cause another provider to make network calls."""
    service = CitationService(scholar=SemanticScholarCitationResolver(enabled=False))
    with patch.object(service.openalex, "fetch_for_paper") as fallback:
        outcome = service.fetch_papers([_make_paper("2401.00001", "A")])["2401.00001"]
    fallback.assert_not_called()
    assert outcome.status == "disabled"


def test_failed_fallback_remains_visible_without_success_shaped_result() -> None:
    """Neither an exhausted primary nor an unsuccessful secondary is hidden."""
    service = CitationService()
    with (
        patch.object(
            service.scholar,
            "fetch_papers",
            return_value={"2401.00001": CitationLookup(status="rate_limited", error="Semantic Scholar HTTP 429")},
        ),
        patch.object(
            service.openalex,
            "fetch_for_paper",
            return_value=CitationLookup(source="openalex", status="not_found", error="No matching work"),
        ),
    ):
        outcome = service.fetch_papers([_make_paper("2401.00001", "A")])["2401.00001"]
    assert outcome.status == "rate_limited"
    assert outcome.bibliography == []
    assert [report.status for report in outcome.attempts] == ["rate_limited", "not_found"]


def test_inline_references_are_not_lost_after_partial_pagination_failure() -> None:
    """Retain inline evidence even if it is absent from the first successful reference page."""
    resolver = SemanticScholarCitationResolver()
    payload = [{"paperId": "a", "referenceCount": 3, "references": [{"paperId": "inline", "title": "Inline"}]}]
    page = {"data": [{"citedPaper": {"paperId": "page", "title": "Page"}}], "next": 1}
    error = urllib.error.HTTPError("https://example.org", 429, "", Message(), None)
    with patch.object(resolver, "_request_json", side_effect=[payload, page, error]):
        outcome = resolver.fetch_batch(["2401.00001"])["2401.00001"]
    assert {entry.target_id for entry in outcome.bibliography} == {"scholar:inline", "scholar:page"}
    assert outcome.status == "partial"


def test_fallback_success_persists_primary_error_and_publication_warning(tmp_path: Path) -> None:
    """A positive fallback must not erase the primary provider failure in stored run diagnostics."""
    config = ProjectConfig.from_yaml(str(Path(__file__).resolve().parents[1] / "project_config.yml"), "dev")
    store = DeltaPaperStore(config, spark=FakeSparkSession(), volume_root_override=tmp_path)  # type: ignore[arg-type]
    with (
        patch.object(
            SemanticScholarCitationResolver,
            "fetch_papers",
            return_value={"2401.00001": CitationLookup(status="rate_limited", error="Semantic Scholar HTTP 429")},
        ),
        patch.object(OpenAlexCitationResolver, "fetch_for_paper", return_value=_external_lookup()),
        patch("autographrs.enrichment.tagger.fetch_arxiv_category_taxonomy", return_value=FAKE_TAXONOMY),
    ):
        result = run_ingestion(
            "topic",
            config=config,
            store=store,
            source=MemorySource([_make_paper("2401.00001", "A")]),
            download_pdfs=False,
            tagger=PaperTagger(config, client=FakeOpenAIClient(FakeChatCompletions(["tag"], "Summary"))),
        )
    run = store.load_run(result.run_id)
    enrichment = store.load_enrichments(result.run_id)[0]
    assert run.status == "COMPLETED"
    assert enrichment.citation_status == "ok"
    assert enrichment.citation_source == "openalex"
    assert enrichment.citation_lookups is not None
    assert enrichment.citation_lookups[0].error == "Semantic Scholar HTTP 429"
    assert any("semantic_scholar rate_limited" in warning for warning in run.warnings)
