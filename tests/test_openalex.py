"""Regression coverage for OpenAlex identity resolution, external references and failures."""

import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest
from test_enrichment import FAKE_TAXONOMY, FakeChatCompletions, FakeOpenAIClient, _make_paper
from test_ingestion import FakeSparkSession
from test_run_storage import MemorySource

from autographrs.config import ProjectConfig
from autographrs.enrichment.citations import resolve_mutual_citations
from autographrs.enrichment.openalex import OpenAlexCitationResolver
from autographrs.enrichment.tagger import PaperTagger
from autographrs.pipeline import run_ingestion
from autographrs.storage.delta import DeltaPaperStore


def _work(
    identifier: str,
    arxiv_id: str | None = None,
    references: list[str] | None = None,
    doi: str | None = None,
) -> dict[str, object]:
    return {
        "id": f"https://openalex.org/{identifier}",
        "title": f"Title {identifier}",
        "doi": doi,
        "publication_year": 2024,
        "authorships": [{"author": {"display_name": "Alice"}}],
        "locations": [{"landing_page_url": f"https://arxiv.org/abs/{arxiv_id}"}] if arxiv_id else [],
        "referenced_works": references or [],
    }


def _response(works: list[dict[str, object]]) -> io.BytesIO:
    return io.BytesIO(json.dumps({"results": works}).encode())


def test_default_provider_preserves_external_metadata_and_openalex_edges() -> None:
    """Two selected papers can link by provider ID even if the cited metadata has no arXiv URL."""
    papers = [_make_paper("2401.00001", "A"), _make_paper("2402.00002", "B")]
    first = _work("W1", papers[0].paper_id, ["https://openalex.org/W2", "https://openalex.org/W3"])
    external = _work("W3", doi="https://doi.org/10.1000/external")
    outcomes = {}
    with (
        patch(
            "urllib.request.urlopen",
            side_effect=[
                _response([first]),
                _response([_work("W2"), external]),
                _response([_work("W2", papers[1].paper_id)]),
            ],
        ) as request,
        patch("autographrs.enrichment.openalex.time.sleep"),
    ):
        _, outgoing, incoming = resolve_mutual_citations(
            papers, citation_provider=OpenAlexCitationResolver(), lookup_outcomes=outcomes
        )
    assert outgoing[papers[0].paper_id] == [papers[1].paper_id]
    assert incoming[papers[1].paper_id] == [papers[0].paper_id]
    assert outcomes[papers[0].paper_id].source == "openalex"
    reference = outcomes[papers[0].paper_id].bibliography[1]
    assert reference.target_id == "openalex:W3"
    assert reference.title == "Title W3"
    assert reference.authors == ["Alice"]
    assert reference.publication_year == 2024
    assert reference.dois == ["10.1000/external"]
    assert reference.evidence_url == first["id"]
    assert all(call.args[0].full_url.startswith("https://api.openalex.org/works?") for call in request.call_args_list)


def test_identity_requires_exact_location_not_title_similarity() -> None:
    """A search response containing an unrelated work is not accepted as a matching source."""
    with patch("urllib.request.urlopen", return_value=_response([_work("W1", "9999.00001")])) as request:
        result = OpenAlexCitationResolver().fetch_references_and_citations("2401.00001")
    assert result.status == "not_found"
    assert result.bibliography == []
    assert request.call_count == 1


def test_doi_fallback_is_exact_and_uses_supplied_paper_metadata() -> None:
    """DOI fallback does not require title search or an arXiv location on the published work."""
    paper = _make_paper("2401.00001", "A").model_copy(update={"doi": "10.1000/ABC"})
    with (
        patch(
            "urllib.request.urlopen",
            side_effect=[_response([]), _response([_work("W1", doi="https://doi.org/10.1000/abc")])],
        ) as request,
        patch("autographrs.enrichment.openalex.time.sleep"),
    ):
        result = OpenAlexCitationResolver(papers=[paper]).fetch_references_and_citations(paper.paper_id)
    assert result.status == "partial"
    assert result.provider_paper_id == "https://openalex.org/W1"
    assert "doi:10.1000/abc" in parse_qs(urlsplit(request.call_args.args[0].full_url).query)["filter"][0]


def test_missing_cited_metadata_keeps_every_reference_id() -> None:
    """Missing metadata does not remove graph destinations."""
    citing = _work("W1", "2401.00001", ["https://openalex.org/W2", "https://openalex.org/W3"])
    with (
        patch("urllib.request.urlopen", side_effect=[_response([citing]), _response([_work("W2")])]),
        patch("autographrs.enrichment.openalex.time.sleep"),
    ):
        result = OpenAlexCitationResolver().fetch_references_and_citations("2401.00001")
    assert result.status == "partial"
    assert len(result.bibliography) == 2
    assert result.bibliography[1].target_id == "openalex:W3"
    assert result.bibliography[1].title is None
    assert result.error


def test_rate_limit_during_hydration_preserves_edges_and_redacts_key(caplog: pytest.LogCaptureFixture) -> None:
    """A secret-bearing request URL must never be logged or persisted in error fields."""
    citing = _work("W1", "2401.00001", ["https://openalex.org/W2"])
    error = urllib.error.HTTPError("https://api.openalex.org/works?api_key=SECRET", 429, "SECRET", None, None)
    with (
        patch("urllib.request.urlopen", side_effect=[_response([citing]), error]),
        patch("autographrs.enrichment.openalex.time.sleep"),
    ):
        result = OpenAlexCitationResolver(api_key="SECRET", max_attempts=1).fetch_references_and_citations("2401.00001")
    assert result.status == "partial"
    assert "429" in (result.error or "")
    assert "SECRET" not in result.model_dump_json()
    assert "SECRET" not in caplog.text
    assert result.bibliography[0].target_id == "openalex:W2"


@pytest.mark.parametrize("code,status", [(401, "error"), (403, "error"), (404, "not_found"), (429, "rate_limited")])
def test_source_http_errors_are_not_successful_empty_bibliographies(code: int, status: str) -> None:
    """Source lookup failures expose status instead of reporting an empty success."""
    error = urllib.error.HTTPError("https://api.openalex.org/works", code, "", None, None)
    with patch("urllib.request.urlopen", side_effect=error):
        result = OpenAlexCitationResolver(max_attempts=1).fetch_references_and_citations("2401.00001")
    assert result.status == status
    assert result.error


def test_retry_after_is_honored_within_budget() -> None:
    """A bounded retry recovers an initial 429 without silently skipping the paper."""
    error = urllib.error.HTTPError("https://api.openalex.org/works", 429, "", {"Retry-After": "2"}, None)
    with (
        patch("urllib.request.urlopen", side_effect=[error, _response([_work("W1", "2401.00001")])]),
        patch("autographrs.enrichment.openalex.time.sleep") as sleep,
    ):
        result = OpenAlexCitationResolver(min_interval=0).fetch_references_and_citations("2401.00001")
    assert result.status == "partial"
    assert any(call.args == (2.0,) for call in sleep.call_args_list)


def test_large_reference_sets_are_batched_and_keep_all_destinations() -> None:
    """Reference hydration does not truncate at one API page; shared IDs are deduplicated."""
    works = [_work(f"W{index}") for index in range(2, 104)]
    identifiers = [str(work["id"]) for work in works]
    citing = _work("W1", "2401.00001", identifiers + identifiers[:1])
    with (
        patch(
            "urllib.request.urlopen",
            side_effect=[_response([citing]), _response(works[:100]), _response(works[100:])],
        ) as request,
        patch("autographrs.enrichment.openalex.time.sleep"),
    ):
        result = OpenAlexCitationResolver().fetch_references_and_citations("2401.00001")
    assert result.status == "ok"
    assert len(result.bibliography) == 102
    assert request.call_count == 3
    assert [entry.position for entry in result.bibliography] == list(range(1, 103))


def test_ambiguous_source_and_malformed_payload_fail_explicitly() -> None:
    """Ambiguity is not resolved by selecting the first result."""
    with patch(
        "urllib.request.urlopen", return_value=_response([_work("W1", "2401.00001"), _work("W2", "2401.00001")])
    ):
        result = OpenAlexCitationResolver().fetch_references_and_citations("2401.00001")
    assert result.status == "error"
    with patch("urllib.request.urlopen", return_value=io.BytesIO(b'{"unexpected": true}')):
        result = OpenAlexCitationResolver().fetch_references_and_citations("2401.00001")
    assert result.status == "error"


def test_openalex_metadata_survives_run_publication(tmp_path: Path) -> None:
    """The actual pipeline writes and reads the new nested metadata using explicit Spark types."""
    config = ProjectConfig.from_yaml(str(Path(__file__).resolve().parents[1] / "project_config.yml"), "dev")
    store = DeltaPaperStore(config, spark=FakeSparkSession(), volume_root_override=tmp_path)  # type: ignore[arg-type]
    citing = _work("W1", "2401.00001", ["https://openalex.org/W2"])
    with (
        patch("urllib.request.urlopen", side_effect=[_response([citing]), _response([_work("W2")])]),
        patch("autographrs.enrichment.openalex.time.sleep"),
        patch("autographrs.enrichment.tagger.fetch_arxiv_category_taxonomy", return_value=FAKE_TAXONOMY),
    ):
        tagger = PaperTagger(store.config, client=FakeOpenAIClient(FakeChatCompletions(["tag"], "Summary")))
        result = run_ingestion(
            "topic",
            config=store.config,
            store=store,
            source=MemorySource([_make_paper("2401.00001", "A")]),
            download_pdfs=False,
            tagger=tagger,
            citation_provider=OpenAlexCitationResolver(),
        )
    enrichment = store.load_enrichments(result.run_id)[0]
    assert enrichment.citation_source == "openalex"
    assert enrichment.bibliography is not None
    assert enrichment.bibliography[0].authors == ["Alice"]
    assert enrichment.bibliography[0].target_id == "openalex:W2"
    assert store.load_run(result.run_id).status == "COMPLETED"


def test_http_arxiv_locations_are_queried_and_verified() -> None:
    """Regression: real unmatched records use http arXiv URLs, not https."""
    citing = _work("W1", "2412.05830", ["https://openalex.org/W2"])
    citing["locations"] = [{"landing_page_url": "http://arxiv.org/abs/2412.05830"}]
    with (
        patch("urllib.request.urlopen", side_effect=[_response([citing]), _response([_work("W2")])]) as request,
        patch("autographrs.enrichment.openalex.time.sleep"),
    ):
        outcome = OpenAlexCitationResolver().fetch_references_and_citations("2412.05830")
    query = parse_qs(urlsplit(request.call_args_list[0].args[0].full_url).query)["filter"][0]
    assert "http://arxiv.org/abs/2412.05830" in query
    assert "https://arxiv.org/abs/2412.05830" in query
    assert outcome.status == "ok"
    assert len(outcome.bibliography) == 1


def test_arxiv_doi_location_is_an_exact_identifier() -> None:
    """DataCite arXiv DOI locations resolve the source without a title match."""
    citing = _work("W1", references=["https://openalex.org/W2"])
    citing["locations"] = [{"landing_page_url": "https://doi.org/10.48550/arXiv.2410.12096"}]
    cited = _work("W2", doi="https://doi.org/10.48550/arxiv.2402.00002")
    with (
        patch("urllib.request.urlopen", side_effect=[_response([citing]), _response([cited])]),
        patch("autographrs.enrichment.openalex.time.sleep"),
    ):
        outcome = OpenAlexCitationResolver().fetch_references_and_citations("2410.12096")
    assert outcome.status == "ok"
    assert outcome.references == {"2402.00002"}


def test_zero_indexed_references_is_unknown_coverage_not_verified_empty_bibliography() -> None:
    """A found work with no indexed edges must emit a publication warning."""
    with patch("urllib.request.urlopen", return_value=_response([_work("W1", "2410.12096")])):
        outcome = OpenAlexCitationResolver().fetch_references_and_citations("2410.12096")
    assert outcome.status == "partial"
    assert outcome.provider_paper_id == "https://openalex.org/W1"
    assert outcome.bibliography == []
    assert "coverage is unknown" in (outcome.error or "")


def test_non_arxiv_papers_resolve_by_doi_and_link_by_provider_identity() -> None:
    """Source-local IDs must never be treated as arXiv identifiers."""
    papers = [
        _make_paper("12345", "PubMed A").model_copy(
            update={"source": "pubmed", "doi": "10.1000/a", "html_url": "https://pubmed.ncbi.nlm.nih.gov/12345/"}
        ),
        _make_paper("67890", "PubMed B").model_copy(
            update={"source": "pubmed", "doi": "10.1000/b", "html_url": "https://pubmed.ncbi.nlm.nih.gov/67890/"}
        ),
    ]
    first = _work("W1", references=["https://openalex.org/W2"], doi="https://doi.org/10.1000/a")
    second = _work("W2", doi="https://doi.org/10.1000/b")
    with (
        patch(
            "urllib.request.urlopen",
            side_effect=[_response([first]), _response([second]), _response([second])],
        ) as request,
        patch("autographrs.enrichment.openalex.time.sleep"),
    ):
        _, outgoing, incoming = resolve_mutual_citations(papers, citation_provider=OpenAlexCitationResolver())
    assert outgoing["12345"] == ["67890"]
    assert incoming["67890"] == ["12345"]
    filters = [parse_qs(urlsplit(call.args[0].full_url).query)["filter"][0] for call in request.call_args_list]
    assert filters[0] == "doi:10.1000/a"
    assert all("arxiv.org/abs/" not in value for value in filters)


def test_unknown_repository_can_resolve_by_exact_landing_url_without_doi() -> None:
    """A new PaperSource can participate without changes to OpenAlex repository-specific logic."""
    paper = _make_paper("local-id", "Repository paper").model_copy(
        update={"source": "institutional", "html_url": "https://repository.example.edu/items/local-id"}
    )
    work = _work("W1")
    work["locations"] = [{"landing_page_url": paper.html_url}]
    with patch("urllib.request.urlopen", return_value=_response([work])) as request:
        outcome = OpenAlexCitationResolver().fetch_for_paper(paper)
    assert outcome.provider_paper_id == "https://openalex.org/W1"
    query = parse_qs(urlsplit(request.call_args.args[0].full_url).query)["filter"][0]
    assert query == f"locations.landing_page_url:{paper.html_url}"


def test_non_arxiv_pipeline_does_not_fetch_arxiv_taxonomy(tmp_path: Path) -> None:
    """Generic ingestion/enrichment keeps source categories separate from arxiv_category."""
    config = ProjectConfig.from_yaml(str(Path(__file__).resolve().parents[1] / "project_config.yml"), "dev")
    store = DeltaPaperStore(config, spark=FakeSparkSession(), volume_root_override=tmp_path)  # type: ignore[arg-type]
    paper = _make_paper("12345", "PubMed paper").model_copy(
        update={
            "source": "pubmed",
            "doi": "10.1000/a",
            "categories": ["Medicine"],
            "html_url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
        }
    )
    source = MemorySource([paper])
    source.source_name = "pubmed"
    citing = _work("W1", references=["https://openalex.org/W2"], doi="https://doi.org/10.1000/a")
    with (
        patch("urllib.request.urlopen", side_effect=[_response([citing]), _response([_work("W2")])]),
        patch("autographrs.enrichment.openalex.time.sleep"),
        patch(
            "autographrs.enrichment.tagger.fetch_arxiv_category_taxonomy",
            side_effect=AssertionError("arXiv taxonomy must not be used"),
        ) as taxonomy,
    ):
        tagger = PaperTagger(config, client=FakeOpenAIClient(FakeChatCompletions(["Medicine"], "Summary")))
        result = run_ingestion(
            "topic",
            config=config,
            store=store,
            source=source,
            download_pdfs=False,
            tagger=tagger,
            citation_provider=OpenAlexCitationResolver(),
        )
    taxonomy.assert_not_called()
    persisted, enrichment = store.load_run_dataset(result.run_id)[0]
    assert persisted.source == enrichment.source == "pubmed"
    assert persisted.categories == ["Medicine"]
    assert enrichment.arxiv_category == []
    assert enrichment.tags == ["medicine"]
    assert enrichment.bibliography is not None
    assert enrichment.bibliography[0].target_id == "openalex:W2"
