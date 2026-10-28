"""Tests for bibliography-only HTML extraction, external references and explicit failures."""

import io
import urllib.error
from unittest.mock import patch

import pytest
from test_enrichment import _make_paper

from autographrs.enrichment.arxiv_html import (
    ArxivHtmlCitationResolver,
    normalize_doi,
    parse_arxiv_bibliography,
)
from autographrs.enrichment.citations import resolve_mutual_citations

DOCUMENT_URL = "https://arxiv.org/html/2401.00001v2"
HTML = """
<article>
<p>Not a bibliography entry: arXiv:2000.00001
<a href="https://doi.org/10.1000/body">Code</a></p>
<section class="ltx_bibliography"><ul class="ltx_biblist">
<li class="ltx_bibitem" id="bib.one">
<span class="ltx_tag ltx_tag_bibitem">[1]</span>
<span class="ltx_bibblock">Alice &amp; Bob. A paper.
<a href="https://arxiv.org/abs/2402.00002v3">arXiv</a></span></li>
<li class="ltx_bibitem" id="bib.two">
<span class="ltx_tag ltx_tag_bibitem">[2]</span>
<span>External journal article. <a href="https://doi.org/10.1000/ABC">DOI</a>.</span></li>
<li class="ltx_bibitem" id="bib.three">
<span>Someone. A book without identifiers. Publisher, 1999.</span></li>
<li class="ltx_bibitem" id="bib.four"><span>Old paper, arXiv:hep-th/9901001v2.</span></li>
<li class="ltx_bibitem" id="bib.five"><span>Text-only DOI:10.5555/HELLO(world).</span></li>
</ul></section></article>
"""


class HtmlResponse(io.BytesIO):
    """A real context manager with the small urllib response interface used by the resolver."""

    def geturl(self) -> str:
        """Return the version-specific final URL for evidence provenance."""
        return DOCUMENT_URL


def test_extracts_all_entries_without_turning_body_mentions_into_edges() -> None:
    """External and unresolved entries survive, while body links are excluded."""
    outcome = parse_arxiv_bibliography(HTML, "2401.00001", DOCUMENT_URL)
    assert outcome.status == "ok"
    assert outcome.source == "arxiv_html"
    assert len(outcome.bibliography) == 5
    assert outcome.references == {"2402.00002", "hep-th/9901001"}
    first, external, unresolved, old, text_doi = outcome.bibliography
    assert first.label == "[1]"
    assert "Alice & Bob." in first.text
    assert first.target_id == "arxiv:2402.00002"
    assert first.evidence_url == f"{DOCUMENT_URL}#bib.one"
    assert external.target_id == "doi:10.1000/abc"
    assert unresolved.match_method == "unresolved"
    assert unresolved.target_id.startswith("ref:")
    assert old.arxiv_ids == ["hep-th/9901001"]
    assert text_doi.dois == ["10.5555/hello(world)"]
    assert all("10.1000/body" not in entry.dois for entry in outcome.bibliography)
    repeated = parse_arxiv_bibliography(HTML, "2401.00001", DOCUMENT_URL)
    assert repeated.bibliography == outcome.bibliography


def test_unresolved_references_are_source_local_not_false_global_paper_matches() -> None:
    """The same unresolved text in separate papers must not imply a shared identified work."""
    first = parse_arxiv_bibliography(HTML, "2401.00001", DOCUMENT_URL)
    second = parse_arxiv_bibliography(HTML, "2409.00009", DOCUMENT_URL)
    assert first.bibliography[2].target_id != second.bibliography[2].target_id
    assert first.bibliography[1].target_id == second.bibliography[1].target_id


def test_multiple_identifiers_remain_unresolved_instead_of_inventing_edges() -> None:
    """A merged/ambiguous entry is retained, but is not claimed as one identified destination."""
    html = '<li class="ltx_bibitem">arXiv:2402.00002 and arXiv:2403.00003</li>'
    outcome = parse_arxiv_bibliography(html, "2401.00001", DOCUMENT_URL)
    assert len(outcome.bibliography) == 1
    assert outcome.bibliography[0].match_method == "unresolved"
    assert outcome.bibliography[0].arxiv_ids == ["2402.00002", "2403.00003"]
    assert outcome.references == set()


@pytest.mark.parametrize("html", ["<html>No HTML conversion</html>", "<ul class='ltx_biblist'></ul>"])
def test_missing_bibliography_is_not_a_successful_empty_result(html: str) -> None:
    """Unknown extraction coverage is explicit even after an HTTP 200 response."""
    outcome = parse_arxiv_bibliography(html, "2401.00001", DOCUMENT_URL)
    assert outcome.status == "not_found"
    assert outcome.error
    assert not outcome.bibliography


def test_truncated_entries_are_preserved_as_partial() -> None:
    """An unclosed final item does not silently disappear."""
    outcome = parse_arxiv_bibliography(
        '<li class="ltx_bibitem"><span>Work arXiv:2402.00002', "2401.00001", DOCUMENT_URL
    )
    assert outcome.status == "partial"
    assert len(outcome.bibliography) == 1
    assert outcome.error


def test_empty_item_does_not_hide_other_entries() -> None:
    """Empty bibliography markup degrades coverage rather than dropping the whole document."""
    outcome = parse_arxiv_bibliography('<li class="ltx_bibitem"></li>' + HTML, "2401.00001", DOCUMENT_URL)
    assert outcome.status == "partial"
    assert len(outcome.bibliography) == 5


@pytest.mark.parametrize(
    ("code", "status"), [(404, "not_found"), (410, "not_found"), (429, "rate_limited"), (503, "error")]
)
def test_http_failures_do_not_fall_back_to_scholar(code: int, status: str) -> None:
    """Missing HTML and API failures have distinguishable persisted outcomes."""
    error = urllib.error.HTTPError(DOCUMENT_URL, code, "fixture", None, None)
    with patch("urllib.request.urlopen", side_effect=error) as request:
        outcome = ArxivHtmlCitationResolver(min_interval=0).fetch_references_and_citations("2401.00001")
    assert request.call_count == 1
    assert outcome.source == "arxiv_html"
    assert outcome.status == status
    assert str(code) in (outcome.error or "")
    assert not outcome.bibliography


def test_transport_failure_is_explicit() -> None:
    """Timeouts are not interpreted as an empty bibliography."""
    with patch("urllib.request.urlopen", side_effect=TimeoutError):
        outcome = ArxivHtmlCitationResolver().fetch_references_and_citations("2401.00001")
    assert outcome.status == "error"
    assert "TimeoutError" in (outcome.error or "")


def test_explicit_html_resolver_and_doi_matching_within_run() -> None:
    """The optional HTML path resolves in-run DOI identities as well as arXiv IDs."""
    papers = [_make_paper("2401.00001", "A"), _make_paper("2402.00002", "B"), _make_paper("2403.00003", "C")]
    papers[2].doi = "https://doi.org/10.1000/ABC"
    responses = [HtmlResponse(HTML.encode()), HtmlResponse(b"<html/>"), HtmlResponse(b"<html/>")]
    with (
        patch("urllib.request.urlopen", side_effect=responses) as request,
        patch("autographrs.enrichment.arxiv_html.time.sleep"),
    ):
        references, outgoing, incoming = resolve_mutual_citations(papers, citation_provider=ArxivHtmlCitationResolver())
    assert request.call_count == 3
    assert all(call.args[0].full_url.startswith("https://arxiv.org/html/") for call in request.call_args_list)
    assert references["2401.00001"] == ["2402.00002", "hep-th/9901001"]
    assert outgoing["2401.00001"] == ["2402.00002", "2403.00003"]
    assert incoming["2403.00003"] == ["2401.00001"]


def test_polite_pacing_and_invalid_identifier() -> None:
    """Requests are serially paced and input cannot change the URL host or endpoint."""
    resolver = ArxivHtmlCitationResolver()
    with (
        patch("urllib.request.urlopen", side_effect=[HtmlResponse(HTML.encode()), HtmlResponse(HTML.encode())]),
        patch("autographrs.enrichment.arxiv_html.time.monotonic", side_effect=[1.0, 2.0, 4.0]),
        patch("autographrs.enrichment.arxiv_html.time.sleep") as sleep,
    ):
        resolver.fetch_references_and_citations("2401.00001")
        resolver.fetch_references_and_citations("2402.00002")
    sleep.assert_called_once_with(2.0)
    with pytest.raises(ValueError, match="Invalid arXiv identifier"):
        resolver.fetch_references_and_citations("../other")


def test_normalize_doi_preserves_balanced_parentheses() -> None:
    """Resolver URL normalization must not destroy valid DOI suffix characters."""
    assert normalize_doi("https://doi.org/10.1000/ABC(def).") == "10.1000/abc(def)"


def test_ambiguous_doi_does_not_create_arbitrary_in_run_edges() -> None:
    """Multiple selected records with one DOI require disambiguation, including self matches."""
    papers = [_make_paper("2401.00001", "A"), _make_paper("2403.00003", "C")]
    for paper in papers:
        paper.doi = "10.1000/abc"
    with (
        patch("urllib.request.urlopen", side_effect=[HtmlResponse(HTML.encode()), HtmlResponse(b"<html/>")]),
        patch("autographrs.enrichment.arxiv_html.time.sleep"),
    ):
        _, outgoing, _ = resolve_mutual_citations(papers, citation_provider=ArxivHtmlCitationResolver())
    assert outgoing["2401.00001"] == []


def test_plain_url_reference_and_void_markup() -> None:
    """Plain URLs identify arXiv targets, and void elements do not corrupt item boundaries."""
    html = (
        '<li class="ltx_bibitem"><span>https://arxiv.org/abs/2402.00002v3</span>'
        '<br/><img src="formula.png" alt=""/>A paper.</li>'
    )
    outcome = parse_arxiv_bibliography(html, "2401.00001", DOCUMENT_URL)
    assert outcome.status == "ok"
    assert outcome.references == {"2402.00002"}
