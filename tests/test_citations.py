"""Offline regression tests for the Semantic Scholar wire format and failures."""

import json
import urllib.error
from email.message import Message
from unittest.mock import MagicMock, patch

import pytest
from test_enrichment import _make_paper

from autographrs.enrichment.citations import SemanticScholarCitationResolver, resolve_mutual_citations


def test_batch_matches_scholar_ids_and_derives_incoming_links() -> None:
    """Provider identity matching works even if a reference has no arXiv external ID."""
    resolver = SemanticScholarCitationResolver(min_interval=0)
    response = [
        {"paperId": "s2-a", "referenceCount": 0, "references": []},
        {
            "paperId": "s2-b",
            "referenceCount": 3,
            "references": [
                {"paperId": "s2-a"},
                {"paperId": "external", "externalIds": {"ArXiv": "2301.00001v2"}},
                {"paperId": None, "externalIds": None},
            ],
        },
    ]
    papers = [_make_paper("2401.00001", "Paper A"), _make_paper("2402.00002", "Paper B")]
    with patch.object(resolver, "_request_json", return_value=response) as request:
        refs, outgoing, incoming = resolve_mutual_citations(papers, resolver)
    assert request.call_count == 1
    assert request.call_args.kwargs["body"] == {"ids": ["ARXIV:2401.00001", "ARXIV:2402.00002"]}
    assert refs["2402.00002"] == ["2301.00001", "2401.00001"]
    assert outgoing["2402.00002"] == ["2401.00001"]
    assert outgoing["2401.00001"] == []
    assert incoming["2401.00001"] == ["2402.00002"]


def test_reference_pagination_reads_all_pages() -> None:
    """Truncated inline lists are completed using citedPaper entries and next offsets."""
    resolver = SemanticScholarCitationResolver(min_interval=0)
    responses = [
        [{"paperId": "s2-a", "referenceCount": 2, "references": []}],
        {"data": [{"citedPaper": {"paperId": "s2-b", "externalIds": {"ArXiv": "2402.00002"}}}], "next": 1},
        {"data": [{"citedPaper": {"paperId": "s2-c", "externalIds": {"ArXiv": "2403.00003"}}}]},
    ]
    with patch.object(resolver, "_request_json", side_effect=responses) as request:
        result = resolver.fetch_batch(["2401.00001"])["2401.00001"]
    assert result.references == {"2402.00002", "2403.00003"}
    assert result.status == "ok"
    assert "offset=1" in request.call_args.args[0]


def test_retry_after_and_api_key_are_used() -> None:
    """HTTP 429 respects Retry-After and an injected key is sent only in the header."""
    resolver = SemanticScholarCitationResolver(api_key="test-key", min_interval=0)
    headers = Message()
    headers["Retry-After"] = "7"
    limited = urllib.error.HTTPError("https://example.org", 429, "rate limit", headers, None)
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(
        [{"paperId": "s2-a", "referenceCount": 0, "references": []}]
    ).encode()
    with (
        patch("urllib.request.urlopen", side_effect=[limited, response]) as opener,
        patch("autographrs.enrichment.citations.time.sleep") as sleep,
    ):
        result = resolver.fetch_batch(["2401.00001"])
    assert result["2401.00001"].status == "partial"
    sleep.assert_any_call(7.0)
    req = opener.call_args.args[0]
    assert req.get_header("X-api-key") == "test-key"
    assert req.get_method() == "POST"


@pytest.mark.parametrize("code, expected", [(429, "rate_limited"), (401, "error"), (404, "not_found"), (503, "error")])
def test_exhausted_http_errors_are_not_empty_success(code: int, expected: str) -> None:
    """Failed requests have an explicit outcome; permanent HTTP errors are not retried."""
    resolver = SemanticScholarCitationResolver(max_attempts=2, min_interval=0)
    error = urllib.error.HTTPError("https://example.org", code, "failure", Message(), None)
    with (
        patch("urllib.request.urlopen", side_effect=error) as opener,
        patch("autographrs.enrichment.citations.time.sleep"),
    ):
        result = resolver.fetch_batch(["2401.00001"])["2401.00001"]
    assert result.status == expected
    assert result.error == f"Semantic Scholar HTTP {code}"
    assert opener.call_count == (2 if code in (429, 503) else 1)


def test_missing_and_malformed_responses_are_distinct() -> None:
    """A null entry is not found; a malformed batch is an error."""
    resolver = SemanticScholarCitationResolver(min_interval=0)
    with patch.object(resolver, "_request_json", return_value=[None]):
        assert resolver.fetch_batch(["2401.00001"])["2401.00001"].status == "not_found"
    for payload in ({"error": "bad response"}, [], [{"paperId": "s2-a", "references": "invalid"}]):
        with patch.object(resolver, "_request_json", return_value=payload):
            assert resolver.fetch_batch(["2401.00001"])["2401.00001"].status == "error"


def test_long_retry_after_does_not_hammer_provider() -> None:
    """A cooldown beyond the retry budget is reported instead of retried prematurely."""
    resolver = SemanticScholarCitationResolver(max_retry_wait=10, min_interval=0)
    headers = Message()
    headers["Retry-After"] = "120"
    error = urllib.error.HTTPError("https://example.org", 429, "limited", headers, None)
    with patch("urllib.request.urlopen", side_effect=error) as opener:
        result = resolver.fetch_batch(["2401.00001"])["2401.00001"]
    assert result.status == "rate_limited"
    assert opener.call_count == 1


def test_partial_references_survive_pagination_failure() -> None:
    """Keep verified edges but record the incomplete lookup when a later page fails."""
    resolver = SemanticScholarCitationResolver(min_interval=0)
    response = [
        {
            "paperId": "s2-a",
            "referenceCount": 2,
            "references": [{"paperId": "s2-b", "externalIds": {"ArXiv": "2402.00002"}}],
        }
    ]
    error = urllib.error.HTTPError("https://example.org", 429, "limited", Message(), None)
    with patch.object(resolver, "_request_json", side_effect=[response, error]):
        result = resolver.fetch_batch(["2401.00001"])["2401.00001"]
    assert result.status == "partial"
    assert result.references == {"2402.00002"}
    assert result.error == "Semantic Scholar HTTP 429"


def test_disabled_and_empty_lookups_do_not_call_network() -> None:
    """Explicit opt-out is distinguishable from a successful empty bibliography."""
    resolver = SemanticScholarCitationResolver(enabled=False)
    with patch.object(resolver, "_request_json") as request:
        assert resolver.fetch_batch([]) == {}
        assert resolver.fetch_batch(["2401.00001"])["2401.00001"].status == "disabled"
        request.assert_not_called()


def test_large_batches_are_split_by_configured_size() -> None:
    """Batches default to 100 IDs to stay under the 10 MB response cap; 500 remains the hard limit."""
    resolver = SemanticScholarCitationResolver(min_interval=0)
    ids = [f"2401.{i:05d}" for i in range(201)]
    with patch.object(resolver, "_request_json", side_effect=[[None] * 100, [None] * 100, [None]]) as request:
        result = resolver.fetch_batch(ids)
    assert len(result) == 201
    assert [len(call.kwargs["body"]["ids"]) for call in request.call_args_list] == [100, 100, 1]

    resolver = SemanticScholarCitationResolver(min_interval=0, batch_size=500)
    ids = [f"2401.{i:05d}" for i in range(501)]
    with patch.object(resolver, "_request_json", side_effect=[[None] * 500, [None]]) as request:
        resolver.fetch_batch(ids)
    assert [len(call.kwargs["body"]["ids"]) for call in request.call_args_list] == [500, 1]

    with pytest.raises(ValueError):
        SemanticScholarCitationResolver(batch_size=501)
