"""OpenAlex reference edges and cited-work metadata, independent of arXiv HTML availability."""

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Literal
from urllib.parse import urlencode, urlsplit

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from autographrs.enrichment.arxiv_html import normalize_doi
from autographrs.enrichment.citations import CitationLookup
from autographrs.models import AuthorIdentifiers, Authorship, BibliographicReference, Paper
from autographrs.sources.arxiv import canonical_arxiv_id

_BASE_URL = "https://api.openalex.org/works"
_ARXIV_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[a-zA-Z][a-zA-Z.-]*/\d{7})(?:v\d+)?")


class _Author(BaseModel):
    id: str | None = None
    display_name: str | None = None
    orcid: str | None = None


class _Authorship(BaseModel):
    author: _Author


class _Location(BaseModel):
    landing_page_url: str | None = None
    pdf_url: str | None = None


class _Work(BaseModel):
    id: str
    doi: str | None = None
    title: str | None = None
    abstract_inverted_index: dict[str, list[int]] | None = None
    publication_year: int | None = None
    authorships: list[_Authorship] | None = None
    locations: list[_Location] = Field(default_factory=list)
    referenced_works: list[str] | None = None


class _Page(BaseModel):
    results: list[_Work]


def _short_id(identifier: str) -> str:
    short = identifier.rsplit("/", 1)[-1]
    if not re.fullmatch(r"W\d+", short):
        raise ValueError("Invalid OpenAlex work identifier")
    return short


def _arxiv_ids(work: _Work) -> list[str]:
    identifiers: set[str] = set()
    urls = [work.doi] if work.doi else []
    for location in work.locations:
        urls.extend(url for url in (location.landing_page_url, location.pdf_url) if url)
    for url in urls:
        parsed = urlsplit(url)
        if parsed.hostname in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
            identifier = re.sub(r"^/(?:abs|pdf|html)/", "", parsed.path).removesuffix(".pdf")
        elif parsed.hostname in {"doi.org", "dx.doi.org"}:
            doi = normalize_doi(url)
            if not doi.startswith("10.48550/arxiv."):
                continue
            identifier = doi.removeprefix("10.48550/arxiv.")
        else:
            continue
        if _ARXIV_ID.fullmatch(identifier):
            identifiers.add(canonical_arxiv_id(identifier))
    return sorted(identifiers)


class OpenAlexCitationResolver:
    """Resolve sources by exact identifiers and hydrate every indexed outgoing reference.

    No title-only matches, recursive expansion or HTML/Scholar fallback are performed.
    API credentials are never included in logged errors.
    """

    def __init__(
        self,
        papers: list[Paper] | None = None,
        api_key: str | None = None,
        timeout: float = 30,
        min_interval: float = 1.0,
        max_attempts: int = 3,
        max_retry_wait: float = 30,
    ) -> None:
        if timeout <= 0 or min_interval < 0 or max_attempts < 1 or max_retry_wait < 0:
            raise ValueError("Invalid OpenAlex timeout, pacing or retry limits")
        self._papers = {paper.paper_id: paper for paper in papers or []}
        self._api_key = api_key or os.getenv("OPENALEX_API_KEY")
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_attempts = max_attempts
        self.max_retry_wait = max_retry_wait
        self._last_request: float | None = None
        self._cache: dict[str, _Work] = {}

    def _request(self, parameters: dict[str, str | int]) -> _Page:
        parameters = dict(parameters)
        if self._api_key:
            parameters["api_key"] = self._api_key
        url = f"{_BASE_URL}?{urlencode(parameters)}"
        for attempt in range(self.max_attempts):
            if self._last_request is not None:
                time.sleep(max(0.0, self.min_interval - (time.monotonic() - self._last_request)))
            self._last_request = time.monotonic()
            request = urllib.request.Request(
                url, headers={"User-Agent": "autographrs/0.1", "Accept": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return _Page.model_validate(json.load(response))
            except (urllib.error.URLError, TimeoutError) as exc:
                retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code == 429 or 500 <= exc.code < 600
                delay = float(2 ** (attempt + 1))
                retry_after = (
                    exc.headers.get("Retry-After") if isinstance(exc, urllib.error.HTTPError) and exc.headers else None
                )
                if retry_after:
                    try:
                        delay = max(0.0, float(retry_after))
                    except ValueError:
                        try:
                            delay = max(0.0, (parsedate_to_datetime(retry_after) - datetime.now(UTC)).total_seconds())
                        except (ValueError, TypeError, OverflowError):
                            logger.warning("Invalid OpenAlex Retry-After header")
                if not retryable or attempt + 1 == self.max_attempts or delay > self.max_retry_wait:
                    raise
                logger.warning("OpenAlex request failed; retry {}/{} in {}s", attempt + 1, self.max_attempts, delay)
                time.sleep(delay)
        raise RuntimeError("OpenAlex retry loop exhausted unexpectedly")

    @staticmethod
    def _failure(exc: Exception) -> tuple[Literal["rate_limited", "error", "not_found"], str]:
        if isinstance(exc, urllib.error.HTTPError):
            status = "rate_limited" if exc.code == 429 else "not_found" if exc.code == 404 else "error"
            return status, f"OpenAlex HTTP {exc.code}"
        return "error", f"OpenAlex {type(exc).__name__}"

    def _find_work(self, paper: Paper) -> _Work | None:
        if paper.source == "arxiv":
            paper_id = canonical_arxiv_id(paper.paper_id)
            if not _ARXIV_ID.fullmatch(paper_id):
                raise ValueError("Invalid arXiv paper identifier")
            locations = "|".join(
                (
                    f"https://arxiv.org/abs/{paper_id}",
                    f"http://arxiv.org/abs/{paper_id}",
                    f"https://doi.org/10.48550/arxiv.{paper_id}",
                )
            )
            page = self._request({"filter": f"locations.landing_page_url:{locations}", "per-page": 100})
            matches = [work for work in page.results if paper_id in _arxiv_ids(work)]
            if len(matches) > 1:
                raise ValueError("Ambiguous OpenAlex arXiv identity")
            if matches:
                return matches[0]
        if paper.doi:
            doi = normalize_doi(paper.doi)
            page = self._request({"filter": f"doi:{doi}", "per-page": 100})
            matches = [
                work
                for work in page.results
                if (work.doi and normalize_doi(work.doi) == doi)
                or any(
                    location.landing_page_url
                    and urlsplit(location.landing_page_url).hostname in {"doi.org", "dx.doi.org"}
                    and normalize_doi(location.landing_page_url) == doi
                    for location in work.locations
                )
            ]
            if len(matches) > 1:
                raise ValueError("Ambiguous OpenAlex DOI identity")
            if matches:
                return matches[0]
        if paper.source != "arxiv":
            parsed = urlsplit(paper.html_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or "|" in paper.html_url:
                raise ValueError("Paper landing URL must be an unambiguous HTTP(S) URL")
            page = self._request({"filter": f"locations.landing_page_url:{paper.html_url}", "per-page": 100})
            matches = [
                work
                for work in page.results
                if any(location.landing_page_url == paper.html_url for location in work.locations)
            ]
            if len(matches) > 1:
                raise ValueError("Ambiguous OpenAlex landing URL identity")
            if matches:
                return matches[0]
        return None

    @staticmethod
    def _authorships(work: _Work) -> list[Authorship] | None:
        if work.authorships is None:
            return None
        return [
            Authorship(
                name=item.author.display_name,
                position=position,
                identifiers=AuthorIdentifiers(openalex=item.author.id, orcid=item.author.orcid),
                provider="openalex",
                evidence_url=work.id,
            )
            for position, item in enumerate(work.authorships, 1)
        ]

    @staticmethod
    def _reference(identifier: str, position: int, citing: _Work, work: _Work | None) -> BibliographicReference:
        short = _short_id(identifier)
        authors = (
            [item.author.display_name for item in work.authorships if item.author.display_name]
            if work and work.authorships is not None
            else None
        )
        urls = [identifier]
        if work:
            urls.extend(location.landing_page_url for location in work.locations if location.landing_page_url)
            if work.doi:
                urls.append(work.doi)
        return BibliographicReference(
            reference_id=short,
            position=position,
            text=(work.title or identifier) if work else identifier,
            urls=list(dict.fromkeys(urls)),
            arxiv_ids=_arxiv_ids(work) if work else [],
            dois=[normalize_doi(work.doi)] if work and work.doi else [],
            target_id=f"openalex:{short}",
            match_method="openalex_id",
            evidence_url=citing.id,
            provider_id=identifier,
            title=work.title if work else None,
            authors=authors,
            authorships=OpenAlexCitationResolver._authorships(work) if work else None,
            publication_year=work.publication_year if work else None,
        )

    def fetch_references_and_citations(self, paper_id: str) -> CitationLookup:
        """Keep the original arXiv-ID interface for callers without a Paper model."""
        if paper := self._papers.get(paper_id):
            return self.fetch_for_paper(paper)
        if not _ARXIV_ID.fullmatch(paper_id):
            raise ValueError(f"Invalid arXiv identifier: {paper_id}")
        paper_id = canonical_arxiv_id(paper_id)
        now = datetime.now(UTC)
        return self.fetch_for_paper(
            Paper(
                paper_id=paper_id,
                source="arxiv",
                title="",
                abstract="",
                published_at=now,
                updated_at=now,
                html_url=f"https://arxiv.org/abs/{paper_id}",
            )
        )

    def fetch_for_paper(self, paper: Paper) -> CitationLookup:
        """Resolve a source-independent Paper, preserving all indexed outgoing references."""
        paper_id = paper.paper_id
        errors: list[str] = []
        try:
            work = self._find_work(paper)
            if work is None:
                error = (
                    "No unambiguous OpenAlex match by source identifiers, supplied DOI or landing URL; "
                    "no title match was assumed"
                )
                logger.warning("{}: {}", paper_id, error)
                return CitationLookup(source="openalex", status="not_found", error=error)
            if work.referenced_works is None:
                raise ValueError("OpenAlex response omitted referenced_works")
            identifiers = list(dict.fromkeys(work.referenced_works))
            if not identifiers:
                errors.append(
                    "OpenAlex returned no indexed references; bibliography coverage is unknown, "
                    "not evidence that the paper has no references"
                )
            for identifier in identifiers:
                _short_id(identifier)
            self._cache[work.id] = work
            missing = [identifier for identifier in identifiers if identifier not in self._cache]
            for offset in range(0, len(missing), 100):
                chunk = missing[offset : offset + 100]
                try:
                    page = self._request(
                        {
                            "filter": "openalex_id:" + "|".join(_short_id(identifier) for identifier in chunk),
                            "per-page": 100,
                        }
                    )
                    self._cache.update({item.id: item for item in page.results if item.id in chunk})
                    if any(identifier not in self._cache for identifier in chunk):
                        errors.append("OpenAlex omitted metadata for some referenced works; IDs retained")
                except (urllib.error.URLError, TimeoutError, ValueError, ValidationError) as exc:
                    _, error = self._failure(exc)
                    errors.append(f"{error} while retrieving cited metadata; IDs retained")
                    break
            bibliography = [
                self._reference(identifier, position, work, self._cache.get(identifier))
                for position, identifier in enumerate(identifiers, 1)
            ]
            error = "; ".join(dict.fromkeys(errors)) or None
            if error:
                logger.warning("{}: {}", paper_id, error)
            return CitationLookup(
                source="openalex",
                status="partial" if errors else "ok",
                error=error,
                provider_paper_id=work.id,
                authorships=self._authorships(work),
                bibliography=bibliography,
                references={identifier for entry in bibliography for identifier in entry.arxiv_ids},
            )
        except (urllib.error.URLError, TimeoutError, ValueError, ValidationError) as exc:
            status, error = self._failure(exc)
            logger.warning("{}: {}; citation coverage is unknown", paper_id, error)
            return CitationLookup(source="openalex", status=status, error=error)
