"""Retrieve verified bibliographic references from Semantic Scholar."""

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Literal, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, Field, TypeAdapter

from autographrs.models import BibliographicReference, CitationProviderReport, Paper
from autographrs.sources.arxiv import canonical_arxiv_id

SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"


class CitationLookup(BaseModel):
    """Outcome of a provider lookup, distinct from an empty bibliography."""

    references: set[str] = Field(default_factory=set)
    bibliography: list[BibliographicReference] = Field(default_factory=list)
    provider_paper_id: str | None = None
    provider_identities: list[str] = Field(default_factory=list)
    attempts: list[CitationProviderReport] = Field(default_factory=list)
    citations: set[str] = Field(default_factory=set)
    status: Literal["ok", "partial", "not_found", "rate_limited", "error", "disabled"] = "ok"
    error: str | None = None
    source: str = "semantic_scholar"
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class _ExternalIds(BaseModel):
    arxiv: str | None = Field(default=None, alias="ArXiv")
    doi: str | None = Field(default=None, alias="DOI")


class _ScholarAuthor(BaseModel):
    name: str


class _ScholarPaper(BaseModel):
    paper_id: str | None = Field(default=None, alias="paperId")
    external_ids: _ExternalIds | None = Field(default=None, alias="externalIds")
    reference_count: int | None = Field(default=None, alias="referenceCount")
    title: str | None = None
    authors: list[_ScholarAuthor] | None = None
    year: int | None = None
    url: str | None = None
    references: list["_ScholarPaper | None"] | None = None


class _ReferenceEdge(BaseModel):
    cited_paper: _ScholarPaper | None = Field(default=None, alias="citedPaper")


class _ReferencePage(BaseModel):
    data: list[_ReferenceEdge]
    next: int | None = None


class CitationProviderProtocol(Protocol):
    """Provider returning both references and their retrieval status."""

    def fetch_references_and_citations(self, paper_id: str) -> CitationLookup:
        """Look up a single canonical arXiv paper."""
        ...


@runtime_checkable
class PaperCitationProviderProtocol(Protocol):
    """Source-independent lookup using the paper's identity and metadata."""

    def fetch_for_paper(self, paper: Paper) -> CitationLookup:
        """Resolve a paper using its source, identifier, DOI and landing URL."""
        ...


@runtime_checkable
class PaperBatchCitationProviderProtocol(Protocol):
    """Source-independent bulk lookup without turning a run into one request per paper."""

    def fetch_papers(self, papers: list[Paper]) -> dict[str, CitationLookup]:
        """Return outcomes keyed by the unique source-local identifiers in one run."""
        ...


@runtime_checkable
class BatchCitationProviderProtocol(CitationProviderProtocol, Protocol):
    """Optional batch capability for bibliographic providers."""

    def fetch_batch(self, paper_ids: list[str]) -> dict[str, CitationLookup]:
        """Return a lookup outcome for each requested identifier."""
        ...


class SemanticScholarCitationResolver:
    """Fetch references with pacing, bounded retries and optional API-key authentication."""

    def __init__(
        self,
        timeout: int = 20,
        enabled: bool = True,
        api_key: str | None = None,
        min_interval: float = 1.1,
        max_attempts: int = 4,
        max_retry_wait: float = 60.0,
        retry_wait_budget: float = 90.0,
        batch_size: int = 100,
    ) -> None:
        """Configure HTTP limits; read the API key from SEMANTIC_SCHOLAR_API_KEY if omitted.

        ``batch_size`` stays well below the API cap of 500 IDs because responses with inline
        references are limited to 10 MB.
        """
        if timeout <= 0 or min_interval < 0 or max_attempts < 1 or max_retry_wait < 0 or retry_wait_budget < 0:
            raise ValueError("Invalid Semantic Scholar timeout, pacing or retry limits")
        if not 1 <= batch_size <= 500:
            raise ValueError("Semantic Scholar batch_size must be between 1 and 500")
        self.timeout = timeout
        self.batch_size = batch_size
        self.enabled = enabled
        self._api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.min_interval = min_interval
        self.max_attempts = max_attempts
        self.max_retry_wait = max_retry_wait
        self.retry_wait_budget = retry_wait_budget
        self._retry_wait_remaining = retry_wait_budget
        self._last_request: float | None = None

    def _retry_delay(self, exc: urllib.error.URLError | TimeoutError, attempt: int) -> float:
        """Respect Retry-After, including HTTP-date values."""
        value = exc.headers.get("Retry-After") if isinstance(exc, urllib.error.HTTPError) and exc.headers else None
        if value:
            try:
                return max(0.0, float(value))
            except ValueError:
                try:
                    return max(0.0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
                except (ValueError, TypeError, OverflowError):
                    logger.warning("Invalid Semantic Scholar Retry-After header")
        base = 10.0 if isinstance(exc, urllib.error.HTTPError) and exc.code == 429 else 2.0
        return base * (2**attempt) + random.uniform(0, 1)

    def _request_json(self, url: str, body: dict[str, list[str]] | None = None) -> object:
        """Request JSON, retrying only transport failures, HTTP 429 and server errors."""
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"User-Agent": "autographrs/0.1", "Accept": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        if data is not None:
            headers["Content-Type"] = "application/json"
        for attempt in range(self.max_attempts):
            if self._last_request is not None:
                time.sleep(max(0.0, self.min_interval - (time.monotonic() - self._last_request)))
            self._last_request = time.monotonic()
            req = urllib.request.Request(url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError) as exc:
                retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code == 429 or 500 <= exc.code < 600
                delay = self._retry_delay(exc, attempt)
                if (
                    not retryable
                    or attempt + 1 == self.max_attempts
                    or delay > self.max_retry_wait
                    or delay > self._retry_wait_remaining
                ):
                    raise
                logger.warning(
                    "Semantic Scholar request failed; retry {}/{} in {}s", attempt + 1, self.max_attempts, delay
                )
                self._retry_wait_remaining -= delay
                time.sleep(delay)
        raise RuntimeError("Semantic Scholar retry loop exhausted unexpectedly")

    @staticmethod
    def _failure(exc: Exception) -> CitationLookup:
        """Represent retrieval failures explicitly without logging credentials or request headers."""
        status: Literal["not_found", "rate_limited", "error"] = "error"
        if isinstance(exc, urllib.error.HTTPError):
            status = "rate_limited" if exc.code == 429 else "not_found" if exc.code == 404 else "error"
            message = f"Semantic Scholar HTTP {exc.code}"
        else:
            message = f"Semantic Scholar {type(exc).__name__}"
        logger.warning("{}; citation data is unavailable, not an empty bibliography", message)
        return CitationLookup(status=status, error=message)

    def fetch_batch(self, paper_ids: list[str]) -> dict[str, CitationLookup]:
        """Keep the original bulk arXiv interface; collect all reference identities and metadata."""
        clean_ids = list(dict.fromkeys(canonical_arxiv_id(pid) for pid in paper_ids))
        return self._fetch_identifiers({pid: f"ARXIV:{pid}" for pid in clean_ids})

    def fetch_papers(self, papers: list[Paper]) -> dict[str, CitationLookup]:
        """Batch papers from arXiv or other repositories using DOI, PMID or paper URL."""
        from autographrs.enrichment.arxiv_html import normalize_doi

        if len({paper.paper_id for paper in papers}) != len(papers):
            raise ValueError("Scholar batch requires unique paper_id values")
        identifiers: dict[str, str] = {}
        for paper in papers:
            if paper.source == "arxiv":
                identifiers[paper.paper_id] = f"ARXIV:{canonical_arxiv_id(paper.paper_id)}"
            elif paper.doi:
                identifiers[paper.paper_id] = f"DOI:{normalize_doi(paper.doi)}"
            elif paper.source == "pubmed" and paper.paper_id.isdigit():
                identifiers[paper.paper_id] = f"PMID:{paper.paper_id}"
            else:
                identifiers[paper.paper_id] = f"URL:{paper.html_url}"
        return self._fetch_identifiers(identifiers)

    @staticmethod
    def _bibliography(
        references: list[_ScholarPaper | None], citing: _ScholarPaper, own_id: str
    ) -> list[BibliographicReference]:
        from autographrs.enrichment.arxiv_html import normalize_doi

        entries: dict[str, BibliographicReference] = {}
        for position, paper in enumerate(references, 1):
            external = paper.external_ids if paper else None
            arxiv_id = canonical_arxiv_id(external.arxiv) if external and external.arxiv else None
            doi = normalize_doi(external.doi) if external and external.doi else None
            provider_id = (
                f"https://www.semanticscholar.org/paper/{paper.paper_id}" if paper and paper.paper_id else None
            )
            method: Literal["scholar_id", "arxiv_id", "doi", "unresolved"]
            if provider_id:
                target, method = f"scholar:{paper.paper_id}", "scholar_id"
            elif doi:
                target, method = f"doi:{doi}", "doi"
            elif arxiv_id:
                target, method = f"arxiv:{arxiv_id}", "arxiv_id"
            else:
                text = paper.model_dump_json() if paper else "null"
                digest = hashlib.sha256(f"{own_id}\0{position}\0{text}".encode()).hexdigest()
                target, method = f"ref:{digest}", "unresolved"
            urls = [url for url in (provider_id, paper.url if paper else None) if url]
            if doi:
                urls.append(f"https://doi.org/{doi}")
            entry = BibliographicReference(
                reference_id=paper.paper_id if paper and paper.paper_id else f"reference-{position}",
                position=len(entries) + 1,
                text=(paper.title or target) if paper else target,
                urls=list(dict.fromkeys(urls)),
                arxiv_ids=[arxiv_id] if arxiv_id else [],
                dois=[doi] if doi else [],
                target_id=target,
                match_method=method,
                evidence_url=f"https://www.semanticscholar.org/paper/{citing.paper_id}",
                provider_id=provider_id,
                title=paper.title if paper else None,
                authors=[author.name for author in paper.authors] if paper and paper.authors is not None else None,
                publication_year=paper.year if paper else None,
            )
            entries.setdefault(target, entry)
        return list(entries.values())

    def _fetch_identifiers(self, identifiers: dict[str, str]) -> dict[str, CitationLookup]:
        clean_ids = list(identifiers)
        self._retry_wait_remaining = self.retry_wait_budget
        if not self.enabled:
            return {pid: CitationLookup(status="disabled") for pid in clean_ids}
        results: dict[str, CitationLookup] = {}
        entries: dict[str, _ScholarPaper] = {}
        blocked: CitationLookup | None = None
        fields = "paperId,externalIds,title,authors,year,url"
        for start in range(0, len(clean_ids), self.batch_size):
            chunk = clean_ids[start : start + self.batch_size]
            try:
                payload = self._request_json(
                    f"{SEMANTIC_SCHOLAR_BASE_URL}/batch?fields={fields},referenceCount,"
                    + ",".join(f"references.{field}" for field in fields.split(",")),
                    body={"ids": [identifiers[pid] for pid in chunk]},
                )
                items = TypeAdapter(list[_ScholarPaper | None]).validate_python(payload)
                if len(items) != len(chunk):
                    raise ValueError("Semantic Scholar batch response length does not match request")
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                failure = self._failure(exc)
                blocked = failure
                results.update({pid: failure.model_copy(deep=True) for pid in clean_ids[start:]})
                break
            for pid, item in zip(chunk, items, strict=True):
                if item is None:
                    logger.warning("Paper {} not found in Semantic Scholar", pid)
                    results[pid] = CitationLookup(status="not_found")
                elif item.paper_id is None:
                    results[pid] = self._failure(ValueError("Missing Scholar identity in batch result"))
                else:
                    entries[pid] = item

        for pid, item in entries.items():
            inline_references = list(item.references or [])
            references = list(inline_references)
            provider_id = f"https://www.semanticscholar.org/paper/{item.paper_id}"
            lookup = CitationLookup(provider_paper_id=provider_id, provider_identities=[provider_id])
            if item.reference_count is None or item.reference_count > len(references):
                if blocked is not None:
                    lookup.status = "partial"
                    lookup.error = blocked.error
                else:
                    offset = 0
                    try:
                        while True:
                            page = _ReferencePage.model_validate(
                                self._request_json(
                                    f"{SEMANTIC_SCHOLAR_BASE_URL}/{item.paper_id}/references"
                                    f"?fields={fields}&limit=1000&offset={offset}"
                                )
                            )
                            if offset == 0:
                                # The paginated list supersedes the inline prefix; keep it if this request fails.
                                references = [edge.cited_paper for edge in page.data]
                            else:
                                references.extend(edge.cited_paper for edge in page.data)
                            if page.next is None:
                                break
                            if page.next <= offset:
                                raise ValueError("Non-advancing Semantic Scholar pagination")
                            offset = page.next
                    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                        references.extend(reference for reference in inline_references if reference not in references)
                        blocked = self._failure(exc)
                        lookup.status = "partial"
                        lookup.error = blocked.error
            lookup.bibliography = self._bibliography(references, item, pid)
            lookup.references = {identifier for reference in lookup.bibliography for identifier in reference.arxiv_ids}
            if not references:
                lookup.status = "partial"
                lookup.error = (
                    lookup.error or "Scholar returned no indexed references; bibliography coverage is unknown"
                )
            elif any(
                paper is None
                or (
                    not paper.paper_id
                    and not (paper.external_ids and (paper.external_ids.arxiv or paper.external_ids.doi))
                )
                for paper in references
            ):
                lookup.status = "partial"
                lookup.error = (
                    lookup.error or "Scholar returned unresolved bibliography entries; identities are incomplete"
                )
            if item.reference_count is not None and len(references) < item.reference_count:
                lookup.status = "partial"
                lookup.error = lookup.error or "Scholar returned fewer references than referenceCount"
            if lookup.error:
                logger.warning("{}: {}", pid, lookup.error)
            results[pid] = lookup
        return results

    def fetch_references_and_citations(self, paper_id: str) -> CitationLookup:
        """Fetch the bibliography of a single arXiv paper."""
        return self.fetch_batch([paper_id])[canonical_arxiv_id(paper_id)]


def fetch_bibliographies(
    papers: list[Paper],
    citation_provider: CitationProviderProtocol | None = None,
) -> dict[str, CitationLookup]:
    """Fetch provider outcomes without deriving relationships between selected papers."""
    from autographrs.enrichment.citation_service import CitationService

    ids = {paper.paper_id for paper in papers}
    if len(ids) != len(papers):
        raise ValueError("Bibliography retrieval requires unique paper_id values within a run")
    provider = citation_provider or CitationService()
    if isinstance(provider, PaperBatchCitationProviderProtocol):
        lookups = provider.fetch_papers(papers)
    elif isinstance(provider, PaperCitationProviderProtocol):
        lookups = {paper.paper_id: provider.fetch_for_paper(paper) for paper in papers}
    elif any(paper.source != "arxiv" for paper in papers):
        raise ValueError("This citation provider only supports arXiv IDs; use a paper-aware provider for other sources")
    elif isinstance(provider, BatchCitationProviderProtocol):
        lookups = provider.fetch_batch([paper.paper_id for paper in papers])
    else:
        lookups = {paper.paper_id: provider.fetch_references_and_citations(paper.paper_id) for paper in papers}
    if set(lookups) != ids:
        raise ValueError("Bibliography provider must return exactly one outcome per paper")
    return lookups
