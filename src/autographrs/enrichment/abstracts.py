"""Batch abstract retrieval for exactly identified works, without bibliography expansion."""

import urllib.error
from datetime import UTC, datetime
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field, TypeAdapter

from autographrs.enrichment.arxiv_html import normalize_doi
from autographrs.enrichment.citations import SemanticScholarCitationResolver
from autographrs.enrichment.openalex import OpenAlexCitationResolver
from autographrs.graph.models import GraphNode
from autographrs.models import CitationProviderReport


def text_variant(title: str | None, abstract: str | None) -> Literal["title_only", "title_abstract"] | None:
    """Classify usable embedding inputs, without treating identifiers as titles."""
    if not title or not title.strip():
        return None
    return "title_abstract" if abstract and abstract.strip() else "title_only"


class AbstractMetadata(BaseModel):
    """Acquired metadata for one graph identity, separate from the original bibliography."""

    node_id: str
    title: str | None = None
    abstract: str | None = None
    text_variant: Literal["title_only", "title_abstract"] | None = None
    status: Literal["ok", "missing", "unresolved", "error", "rate_limited"] = "missing"
    provider: str | None = None
    evidence_url: str | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    attempts: list[CitationProviderReport] = Field(default_factory=list)


class _ScholarMetadata(BaseModel):
    paper_id: str = Field(alias="paperId")
    title: str | None = None
    abstract: str | None = None


class AbstractResolver:
    """Use Scholar batches, then exact OpenAlex IDs/DOIs only for missing abstracts."""

    def __init__(
        self,
        scholar: SemanticScholarCitationResolver | None = None,
        openalex: OpenAlexCitationResolver | None = None,
    ) -> None:
        self.scholar = scholar or SemanticScholarCitationResolver()
        self.openalex = openalex or OpenAlexCitationResolver()

    def fetch(self, nodes: list[GraphNode]) -> list[AbstractMetadata]:
        """Return one outcome per external/unresolved node with a bounded shared Scholar retry budget."""
        results = {
            node.node_id: AbstractMetadata(
                node_id=node.node_id,
                title=node.title if node.text_variant else None,
                status="unresolved" if node.kind == "unresolved" else "missing",
            )
            for node in nodes
        }
        # A graph display title can be an identifier or raw citation: only API titles are usable here.
        queries: dict[str, list[str]] = {}
        for node in nodes:
            if node.kind == "unresolved":
                continue
            for namespace, prefix in (("scholar", ""), ("doi", "DOI:"), ("arxiv", "ARXIV:")):
                values = [value.split(":", 1)[1] for value in node.identifiers if value.startswith(namespace + ":")]
                if len(values) == 1:
                    queries.setdefault(prefix + values[0], []).append(node.node_id)
                    break
        identifiers = sorted(queries)
        self.scholar._retry_wait_remaining = self.scholar.retry_wait_budget
        blocked: tuple[str, str] | None = None
        for start in range(0, len(identifiers), self.scholar.batch_size):
            chunk = identifiers[start : start + self.scholar.batch_size]
            try:
                if blocked:
                    for query in chunk:
                        for key in queries[query]:
                            self._report(results[key], "semantic_scholar", *blocked)
                    continue
                payload = self.scholar._request_json(
                    "https://api.semanticscholar.org/graph/v1/paper/batch?fields=paperId,title,abstract",
                    body={"ids": chunk},
                )
                items = TypeAdapter(list[_ScholarMetadata | None]).validate_python(payload)
                if len(items) != len(chunk):
                    raise ValueError("Scholar abstract batch response length does not match request")
                for query, item in zip(chunk, items, strict=True):
                    for key in queries[query]:
                        result = results[key]
                        if item:
                            self._apply(
                                result,
                                item.title,
                                item.abstract,
                                "semantic_scholar",
                                f"https://www.semanticscholar.org/paper/{item.paper_id}",
                            )
                        self._report(
                            result,
                            "semantic_scholar",
                            "ok" if result.abstract else "partial",
                            None if result.abstract else "No abstract returned",
                        )
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                failure = self.scholar._failure(exc)
                blocked = (failure.status, failure.error or "Abstract lookup failed")
                for query in chunk:
                    for key in queries[query]:
                        self._report(results[key], "semantic_scholar", *blocked)

        pending: dict[tuple[str, str], list[str]] = {}
        for node in nodes:
            if node.kind == "unresolved" or results[node.node_id].abstract:
                continue
            for namespace, field in (("openalex", "openalex_id"), ("doi", "doi")):
                values = [value.split(":", 1)[1] for value in node.identifiers if value.startswith(namespace + ":")]
                if len(values) == 1:
                    pending.setdefault((field, values[0]), []).append(node.node_id)
                    break
        for field in ("openalex_id", "doi"):
            values = sorted(value for kind, value in pending if kind == field)
            for start in range(0, len(values), 100):
                chunk = values[start : start + 100]
                try:
                    page = self.openalex._request(
                        {
                            "filter": field + ":" + "|".join(chunk),
                            "per-page": 100,
                            "select": "id,doi,title,abstract_inverted_index",
                        }
                    )
                    for value in chunk:
                        matches = [
                            work
                            for work in page.results
                            if (
                                work.id.rsplit("/", 1)[-1] == value
                                if field == "openalex_id"
                                else work.doi and normalize_doi(work.doi) == value
                            )
                        ]
                        if len(matches) > 1:
                            raise ValueError("Ambiguous OpenAlex abstract identity")
                        for key in pending[(field, value)]:
                            result = results[key]
                            if matches:
                                work = matches[0]
                                self._apply(
                                    result,
                                    work.title,
                                    reconstruct_abstract(work.abstract_inverted_index),
                                    "openalex",
                                    work.id,
                                )
                            self._report(
                                result,
                                "openalex",
                                "ok" if result.abstract else "partial",
                                None if result.abstract else "No abstract returned",
                            )
                except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                    status, error = self.openalex._failure(exc)
                    for value in chunk:
                        for key in pending[(field, value)]:
                            self._report(results[key], "openalex", status, error)
                    # Exhausted retries stop this provider rather than repeating for every chunk.
                    for value in values[start + 100 :]:
                        for key in pending[(field, value)]:
                            self._report(results[key], "openalex", status, error)
                    break
        for result in results.values():
            result.text_variant = text_variant(result.title, result.abstract)
            if result.abstract:
                result.status = "ok"
            elif any(report.status == "rate_limited" for report in result.attempts):
                result.status = "rate_limited"
            elif any(report.status == "error" for report in result.attempts):
                result.status = "error"
            if result.status != "ok":
                logger.warning("Bibliography metadata {}: {}", result.node_id, result.status)
        return [results[key] for key in sorted(results)]

    @staticmethod
    def _apply(result: AbstractMetadata, title: str | None, abstract: str | None, provider: str, url: str) -> None:
        result.title = title.strip() if title and title.strip() else result.title
        result.abstract = abstract.strip() if abstract and abstract.strip() else result.abstract
        result.provider = provider
        result.evidence_url = url

    @staticmethod
    def _report(result: AbstractMetadata, provider: str, status: str, error: str | None) -> None:
        result.attempts.append(
            CitationProviderReport.model_validate(
                {"source": provider, "status": status, "error": error, "fetched_at": datetime.now(UTC)}
            )
        )


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str | None:
    """Reconstruct OpenAlex text by positions; reject malformed indexes explicitly."""
    if not index:
        return None
    words: dict[int, str] = {}
    for word, positions in index.items():
        for position in positions:
            if position < 0 or position in words:
                raise ValueError("Invalid OpenAlex abstract word positions")
            words[position] = word
    if not words:
        return None
    if sorted(words) != list(range(len(words))):
        raise ValueError("Non-contiguous OpenAlex abstract word positions")
    return " ".join(words[position] for position in range(len(words)))
