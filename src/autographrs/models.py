"""Domain models for AutoGraphRS."""

import re
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class RankingCriteria(StrEnum):
    """Supported search ranking criteria."""

    RELEVANCE = "relevance"
    SUBMITTED_DATE = "submittedDate"
    LAST_UPDATED_DATE = "lastUpdatedDate"


class SortOrder(StrEnum):
    """Sort direction for search queries."""

    DESCENDING = "descending"
    ASCENDING = "ascending"


class SearchRequest(BaseModel):
    """Parameters for searching and selecting papers."""

    topic: str = Field(..., min_length=1, description="Comma-separated concepts, or a legacy query without commas")
    num_papers: int = Field(default=10, ge=1, le=200, description="Number of top papers to retrieve")
    ranking_criteria: RankingCriteria = Field(
        default=RankingCriteria.RELEVANCE,
        description="Ranking criterion: relevance, submittedDate, or lastUpdatedDate",
    )
    criteria_date: str | date | None = Field(
        default=None,
        description=(
            "Optional date value associated with the criterion, e.g. a date ('YYYY-MM-DD') or "
            "date range ('YYYY-MM-DD..YYYY-MM-DD') when filtering by submittedDate or lastUpdatedDate"
        ),
    )
    sort_order: SortOrder = Field(
        default=SortOrder.DESCENDING,
        description="Sort direction (descending by default)",
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Optional list of category filters (e.g. ['cs.AI', 'cs.CL'])",
    )

    @field_validator("topic")
    @classmethod
    def strip_topic(cls, value: str) -> str:
        """Validate comma-separated literal concepts while preserving legacy queries."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("topic must not be empty")
        if "," in cleaned:
            terms = [term.strip() for term in cleaned.split(",")]
            if not all(terms):
                raise ValueError("Comma-separated topics must not contain empty concepts")
            if any('"' in term for term in terms):
                raise ValueError(
                    "Comma-separated topics must not contain double quotes; phrases are quoted automatically"
                )
        return cleaned


class Paper(BaseModel):
    """Normalized scientific paper metadata."""

    paper_id: str = Field(..., description="Canonical paper identifier (e.g. arXiv ID '2301.12345')")
    source: str = Field(default="arxiv", description="Source index name (e.g. 'arxiv', 'pubmed')")
    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(default_factory=list, description="Ordered list of author names")
    abstract: str = Field(..., description="Paper abstract/summary")
    published_at: datetime = Field(..., description="Initial publication/submission timestamp (UTC)")
    updated_at: datetime = Field(..., description="Last update timestamp (UTC)")
    categories: list[str] = Field(default_factory=list, description="Subject categories or index tags")
    primary_category: str | None = Field(default=None, description="Primary category if provided by source")
    doi: str | None = Field(default=None, description="Digital Object Identifier if available")
    journal_ref: str | None = Field(default=None, description="Journal reference if available")
    html_url: str = Field(..., description="URL to the paper landing page")
    pdf_url: str | None = Field(default=None, description="Direct URL to the PDF")
    pdf_volume_path: str | None = Field(
        default=None,
        description="Path where the PDF is stored in Unity Catalog Volume or local storage",
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the paper was ingested",
    )

    @property
    def safe_filename(self) -> str:
        """Return a filesystem-safe base name derived from paper_id."""
        return re.sub(r"[^A-Za-z0-9._-]", "_", self.paper_id)


class AuthorIdentifiers(BaseModel):
    """Identifiers reported by a provider, not inferred from an author's name."""

    semantic_scholar: str | None = None
    openalex: str | None = None
    orcid: str | None = None


class Authorship(BaseModel):
    """One provider's author attribution, with position in that provider's author list."""

    name: str | None = None
    position: int = Field(ge=1)
    identifiers: AuthorIdentifiers = Field(default_factory=AuthorIdentifiers)
    provider: Literal["semantic_scholar", "openalex"]
    evidence_url: str


class BibliographicReference(BaseModel):
    """One bibliography entry, retained even without a resolvable paper identifier."""

    reference_id: str
    position: int
    label: str | None = None
    text: str
    urls: list[str] = Field(default_factory=list)
    arxiv_ids: list[str] = Field(default_factory=list)
    dois: list[str] = Field(default_factory=list)
    target_id: str = Field(description="Namespaced arxiv:/doi:/openalex: identifier or source-local ref: placeholder")
    match_method: Literal["arxiv_id", "doi", "openalex_id", "scholar_id", "unresolved"]
    evidence_url: str
    provider_id: str | None = None
    title: str | None = None
    authors: list[str] | None = None
    authorships: list[Authorship] | None = None
    publication_year: int | None = None


class CitationProviderReport(BaseModel):
    """Persist the outcome of each provider, including a failed primary before fallback."""

    source: str
    status: Literal["ok", "partial", "not_found", "rate_limited", "error", "disabled"]
    error: str | None = None
    fetched_at: datetime


class PaperEnrichment(BaseModel):
    """LLM and citation enrichment data for a single paper."""

    paper_id: str = Field(..., description="Canonical paper identifier")
    source: str = "arxiv"
    run_id: str = Field(..., description="Identifier of the run that produced this enrichment")
    title: str = Field(..., description="Paper title (kept for direct Obsidian display)")
    authors: list[str] = Field(
        default_factory=list,
        description="Ordered list of author names (kept for direct Obsidian display)",
    )
    authorships: list[Authorship] | None = Field(
        default=None,
        description="Provider author attributions; separate evidence, not merged by name or position",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Normalized thematic tags extracted by the LLM",
    )
    arxiv_category: list[str] = Field(
        default_factory=list,
        description="Humanized official arXiv categories (e.g. ['machine-learning', 'computation-and-language'])",
    )
    summary: str = Field(
        default="",
        description="Concise 2-3 sentence summary generated by the LLM",
    )
    bibliography: list[BibliographicReference] | None = Field(
        default=None,
        description="All extracted entries, including external/unresolved references; None for legacy uncollected data",
    )
    citation_status: Literal["ok", "partial", "not_found", "rate_limited", "error", "disabled"] = "disabled"
    citation_error: str | None = None
    citation_source: str | None = None
    citation_fetched_at: datetime | None = None
    citation_lookups: list[CitationProviderReport] | None = None
    citation_paper_ids: list[str] | None = None
    enriched_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when enrichment was generated (UTC)",
    )


class IngestionResult(BaseModel):
    """Summary of an ingestion and enrichment run."""

    run_id: str = Field(..., description="Unique identifier for the ingestion run")
    request: SearchRequest = Field(..., description="Search parameters used for the run")
    papers: list[Paper] = Field(default_factory=list, description="Ingested papers")
    enrichments: list[PaperEnrichment] = Field(
        default_factory=list,
        description="Enrichment records (tags, summary, and bibliography) for the ingested papers",
    )
    delta_table: str = Field(..., description="Fully qualified Delta table name where papers were saved")
    enrichment_table: str | None = Field(
        default=None,
        description="Fully qualified Delta table name where paper enrichments were saved",
    )
    volume_dir: str = Field(..., description="Directory path where PDFs were stored")
    started_at: datetime = Field(..., description="Run start timestamp (UTC)")
    completed_at: datetime = Field(..., description="Run completion timestamp (UTC)")


class PipelineRun(BaseModel):
    """Publication record and retry checkpoint for a single search snapshot."""

    run_id: str
    attempt_id: str
    status: Literal["RUNNING", "COMPLETED", "FAILED"] = "RUNNING"
    request_json: str
    source: str
    download_pdfs: bool
    enrich: bool
    selection_complete: bool = False
    ingestion_status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"] = "PENDING"
    enrichment_status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED"] = "PENDING"
    export_status: Literal["NOT_IMPLEMENTED"] = "NOT_IMPLEMENTED"
    paper_count: int = 0
    pdf_count: int = 0
    enrichment_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    vault_path: str | None = None
    zip_path: str | None = None


def build_run_id(topic: str, now: datetime | None = None) -> str:
    """Create a readable run ID with a UUID to avoid same-second collisions.

    Args:
        topic: Search topic string
        now: Optional timestamp override for testing

    Returns:
        Run identifier containing timestamp, topic slug and UUID

    """
    ts = (now or datetime.now(UTC)).strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:32] or "run"
    return f"{ts}_{slug}_{uuid4().hex}"
