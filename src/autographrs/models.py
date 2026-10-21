"""Domain models for AutoGraphRS."""

import re
from datetime import UTC, date, datetime
from enum import StrEnum

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

    topic: str = Field(..., min_length=1, description="Search topic or query string")
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
        """Ensure topic is not blank after stripping whitespace."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("topic must not be empty")
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
    cited_paper_ids: list[str] = Field(
        default_factory=list,
        description="Explicitly referenced paper IDs when available",
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the paper was ingested",
    )

    @property
    def safe_filename(self) -> str:
        """Return a filesystem-safe base name derived from paper_id."""
        return re.sub(r"[^A-Za-z0-9._-]", "_", self.paper_id)


class IngestionResult(BaseModel):
    """Summary of an ingestion run."""

    run_id: str = Field(..., description="Unique identifier for the ingestion run")
    request: SearchRequest = Field(..., description="Search parameters used for the run")
    papers: list[Paper] = Field(default_factory=list, description="Ingested papers")
    delta_table: str = Field(..., description="Fully qualified Delta table name where papers were saved")
    volume_dir: str = Field(..., description="Directory path where PDFs were stored")
    started_at: datetime = Field(..., description="Run start timestamp (UTC)")
    completed_at: datetime = Field(..., description="Run completion timestamp (UTC)")


def build_run_id(topic: str, now: datetime | None = None) -> str:
    """Create a deterministic-format run ID from a timestamp and topic slug.

    Args:
        topic: Search topic string
        now: Optional timestamp override for testing

    Returns:
        Run identifier string (e.g. '20261021_144500_graph_rag')

    """
    ts = (now or datetime.now(UTC)).strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:32] or "run"
    return f"{ts}_{slug}"
