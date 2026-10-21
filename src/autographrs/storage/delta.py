"""Unity Catalog Volume and Delta Table storage for ingested papers."""

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from autographrs.config import ProjectConfig
from autographrs.models import IngestionResult, Paper, SearchRequest, build_run_id
from autographrs.sources.base import PaperSource

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

PAPERS_TABLE_DDL_SCHEMA = (
    "run_id STRING, "
    "rank INT, "
    "paper_id STRING, "
    "source STRING, "
    "title STRING, "
    "authors ARRAY<STRING>, "
    "abstract STRING, "
    "published_at TIMESTAMP, "
    "updated_at TIMESTAMP, "
    "categories ARRAY<STRING>, "
    "primary_category STRING, "
    "doi STRING, "
    "journal_ref STRING, "
    "html_url STRING, "
    "pdf_url STRING, "
    "pdf_volume_path STRING, "
    "cited_paper_ids ARRAY<STRING>, "
    "search_topic STRING, "
    "ranking_criteria STRING, "
    "criteria_date STRING, "
    "ingested_at TIMESTAMP"
)


def papers_to_records(
    papers: list[Paper],
    run_id: str,
    request: SearchRequest,
) -> list[dict[str, Any]]:
    """Serialize `Paper` instances into flat dictionary records ready for a Spark DataFrame.

    Args:
        papers: Normalized paper objects
        run_id: Unique identifier of the ingestion run
        request: Search request that produced these papers

    Returns:
        List of row dictionaries matching the `papers` Delta table schema

    """
    criteria_date = str(request.criteria_date) if request.criteria_date is not None else None
    records: list[dict[str, Any]] = []
    for rank, paper in enumerate(papers, start=1):
        records.append(
            {
                "run_id": run_id,
                "rank": rank,
                "paper_id": paper.paper_id,
                "source": paper.source,
                "title": paper.title,
                "authors": list(paper.authors),
                "abstract": paper.abstract,
                "published_at": paper.published_at,
                "updated_at": paper.updated_at,
                "categories": list(paper.categories),
                "primary_category": paper.primary_category,
                "doi": paper.doi,
                "journal_ref": paper.journal_ref,
                "html_url": paper.html_url,
                "pdf_url": paper.pdf_url,
                "pdf_volume_path": paper.pdf_volume_path,
                "cited_paper_ids": list(paper.cited_paper_ids),
                "search_topic": request.topic,
                "ranking_criteria": str(request.ranking_criteria),
                "criteria_date": criteria_date,
                "ingested_at": paper.ingested_at,
            }
        )
    return records


class DeltaPaperStore:
    """Manage Unity Catalog Volume directories and Delta tables for paper ingestion."""

    def __init__(
        self,
        config: ProjectConfig,
        spark: "SparkSession | None" = None,
        table_name: str = "papers",
        volume_root_override: str | Path | None = None,
    ) -> None:
        """Initialize the Delta and Volume paper store.

        Args:
            config: Loaded `ProjectConfig` for the target environment
            spark: Optional `SparkSession`; lazily acquired if omitted
            table_name: Unqualified Delta table name inside `config.full_schema_name`
            volume_root_override: Optional local directory override for offline testing

        """
        self.config = config
        self._spark = spark
        self.table_name = table_name
        self._volume_root_override = Path(volume_root_override) if volume_root_override else None

    @property
    def spark(self) -> "SparkSession":
        """Lazily obtain or create the active SparkSession."""
        if self._spark is None:
            from pyspark.sql import SparkSession

            self._spark = SparkSession.builder.getOrCreate()
        return self._spark

    @property
    def full_table_name(self) -> str:
        """Return the fully qualified Unity Catalog table name (`catalog.schema.papers`)."""
        return f"{self.config.full_schema_name}.{self.table_name}"

    @property
    def volume_mount_root(self) -> Path:
        """Return the POSIX `/Volumes/<catalog>/<schema>/<volume>` root path (or test override)."""
        if self._volume_root_override is not None:
            return self._volume_root_override
        return Path(f"/Volumes/{self.config.catalog}/{self.config.schema}/{self.config.volume}")

    def run_pdf_dir(self, run_id: str) -> Path:
        """Return the directory path where PDFs for `run_id` are stored."""
        return self.volume_mount_root / "runs" / run_id / "pdfs"

    def ensure_volume_exists(self) -> None:
        """Create the configured Unity Catalog Volume if it does not already exist."""
        if self._volume_root_override is not None:
            self._volume_root_override.mkdir(parents=True, exist_ok=True)
            return

        sql_stmt = f"CREATE VOLUME IF NOT EXISTS {self.config.full_volume_path}"
        logger.info("Ensuring Unity Catalog Volume exists: {}", self.config.full_volume_path)
        self.spark.sql(sql_stmt)

    def save_papers(
        self,
        papers: list[Paper],
        run_id: str,
        request: SearchRequest,
        mode: str = "append",
    ) -> int:
        """Persist a batch of ingested papers into the Unity Catalog Delta table.

        Args:
            papers: Normalized papers to write
            run_id: Identifier of the ingestion run
            request: Search parameters used for the run
            mode: Spark write mode ('append' by default)

        Returns:
            Number of rows written

        """
        if not papers:
            logger.warning("No papers to save for run_id={}", run_id)
            return 0

        records = papers_to_records(papers, run_id=run_id, request=request)
        df = self.spark.createDataFrame(records, schema=PAPERS_TABLE_DDL_SCHEMA)
        df.write.format("delta").mode(mode).option("mergeSchema", "true").saveAsTable(self.full_table_name)
        logger.info("Saved {} papers to Delta table {}", len(records), self.full_table_name)
        return len(records)


def ingest_and_store_papers(
    request: SearchRequest,
    config: ProjectConfig,
    source: PaperSource,
    store: DeltaPaperStore | None = None,
    download_pdfs: bool = True,
    run_id: str | None = None,
) -> IngestionResult:
    """Search papers from `source`, download their PDFs to the Volume, and store metadata in Delta.

    Args:
        request: Topic, ranking criterion, optional criterion value, and top-N limit
        config: Target environment configuration
        source: Bibliographic source provider (e.g. `ArxivSource`)
        store: Optional pre-configured `DeltaPaperStore`
        download_pdfs: Whether to download PDF files into the Unity Catalog Volume
        run_id: Optional explicit run identifier

    Returns:
        `IngestionResult` summarizing the stored papers, table name, and volume path

    """
    started_at = datetime.now(UTC)
    effective_run_id = run_id or build_run_id(request.topic, now=started_at)
    paper_store = store or DeltaPaperStore(config=config)

    paper_store.ensure_volume_exists()
    papers = source.search(request)

    pdf_dir = paper_store.run_pdf_dir(effective_run_id)
    if download_pdfs and papers:
        papers = source.download_pdfs(papers, pdf_dir)

    paper_store.save_papers(papers, run_id=effective_run_id, request=request)
    completed_at = datetime.now(UTC)

    return IngestionResult(
        run_id=effective_run_id,
        request=request,
        papers=papers,
        delta_table=paper_store.full_table_name,
        volume_dir=pdf_dir.as_posix(),
        started_at=started_at,
        completed_at=completed_at,
    )
