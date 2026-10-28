"""Unity Catalog Volume and Delta Table storage for ingested papers."""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from loguru import logger

from autographrs.config import ProjectConfig
from autographrs.enrichment.citations import CitationLookup, CitationProviderProtocol, resolve_mutual_citations
from autographrs.enrichment.tagger import PaperTagger
from autographrs.models import IngestionResult, Paper, PaperEnrichment, PipelineRun, SearchRequest, build_run_id
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

PAPER_ENRICHMENT_DDL_SCHEMA = (
    "run_id STRING, "
    "paper_id STRING, "
    "source STRING, "
    "title STRING, "
    "authors ARRAY<STRING>, "
    "tags ARRAY<STRING>, "
    "arxiv_category ARRAY<STRING>, "
    "summary STRING, "
    "referenced_arxiv_ids ARRAY<STRING>, "
    "bibliography ARRAY<STRUCT<reference_id: STRING, position: INT, label: STRING, text: STRING, "
    "urls: ARRAY<STRING>, arxiv_ids: ARRAY<STRING>, dois: ARRAY<STRING>, target_id: STRING, "
    "match_method: STRING, evidence_url: STRING, provider_id: STRING, title: STRING, "
    "authors: ARRAY<STRING>, publication_year: INT>>, "
    "cites_in_set ARRAY<STRING>, "
    "cited_by_in_set ARRAY<STRING>, "
    "citation_status STRING, "
    "citation_error STRING, "
    "citation_source STRING, "
    "citation_fetched_at TIMESTAMP, "
    "citation_lookups ARRAY<STRUCT<source: STRING, status: STRING, error: STRING, fetched_at: TIMESTAMP>>, "
    "citation_paper_ids ARRAY<STRING>, "
    "enriched_at TIMESTAMP"
)

PIPELINE_RUNS_DDL_SCHEMA = (
    "run_id STRING, attempt_id STRING, status STRING, request_json STRING, source STRING, "
    "download_pdfs BOOLEAN, enrich BOOLEAN, selection_complete BOOLEAN, ingestion_status STRING, "
    "enrichment_status STRING, export_status STRING, paper_count INT, pdf_count INT, enrichment_count INT, "
    "warnings ARRAY<STRING>, error STRING, started_at TIMESTAMP, completed_at TIMESTAMP, "
    "vault_path STRING, zip_path STRING"
)


def _validate_run_id(run_id: str) -> str:
    """Keep run identifiers safe for paths and explicit for every operation."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,159}", run_id):
        raise ValueError("run_id must contain 1-160 letters, digits, underscores or hyphens")
    return run_id


def _quoted_name(name: str) -> str:
    """Quote configured SQL identifiers, rejecting unexpected qualification."""
    parts = name.split(".")
    if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", part) for part in parts):
        raise ValueError(f"Invalid SQL identifier: {name}")
    return ".".join(f"`{part}`" for part in parts)


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


def enrichments_to_records(enrichments: list[PaperEnrichment]) -> list[dict[str, Any]]:
    """Serialize `PaperEnrichment` instances into flat dictionary records for Spark."""
    return [
        {
            "run_id": item.run_id,
            "paper_id": item.paper_id,
            "source": item.source,
            "title": item.title,
            "authors": list(item.authors),
            "tags": list(item.tags),
            "arxiv_category": list(item.arxiv_category),
            "summary": item.summary,
            "referenced_arxiv_ids": list(item.referenced_arxiv_ids),
            "bibliography": (
                [reference.model_dump() for reference in item.bibliography] if item.bibliography is not None else None
            ),
            "cites_in_set": list(item.cites_in_set),
            "cited_by_in_set": list(item.cited_by_in_set),
            "citation_status": item.citation_status,
            "citation_error": item.citation_error,
            "citation_source": item.citation_source,
            "citation_fetched_at": item.citation_fetched_at,
            "citation_lookups": (
                [report.model_dump() for report in item.citation_lookups] if item.citation_lookups is not None else None
            ),
            "citation_paper_ids": item.citation_paper_ids,
            "enriched_at": item.enriched_at,
        }
        for item in enrichments
    ]


class DeltaPaperStore:
    """Manage Unity Catalog Volume directories and Delta tables for paper ingestion and enrichment."""

    def __init__(
        self,
        config: ProjectConfig,
        spark: "SparkSession | None" = None,
        table_name: str = "papers",
        enrichment_table_name: str = "paper_enrichment",
        volume_root_override: str | Path | None = None,
    ) -> None:
        """Initialize the Delta and Volume paper store.

        Args:
            config: Loaded `ProjectConfig` for the target environment
            spark: Optional `SparkSession`; lazily acquired if omitted
            table_name: Unqualified Delta table name for papers inside `config.full_schema_name`
            enrichment_table_name: Unqualified Delta table name for enrichments
            volume_root_override: Optional local directory override for offline testing

        """
        self.config = config
        self._spark = spark
        self.table_name = table_name
        self.enrichment_table_name = enrichment_table_name
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
    def full_enrichment_table_name(self) -> str:
        """Return the fully qualified Unity Catalog enrichment table name (`catalog.schema.paper_enrichment`)."""
        return f"{self.config.full_schema_name}.{self.enrichment_table_name}"

    @property
    def full_runs_table_name(self) -> str:
        """Return the run registry in the same environment as the paper tables."""
        return f"{self.config.full_schema_name}.pipeline_runs"

    def _ensure_table(self, table: str, schema: str) -> None:
        """Create a Delta table using an explicit schema, including empty runs."""
        self.spark.sql(f"CREATE TABLE IF NOT EXISTS {_quoted_name(table)} ({schema}) USING DELTA")

    def _read_rows(self, table: str, run_id: str) -> list[dict[str, Any]]:
        """Read only one run, using bound SQL values."""
        _validate_run_id(run_id)
        return [
            row.asDict(recursive=True)
            for row in self.spark.sql(
                f"SELECT * FROM {_quoted_name(table)} WHERE run_id = :run_id", args={"run_id": run_id}
            ).collect()
        ]

    @staticmethod
    def _unique_records(records: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
        """Deduplicate identical source rows and reject conflicting source/target keys."""
        unique: dict[tuple[object, ...], dict[str, Any]] = {}
        for row in records:
            key = tuple(row.get(field) for field in keys)
            if any(value is None for value in key):
                raise ValueError(f"Missing logical key {keys}")
            if key in unique and unique[key] != row:
                raise ValueError(f"Conflicting duplicate logical key {key}")
            unique[key] = row
        return list(unique.values())

    def _merge_records(
        self,
        records: list[dict[str, Any]],
        table: str,
        schema: str,
        keys: tuple[str, ...],
        matched_condition: str = "true",
    ) -> int:
        """Upsert one run with schema evolution; never overwrite another run."""
        if not records:
            return 0
        run_ids = {row["run_id"] for row in records}
        if len(run_ids) != 1:
            raise ValueError("A write must contain exactly one run_id")
        run_id = _validate_run_id(records[0]["run_id"])
        records = self._unique_records(records, keys)
        self._ensure_table(table, schema)
        # Schema evolution cannot resolve a new column used in the MERGE's ON clause.
        if "source" in keys and "source" not in self.spark.table(table).columns:
            self.spark.sql(f"ALTER TABLE {_quoted_name(table)} ADD COLUMNS (source STRING)")
        existing = self._read_rows(table, run_id)
        if len(self._unique_records(existing, keys)) != len(existing):
            raise ValueError(f"Duplicate historical keys in {table} for run {run_id}; repair explicitly before retry")
        view = f"autographrs_merge_{uuid4().hex}"
        frame = self.spark.createDataFrame(records, schema=schema)
        frame.createOrReplaceTempView(view)
        condition = " AND ".join(f"target.`{key}` = incoming.`{key}`" for key in keys)
        try:
            self.spark.sql(
                f"MERGE WITH SCHEMA EVOLUTION INTO {_quoted_name(table)} AS target "
                f"USING `{view}` AS incoming ON {condition} "
                f"WHEN MATCHED AND ({matched_condition}) THEN UPDATE SET * "
                "WHEN NOT MATCHED THEN INSERT *"
            )
        finally:
            self.spark.catalog.dropTempView(view)
        return len(records)

    def load_run(self, run_id: str) -> PipelineRun:
        """Require one unambiguous registry entry; legacy unregistered runs are not published."""
        rows = self._read_rows(self.full_runs_table_name, run_id)
        if len(rows) != 1:
            raise ValueError(f"Expected one registered run for {run_id}, found {len(rows)}")
        return PipelineRun.model_validate(rows[0])

    def start_run(
        self, run_id: str, request: SearchRequest, source: str, download_pdfs: bool, enrich: bool, retry: bool
    ) -> PipelineRun:
        """Claim a new run or a failed attempt with unchanged parameters."""
        _validate_run_id(run_id)
        self._ensure_table(self.full_runs_table_name, PIPELINE_RUNS_DDL_SCHEMA)
        previous: PipelineRun | None = None
        if retry:
            previous = self.load_run(run_id)
            if previous.status != "FAILED":
                raise ValueError(f"Run {run_id} is {previous.status}; only FAILED runs may be retried")
            if (
                previous.request_json != request.model_dump_json()
                or previous.source != source
                or previous.download_pdfs != download_pdfs
                or previous.enrich != enrich
            ):
                raise ValueError("Retry parameters must match the original run; use an empty run_id for a new search")
        elif self._read_rows(self.full_runs_table_name, run_id):
            raise ValueError(f"Run {run_id} already exists")
        run = PipelineRun(
            run_id=run_id,
            attempt_id=uuid4().hex,
            request_json=request.model_dump_json(),
            source=source,
            download_pdfs=download_pdfs,
            enrich=enrich,
            selection_complete=previous.selection_complete if previous else False,
            paper_count=previous.paper_count if previous else 0,
            started_at=previous.started_at if previous else datetime.now(UTC),
        )
        self._merge_records(
            [run.model_dump()],
            self.full_runs_table_name,
            PIPELINE_RUNS_DDL_SCHEMA,
            ("run_id",),
            matched_condition="target.status = 'FAILED' AND target.request_json = incoming.request_json "
            "AND target.source = incoming.source AND target.download_pdfs = incoming.download_pdfs "
            "AND target.enrich = incoming.enrich",
        )
        self._assert_owner(run)
        logger.info("Starting run {} (attempt {})", run.run_id, run.attempt_id)
        return run

    def _assert_owner(self, run: PipelineRun) -> None:
        """Reject concurrent attempts and already-published runs."""
        stored = self.load_run(run.run_id)
        if stored.attempt_id != run.attempt_id or stored.status != "RUNNING":
            raise ValueError(f"Run {run.run_id} is not owned by this active attempt")

    def update_run(self, run: PipelineRun) -> None:
        """Checkpoint or publish only the current active attempt."""
        self._assert_owner(run)
        self._merge_records(
            [run.model_dump()],
            self.full_runs_table_name,
            PIPELINE_RUNS_DDL_SCHEMA,
            ("run_id",),
            matched_condition="target.attempt_id = incoming.attempt_id AND target.status = 'RUNNING'",
        )
        stored = self.load_run(run.run_id)
        if stored.attempt_id != run.attempt_id or stored.status != run.status:
            raise ValueError(f"Run checkpoint failed for {run.run_id}")

    def _paper_rows(self, run: PipelineRun) -> list[dict[str, Any]]:
        """Validate snapshot uniqueness, source and stable ranking before reconstructing models."""
        rows = self._read_rows(self.full_table_name, run.run_id)
        if len(self._unique_records(rows, ("run_id", "source", "paper_id"))) != len(rows):
            raise ValueError(f"Duplicate paper keys in run {run.run_id}")
        if len(rows) != run.paper_count or any(row["source"] != run.source for row in rows):
            raise ValueError(f"Paper snapshot is inconsistent for run {run.run_id}")
        rows.sort(key=lambda row: row["rank"])
        if [row["rank"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"Invalid paper ranks in run {run.run_id}")
        return rows

    def load_papers(self, run_id: str) -> list[Paper]:
        """Read ranked papers only from a successfully published run."""
        run = self.load_run(run_id)
        if run.status != "COMPLETED" or run.ingestion_status != "COMPLETED":
            raise ValueError(f"Run {run_id} is not ready for reading")
        return [Paper.model_validate(row) for row in self._paper_rows(run)]

    def load_enrichments(self, run_id: str) -> list[PaperEnrichment]:
        """Read a complete enrichment snapshot without confusing skipped or partial runs."""
        run = self.load_run(run_id)
        if run.status != "COMPLETED" or run.enrichment_status != "COMPLETED":
            raise ValueError(f"Enrichment for run {run_id} is not ready (or was skipped)")
        return self._enrichment_models(run)

    def _enrichment_models(self, run: PipelineRun) -> list[PaperEnrichment]:
        """Check exact paper/enrichment key coverage before publication or reading."""
        run_id = run.run_id
        rows = self._read_rows(self.full_enrichment_table_name, run_id)
        if len(self._unique_records(rows, ("run_id", "source", "paper_id"))) != len(rows):
            raise ValueError(f"Duplicate enrichment keys in run {run_id}")
        if len(rows) != run.enrichment_count:
            raise ValueError(f"Enrichment count is inconsistent for run {run_id}")
        papers = self._paper_rows(run)
        by_key = {(row["source"], row["paper_id"]): row for row in rows}
        if set(by_key) != {(row["source"], row["paper_id"]) for row in papers}:
            raise ValueError(f"Paper/enrichment keys do not match for run {run_id}")
        return [PaperEnrichment.model_validate(by_key[(row["source"], row["paper_id"])]) for row in papers]

    def load_run_dataset(self, run_id: str) -> list[tuple[Paper, PaperEnrichment]]:
        """Join the same published run by source and paper_id, preserving rank."""
        return list(zip(self.load_papers(run_id), self.load_enrichments(run_id), strict=True))

    def _validate_write_keys(self, run_id: str, keys: set[tuple[str, str]], enrichment: bool = False) -> None:
        """Reject writes to terminal runs and changes to a finalized selection."""
        run = self.load_run(run_id)
        if run.status != "RUNNING":
            raise ValueError(f"Run {run_id} is {run.status}; terminal snapshots cannot be modified")
        if any(source != run.source for source, _ in keys):
            raise ValueError(f"Unexpected source in run {run_id}")
        if enrichment and not run.selection_complete:
            raise ValueError("Enrichment requires a finalized paper selection")
        if run.selection_complete:
            expected = {(row["source"], row["paper_id"]) for row in self._paper_rows(run)}
            if keys != expected:
                raise ValueError(f"Write keys do not match the finalized snapshot for run {run_id}")

    @property
    def volume_mount_root(self) -> Path:
        """Return the POSIX `/Volumes/<catalog>/<schema>/<volume>` root path (or test override)."""
        if self._volume_root_override is not None:
            return self._volume_root_override
        return Path(f"/Volumes/{self.config.catalog}/{self.config.schema}/{self.config.volume}")

    def run_pdf_dir(self, run_id: str) -> Path:
        """Return the directory path where PDFs for `run_id` are stored."""
        return self.volume_mount_root / "runs" / _validate_run_id(run_id) / "pdfs"

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
    ) -> int:
        """Persist a batch of ingested papers into the Unity Catalog Delta table.

        Args:
            papers: Normalized papers to write
            run_id: Identifier of the ingestion run
            request: Search parameters used for the run

        Returns:
            Number of rows written

        """
        if not papers:
            logger.warning("No papers to save for run_id={}", run_id)
            return 0

        papers = [
            Paper.model_validate(row)
            for row in self._unique_records([paper.model_dump() for paper in papers], ("source", "paper_id"))
        ]
        records = papers_to_records(papers, run_id=run_id, request=request)
        self._validate_write_keys(run_id, {(paper.source, paper.paper_id) for paper in papers})
        self._merge_records(records, self.full_table_name, PAPERS_TABLE_DDL_SCHEMA, ("run_id", "source", "paper_id"))
        logger.info("Saved {} papers to Delta table {}", len(records), self.full_table_name)
        return len(records)

    def save_enrichments(
        self,
        enrichments: list[PaperEnrichment],
    ) -> int:
        """Persist paper enrichment records (tags, summary, mutual citations) into Delta.

        Args:
            enrichments: Enrichment records to write

        Returns:
            Number of rows written

        """
        if not enrichments:
            logger.warning("No paper enrichments to save")
            return 0

        records = enrichments_to_records(enrichments)
        if len({item.run_id for item in enrichments}) != 1:
            raise ValueError("An enrichment write must contain exactly one run_id")
        self._validate_write_keys(
            enrichments[0].run_id, {(item.source, item.paper_id) for item in enrichments}, enrichment=True
        )
        self._merge_records(
            records, self.full_enrichment_table_name, PAPER_ENRICHMENT_DDL_SCHEMA, ("run_id", "source", "paper_id")
        )
        logger.info("Saved {} paper enrichments to Delta table {}", len(records), self.full_enrichment_table_name)
        return len(records)


def enrich_papers(
    papers: list[Paper],
    run_id: str,
    config: ProjectConfig,
    tagger: PaperTagger | None = None,
    citation_provider: CitationProviderProtocol | None = None,
    known_paper_ids: set[str] | None = None,
) -> tuple[list[Paper], list[PaperEnrichment]]:
    """Run LLM tool-calling tag/summary extraction and mutual citation resolution for `papers`.

    Also updates each `Paper.cited_paper_ids` with the resolved `referenced_arxiv_ids`.

    Args:
        papers: Papers ingested in the current run
        run_id: Run identifier
        config: Target environment configuration
        tagger: Optional `PaperTagger` instance
        citation_provider: Optional `CitationProviderProtocol` instance
        known_paper_ids: Optional additional paper IDs known in the catalog

    Returns:
        Tuple `(updated_papers, enrichments)`

    """
    if not papers:
        return [], []

    active_tagger = tagger or PaperTagger(config=config)
    outcomes: dict[str, CitationLookup] = {}
    all_refs_map, cites_in_set_map, cited_by_in_set_map = resolve_mutual_citations(
        papers=papers,
        citation_provider=citation_provider,
        known_paper_ids=known_paper_ids,
        lookup_outcomes=outcomes,
    )

    updated_papers: list[Paper] = []
    enrichments: list[PaperEnrichment] = []
    for paper in papers:
        llm_tags, arxiv_cats, summary = active_tagger.extract_tags_and_summary(paper)
        all_refs = all_refs_map.get(paper.paper_id, [])
        cites_in_set = cites_in_set_map.get(paper.paper_id, [])
        cited_by_in_set = cited_by_in_set_map.get(paper.paper_id, [])

        updated_papers.append(paper.model_copy(update={"cited_paper_ids": all_refs}))
        enrichments.append(
            PaperEnrichment(
                paper_id=paper.paper_id,
                source=paper.source,
                run_id=run_id,
                title=paper.title,
                authors=list(paper.authors),
                tags=llm_tags,
                arxiv_category=arxiv_cats,
                summary=summary,
                referenced_arxiv_ids=all_refs,
                bibliography=outcomes[paper.paper_id].bibliography,
                cites_in_set=cites_in_set,
                cited_by_in_set=cited_by_in_set,
                citation_status=outcomes[paper.paper_id].status,
                citation_error=outcomes[paper.paper_id].error,
                citation_source=outcomes[paper.paper_id].source,
                citation_fetched_at=outcomes[paper.paper_id].fetched_at,
                citation_lookups=outcomes[paper.paper_id].attempts,
                citation_paper_ids=list(
                    dict.fromkeys(
                        outcomes[paper.paper_id].provider_identities
                        + (
                            [outcomes[paper.paper_id].provider_paper_id]
                            if outcomes[paper.paper_id].provider_paper_id
                            else []
                        )
                    )
                ),
            )
        )

    return updated_papers, enrichments


def ingest_and_store_papers(
    request: SearchRequest,
    config: ProjectConfig,
    source: PaperSource,
    store: DeltaPaperStore | None = None,
    download_pdfs: bool = True,
    enrich: bool = True,
    tagger: PaperTagger | None = None,
    citation_provider: CitationProviderProtocol | None = None,
    run_id: str | None = None,
) -> IngestionResult:
    """Search papers, download PDFs, enrich with LLM tags & mutual citations, and store in Delta.

    Args:
        request: Topic, ranking criterion, optional criterion date, and top-N limit
        config: Target environment configuration
        source: Bibliographic source provider (e.g. `ArxivSource`)
        store: Optional pre-configured `DeltaPaperStore`
        download_pdfs: Whether to download PDF files into the Unity Catalog Volume
        enrich: Whether to run LLM tool-calling enrichment and mutual citation resolution
        tagger: Optional pre-configured `PaperTagger`
        citation_provider: Optional pre-configured `CitationProviderProtocol`
        run_id: Optional explicit run identifier

    Returns:
        `IngestionResult` summarizing the stored papers, enrichments, table names, and volume path

    """
    started_at = datetime.now(UTC)
    effective_run_id = run_id or build_run_id(request.topic, now=started_at)
    paper_store = store or DeltaPaperStore(config=config)
    run = paper_store.start_run(
        effective_run_id, request, source.source_name, download_pdfs, enrich, retry=run_id is not None
    )
    papers: list[Paper] = []
    enrichments: list[PaperEnrichment] = []
    pdf_dir = paper_store.run_pdf_dir(effective_run_id)
    enrichment_table: str | None = None
    try:
        paper_store.ensure_volume_exists()
        paper_store._ensure_table(paper_store.full_table_name, PAPERS_TABLE_DDL_SCHEMA)
        paper_store._ensure_table(paper_store.full_enrichment_table_name, PAPER_ENRICHMENT_DDL_SCHEMA)
        run.ingestion_status = "RUNNING"
        paper_store.update_run(run)
        if run.selection_complete:
            papers = [Paper.model_validate(row) for row in paper_store._paper_rows(run)]
        else:
            # Only discard a failed, unpublished selection; never delete another run's rows.
            paper_store.spark.sql(
                f"DELETE FROM {_quoted_name(paper_store.full_table_name)} WHERE run_id = :run_id",
                args={"run_id": effective_run_id},
            )
            papers = [
                Paper.model_validate(row)
                for row in paper_store._unique_records(
                    [paper.model_dump() for paper in source.search(request)], ("source", "paper_id")
                )
            ]
            if len(papers) > request.num_papers or any(paper.source != run.source for paper in papers):
                raise ValueError("Source returned more papers than requested or an unexpected source")
            paper_store.save_papers(papers, effective_run_id, request)
            run.paper_count = len(papers)
            run.selection_complete = True
            paper_store.update_run(run)
            papers = [Paper.model_validate(row) for row in paper_store._paper_rows(run)]

        if download_pdfs and papers:
            pending = [
                paper for paper in papers if not paper.pdf_volume_path or not Path(paper.pdf_volume_path).is_file()
            ]
            downloaded = (
                {(paper.source, paper.paper_id): paper for paper in source.download_pdfs(pending, pdf_dir)}
                if pending
                else {}
            )
            papers = [downloaded.get((paper.source, paper.paper_id), paper) for paper in papers]
            paper_store.save_papers(papers, effective_run_id, request)
            run.pdf_count = sum(
                bool(paper.pdf_volume_path and Path(paper.pdf_volume_path).is_file()) for paper in papers
            )
            if run.pdf_count != run.paper_count:
                raise RuntimeError(
                    f"Only {run.pdf_count}/{run.paper_count} PDFs downloaded; retry run {effective_run_id}"
                )
        run.ingestion_status = "COMPLETED"
        paper_store.update_run(run)
        if enrich:
            run.enrichment_status = "RUNNING"
            paper_store.update_run(run)
            papers, enrichments = enrich_papers(
                papers=papers,
                run_id=effective_run_id,
                config=config,
                tagger=tagger,
                citation_provider=citation_provider,
            )
            paper_store.save_papers(papers, effective_run_id, request)
            paper_store.save_enrichments(enrichments)
            run.enrichment_count = len(enrichments)
            paper_store._enrichment_models(run)
            run.warnings = [
                f"{item.paper_id}: citation lookup {item.citation_status}"
                for item in enrichments
                if item.citation_status not in ("ok", "disabled")
            ]
            run.warnings.extend(
                f"{item.paper_id}: {report.source} {report.status}: {report.error or 'incomplete citation lookup'}"
                for item in enrichments
                for report in (item.citation_lookups or [])
                if report.status not in ("ok", "disabled")
            )
            run.enrichment_status = "COMPLETED"
            enrichment_table = paper_store.full_enrichment_table_name
        else:
            run.enrichment_status = "SKIPPED"
        paper_store._paper_rows(run)
        run.status = "COMPLETED"
        completed_at = datetime.now(UTC)
        run.completed_at = completed_at
        paper_store.update_run(run)
    except Exception as exc:
        logger.exception("Run {} failed; published readers will reject it", effective_run_id)
        run.status = "FAILED"
        run.completed_at = datetime.now(UTC)
        run.error = f"{type(exc).__name__}: {exc}"[:2000]
        if run.ingestion_status == "RUNNING":
            run.ingestion_status = "FAILED"
        if run.enrichment_status == "RUNNING":
            run.enrichment_status = "FAILED"
        try:
            paper_store.update_run(run)
        except Exception as checkpoint_error:
            raise ExceptionGroup("Pipeline and failure-checkpoint writes both failed", [exc, checkpoint_error]) from exc
        raise

    return IngestionResult(
        run_id=effective_run_id,
        request=request,
        papers=papers,
        enrichments=enrichments,
        delta_table=paper_store.full_table_name,
        enrichment_table=enrichment_table,
        volume_dir=pdf_dir.as_posix(),
        started_at=run.started_at,
        completed_at=completed_at,
    )
