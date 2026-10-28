"""Unit tests for paper ingestion, ArxivSource, and Delta/Volume storage."""

import re
from copy import deepcopy
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import arxiv
import pytest

from autographrs.config import ProjectConfig
from autographrs.models import Paper, RankingCriteria, SearchRequest, build_run_id
from autographrs.pipeline import run_ingestion
from autographrs.sources.arxiv import (
    ArxivSource,
    canonical_arxiv_id,
    format_arxiv_query,
    parse_date_bounds,
)
from autographrs.sources.base import PaperSource
from autographrs.storage.delta import PAPERS_TABLE_DDL_SCHEMA, DeltaPaperStore, papers_to_records

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "project_config.yml"


def _fake_arxiv_result(
    arxiv_id: str = "2401.12345v2",
    title: str = "Graph RAG for Scientific Literature",
    authors: list[str] | None = None,
    summary: str = "We build a knowledge graph building on 2305.00001 and 2308.99999v1.",
    published: datetime | None = None,
    updated: datetime | None = None,
    categories: list[str] | None = None,
) -> SimpleNamespace:
    """Create a lightweight fake `arxiv.Result` object for offline testing."""
    pub = published or datetime(2024, 5, 10, 12, 0, tzinfo=UTC)
    upd = updated or datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
    author_objs = [SimpleNamespace(name=a) for a in (authors or ["Alice Smith", "Bob Jones"])]
    clean_id = canonical_arxiv_id(arxiv_id)
    return SimpleNamespace(
        entry_id=f"http://arxiv.org/abs/{arxiv_id}",
        title=title,
        authors=author_objs,
        summary=summary,
        published=pub,
        updated=upd,
        categories=categories or ["cs.CL", "cs.AI"],
        primary_category="cs.CL",
        doi="10.1234/example",
        journal_ref=None,
        pdf_url=f"https://arxiv.org/pdf/{clean_id}.pdf",
        comment="See also 2201.11111",
    )


class FakeArxivClient:
    """Stub `arxiv.Client` that records searches and returns preset results."""

    def __init__(self, results: list[SimpleNamespace]) -> None:
        self._results = results
        self.last_search: arxiv.Search | None = None

    def results(self, search: arxiv.Search) -> list[SimpleNamespace]:
        """Record the search call and return preset results."""
        self.last_search = search
        return self._results


class FakeSparkWriter:
    """In-memory stub for `DataFrame.write`."""

    def __init__(self, rows: list[dict[str, Any]], sink: dict[str, list[dict[str, Any]]]) -> None:
        self._rows = rows
        self._sink = sink
        self.mode_used: str | None = None

    def format(self, _fmt: str) -> "FakeSparkWriter":
        """No-op format setter."""
        return self

    def mode(self, mode_name: str) -> "FakeSparkWriter":
        """Record write mode."""
        self.mode_used = mode_name
        return self

    def option(self, _key: str, _val: str) -> "FakeSparkWriter":
        """No-op option setter."""
        return self

    def saveAsTable(self, table_name: str) -> None:
        """Store rows in the in-memory table dictionary."""
        self._sink.setdefault(table_name, []).extend(self._rows)


class FakeSparkSession:
    """Minimal in-memory SparkSession double for testing `DeltaPaperStore`."""

    def __init__(self) -> None:
        self.sql_queries: list[str] = []
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.last_schema: str | None = None
        self.schemas: dict[str, str] = {}
        self.columns: dict[str, list[str]] = {}
        self.views: dict[str, list[dict[str, Any]]] = {}
        self.catalog = SimpleNamespace(dropTempView=lambda name: self.views.pop(name))

    def sql(self, query: str, args: dict[str, object] | None = None) -> SimpleNamespace:
        """Execute the store's small SQL surface in memory, recording the exact statements."""
        self.sql_queries.append(query)
        names = re.findall(r"`([^`]+)`", query)
        table = ".".join(names[:3])
        selected: list[dict[str, Any]] = []
        if query.startswith("CREATE TABLE"):
            self.tables.setdefault(table, [])
            schema = query.split("(", maxsplit=1)[1].rsplit(")", maxsplit=1)[0]
            self.columns.setdefault(table, [part.strip().split()[0] for part in schema.split(",")])
        elif query.startswith("ALTER TABLE"):
            self.columns[table].append("source")
            for record in self.tables[table]:
                record.setdefault("source", None)
        elif query.startswith("SELECT *"):
            selected = [deepcopy(row) for row in self.tables[table] if row["run_id"] == (args or {})["run_id"]]
        elif query.startswith("DELETE FROM"):
            self.tables[table] = [row for row in self.tables[table] if row["run_id"] != (args or {})["run_id"]]
        elif query.startswith("MERGE WITH"):
            view = names[3]
            keys = re.findall(r"target.`([^`]+)` = incoming.", query)
            for record in self.views[view]:
                existing = next(
                    (row for row in self.tables[table] if all(row[key] == record[key] for key in keys)), None
                )
                if existing is None:
                    self.tables[table].append(deepcopy(record))
                elif "target.status = 'FAILED'" in query:
                    if existing["status"] == "FAILED" and all(
                        existing[key] == record[key] for key in ("request_json", "source", "download_pdfs", "enrich")
                    ):
                        existing.update(deepcopy(record))
                elif "target.attempt_id = incoming.attempt_id" in query:
                    if existing["status"] == "RUNNING" and existing["attempt_id"] == record["attempt_id"]:
                        existing.update(deepcopy(record))
                else:
                    existing.update(deepcopy(record))
            self.schemas[table] = self.last_schema or ""
        elif not query.startswith("CREATE VOLUME"):
            raise AssertionError(f"Unsupported test SQL: {query}")
        return SimpleNamespace(
            collect=lambda: [SimpleNamespace(asDict=lambda recursive, record=row: deepcopy(record)) for row in selected]
        )

    def table(self, name: str) -> SimpleNamespace:
        """Return persisted table metadata used for key-column evolution."""
        return SimpleNamespace(columns=self.columns[name])

    def createDataFrame(self, rows: list[dict[str, Any]], schema: str | None = None) -> SimpleNamespace:
        """Return an object with a `.write` attribute and record schema."""
        self.last_schema = schema
        return SimpleNamespace(
            write=FakeSparkWriter(rows, self.tables),
            createOrReplaceTempView=lambda name: self.views.update({name: deepcopy(rows)}),
        )


def test_canonical_arxiv_id_strips_url_and_version() -> None:
    """Version suffixes and URL prefixes are removed from arXiv entry IDs."""
    assert canonical_arxiv_id("http://arxiv.org/abs/2401.12345v3") == "2401.12345"
    assert canonical_arxiv_id("2305.00001") == "2305.00001"


def test_parse_date_bounds_supports_single_and_range() -> None:
    """Single dates, ranges, and year strings parse into expected date bounds."""
    assert parse_date_bounds(None) == (None, None)
    assert parse_date_bounds("2024-06-01") == (None, date(2024, 6, 1))
    assert parse_date_bounds("2024-01-01..2024-06-30") == (date(2024, 1, 1), date(2024, 6, 30))
    assert parse_date_bounds(">=2024-03-01") == (date(2024, 3, 1), None)
    assert parse_date_bounds("2024") == (date(2024, 1, 1), date(2024, 12, 31))

    with pytest.raises(ValueError, match="Invalid date range"):
        parse_date_bounds("2024-07-01..2024-01-01")


def test_format_arxiv_query_with_categories_and_date() -> None:
    """Query builder combines topic, categories, and date window when criterion is date-based."""
    req = SearchRequest(
        topic="graph neural networks",
        num_papers=5,
        ranking_criteria=RankingCriteria.SUBMITTED_DATE,
        criteria_date="2024-01-01..2024-06-30",
        categories=["cs.LG", "cs.AI"],
    )
    query = format_arxiv_query(req)
    assert "(graph neural networks)" in query
    assert "(cat:cs.LG OR cat:cs.AI)" in query
    assert "submittedDate:[202401010000 TO 202406302359]" in query


def test_arxiv_source_implements_protocol_and_filters_by_criteria() -> None:
    """ArxivSource satisfies PaperSource, maps criteria, deduplicates, and extracts citations."""
    r1 = _fake_arxiv_result(
        arxiv_id="2405.00001v1",
        published=datetime(2024, 5, 1, tzinfo=UTC),
        updated=datetime(2024, 5, 20, tzinfo=UTC),
    )
    r1_dup = _fake_arxiv_result(
        arxiv_id="2405.00001v2",
        published=datetime(2024, 5, 1, tzinfo=UTC),
        updated=datetime(2024, 5, 22, tzinfo=UTC),
    )
    r2_too_new = _fake_arxiv_result(
        arxiv_id="2408.00002v1",
        published=datetime(2024, 8, 1, tzinfo=UTC),
        updated=datetime(2024, 8, 2, tzinfo=UTC),
    )

    fake_client = FakeArxivClient([r1, r1_dup, r2_too_new])
    source = ArxivSource(client=fake_client)  # type: ignore[arg-type]
    assert isinstance(source, PaperSource)

    req = SearchRequest(
        topic="knowledge graphs",
        num_papers=5,
        ranking_criteria=RankingCriteria.LAST_UPDATED_DATE,
        criteria_date="2024-06-01",
    )
    papers = source.search(req)

    assert fake_client.last_search is not None
    assert fake_client.last_search.sort_by == arxiv.SortCriterion.LastUpdatedDate
    assert len(papers) == 1
    assert papers[0].paper_id == "2405.00001"
    assert "cited_paper_ids" not in papers[0].model_dump()


def test_run_ingestion_downloads_pdfs_and_stores_in_delta(tmp_path: Path) -> None:
    """End-to-end ingestion downloads PDFs to the Volume path and writes rows to Delta."""
    cfg = ProjectConfig.from_yaml(str(PROJECT_CONFIG), "dev")
    fake_spark = FakeSparkSession()
    store = DeltaPaperStore(config=cfg, spark=fake_spark, volume_root_override=tmp_path)  # type: ignore[arg-type]

    fake_client = FakeArxivClient([_fake_arxiv_result(arxiv_id="2401.12345v1")])
    source = ArxivSource(client=fake_client)  # type: ignore[arg-type]

    def _fake_download(_url: str, dest_file: Path) -> None:
        dest_file.write_bytes(b"%PDF-1.4 fake pdf content")

    with patch.object(source, "_download_file", side_effect=_fake_download):
        result = run_ingestion(
            topic="agentic workflows",
            num_papers=3,
            ranking_criteria="relevance",
            config=cfg,
            source=source,
            store=store,
            download_pdfs=True,
            enrich=False,
        )

    assert len(result.papers) == 1
    paper: Paper = result.papers[0]
    assert paper.pdf_volume_path is not None
    assert Path(paper.pdf_volume_path).exists()

    written_rows = fake_spark.tables["llmops_dev.riccardo_tambone.papers"]
    assert fake_spark.schemas["llmops_dev.riccardo_tambone.papers"] == PAPERS_TABLE_DDL_SCHEMA
    assert len(written_rows) == 1
    assert written_rows[0]["paper_id"] == "2401.12345"
    assert written_rows[0]["ranking_criteria"] == "relevance"
    assert written_rows[0]["criteria_date"] is None
    assert written_rows[0]["pdf_volume_path"] == paper.pdf_volume_path


def test_build_run_id_and_records_serialization() -> None:
    """Run IDs and Delta records preserve expected fields."""
    fixed_now = datetime(2026, 10, 21, 14, 30, 0, tzinfo=UTC)
    run_id = build_run_id("Graph RAG & LLMs!", now=fixed_now)
    assert run_id.startswith("20261021_143000_graph_rag_llms_")
    assert run_id != build_run_id("Graph RAG & LLMs!", now=fixed_now)

    req = SearchRequest(topic="Graph RAG", num_papers=1)
    records = papers_to_records([], run_id=run_id, request=req)
    assert records == []
