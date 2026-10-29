"""Publish external abstract snapshots without modifying acquired bibliography rows."""

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field

from autographrs.enrichment.abstracts import AbstractMetadata, AbstractResolver
from autographrs.graph.builder import build_citation_graph
from autographrs.graph.models import GRAPH_BUILDER_VERSION
from autographrs.storage.delta import DeltaPaperStore, _validate_run_id

METADATA_DDL = (
    "run_id STRING, snapshot_id STRING, node_id STRING, title STRING, abstract STRING, text_variant STRING, "
    "status STRING, provider STRING, evidence_url STRING, fetched_at TIMESTAMP, "
    "attempts ARRAY<STRUCT<source: STRING, status: STRING, error: STRING, fetched_at: TIMESTAMP>>"
)
SNAPSHOT_DDL = (
    "run_id STRING, snapshot_id STRING, builder_version STRING, status STRING, record_count INT, "
    "warnings ARRAY<STRING>, error STRING, started_at TIMESTAMP, completed_at TIMESTAMP"
)


class BibliographySnapshot(BaseModel):
    """Publication barrier for one immutable set of external metadata outcomes."""

    run_id: str
    snapshot_id: str
    builder_version: str = GRAPH_BUILDER_VERSION
    status: Literal["RUNNING", "COMPLETED", "FAILED"] = "RUNNING"
    record_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


class BibliographyStore(DeltaPaperStore):
    """Run-scoped metadata acquisition and explicitly published snapshots."""

    @property
    def full_metadata_table_name(self) -> str:
        """Return the external metadata table."""
        return f"{self.config.full_schema_name}.bibliography_metadata"

    @property
    def full_snapshots_table_name(self) -> str:
        """Return the metadata snapshot registry."""
        return f"{self.config.full_schema_name}.bibliography_snapshots"

    def _save_snapshot(self, snapshot: BibliographySnapshot) -> None:
        self._merge_records(
            [snapshot.model_dump()], self.full_snapshots_table_name, SNAPSHOT_DDL, ("run_id", "snapshot_id")
        )

    def load_metadata(self, run_id: str, snapshot_id: str) -> list[AbstractMetadata]:
        """Read only one complete snapshot compatible with the current identity rules."""
        _validate_run_id(snapshot_id)
        snapshots = [
            BibliographySnapshot.model_validate(row)
            for row in self._read_rows(self.full_snapshots_table_name, run_id)
            if row["snapshot_id"] == snapshot_id
        ]
        if len(snapshots) != 1 or snapshots[0].status != "COMPLETED":
            raise ValueError("Bibliography metadata snapshot is not published")
        if snapshots[0].builder_version != GRAPH_BUILDER_VERSION:
            raise ValueError("Bibliography snapshot identity rules are incompatible; rebuild metadata")
        records = [
            AbstractMetadata.model_validate(row)
            for row in self._read_rows(self.full_metadata_table_name, run_id)
            if row["snapshot_id"] == snapshot_id
        ]
        if len(records) != snapshots[0].record_count or len({item.node_id for item in records}) != len(records):
            raise ValueError("Bibliography metadata snapshot counts or keys are inconsistent")
        return sorted(records, key=lambda item: item.node_id)

    def enrich_bibliography(self, run_id: str, resolver: AbstractResolver | None = None) -> BibliographySnapshot:
        """Retrieve metadata for one-hop cited works; preserve originals and earlier snapshots."""
        graph = build_citation_graph(run_id, self.load_run_dataset(run_id))
        nodes = [node for node in graph.nodes if node.kind != "selected"]
        snapshot = BibliographySnapshot(run_id=run_id, snapshot_id=uuid4().hex)
        self._save_snapshot(snapshot)
        try:
            metadata = (resolver or AbstractResolver()).fetch(nodes)
            if {item.node_id for item in metadata} != {node.node_id for node in nodes} or len(metadata) != len(nodes):
                raise ValueError("Abstract resolver must return exactly one outcome per cited node")
            snapshot.record_count = len(metadata)
            snapshot.warnings = [f"{item.node_id}: abstract {item.status}" for item in metadata if item.status != "ok"]
            for warning in snapshot.warnings:
                logger.warning("Bibliography {}: {}", snapshot.snapshot_id, warning)
            self._ensure_table(self.full_metadata_table_name, METADATA_DDL)
            self._merge_records(
                [{"run_id": run_id, "snapshot_id": snapshot.snapshot_id, **item.model_dump()} for item in metadata],
                self.full_metadata_table_name,
                METADATA_DDL,
                ("run_id", "snapshot_id", "node_id"),
            )
            stored = [
                AbstractMetadata.model_validate(row)
                for row in self._read_rows(self.full_metadata_table_name, run_id)
                if row["snapshot_id"] == snapshot.snapshot_id
            ]
            if sorted(stored, key=lambda item: item.node_id) != sorted(metadata, key=lambda item: item.node_id):
                raise ValueError("Persisted bibliography metadata differs from retrieved data")
            snapshot.status = "COMPLETED"
            snapshot.completed_at = datetime.now(UTC)
            self._save_snapshot(snapshot)
        except Exception as exc:
            snapshot.status = "FAILED"
            snapshot.error = f"{type(exc).__name__}: {exc}"
            snapshot.completed_at = datetime.now(UTC)
            logger.exception("Bibliography enrichment failed for run {}", run_id)
            self._save_snapshot(snapshot)
            raise
        return snapshot
