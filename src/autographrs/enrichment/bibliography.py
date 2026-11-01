"""Publish external abstract snapshots without modifying acquired bibliography rows."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field

from autographrs.enrichment.abstracts import AbstractMetadata, AbstractResolver
from autographrs.graph.builder import build_citation_graph
from autographrs.graph.models import GRAPH_BUILDER_VERSION, CitationGraph, GraphNode
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
CACHE_DDL = "run_id STRING, cache_key STRING, metadata_json STRING, cached_at TIMESTAMP"
CACHE_PARTITION = "abstract-cache-v1"


def _normalize_metadata_timestamps(item: AbstractMetadata) -> AbstractMetadata:
    """Restore instants after Spark's Python TIMESTAMP conversion strips timezone information."""
    # Naive Python timestamps collected by Spark use the driver's local timezone.
    return item.model_copy(
        update={
            "fetched_at": item.fetched_at.astimezone(UTC),
            "attempts": [
                attempt.model_copy(update={"fetched_at": attempt.fetched_at.astimezone(UTC)})
                for attempt in item.attempts
            ],
        }
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
    def full_cache_table_name(self) -> str:
        """Return the environment-local cache, independent of immutable run snapshots."""
        return f"{self.config.full_schema_name}.bibliography_abstract_cache"

    @staticmethod
    def _cache_key(node: GraphNode) -> str | None:
        if node.kind == "unresolved" or not node.identifiers:
            return None
        # Require the entire reconciled alias set, not just a potentially conflicting single alias.
        payload = json.dumps([GRAPH_BUILDER_VERSION, sorted(set(node.identifiers))])
        return hashlib.sha256(payload.encode()).hexdigest()

    def _fetch_cached(self, nodes: list[GraphNode], resolver: AbstractResolver) -> list[AbstractMetadata]:
        if not self.config.bibliography_cache_days:
            return resolver.fetch(nodes)
        self._ensure_table(self.full_cache_table_name, CACHE_DDL)
        cached = {}
        cutoff = datetime.now(UTC) - timedelta(days=self.config.bibliography_cache_days)
        for row in self._read_rows(self.full_cache_table_name, CACHE_PARTITION):
            if row["cache_key"] in cached:
                raise ValueError("Duplicate bibliography cache keys")
            item = _normalize_metadata_timestamps(AbstractMetadata.model_validate_json(row["metadata_json"]))
            if item.status != "ok" or item.text_variant != "title_abstract" or not item.abstract or not item.title:
                raise ValueError("Invalid successful abstract cache entry")
            cached[row["cache_key"]] = item
        hits = {}
        for node in nodes:
            key = self._cache_key(node)
            item = cached.get(key)
            if item and cutoff <= item.fetched_at <= datetime.now(UTC):
                hits[node.node_id] = item.model_copy(update={"node_id": node.node_id}, deep=True)
        pending = [node for node in nodes if node.node_id not in hits]
        fetched = resolver.fetch(pending) if pending else []
        if {item.node_id for item in fetched} != {node.node_id for node in pending} or len(fetched) != len(pending):
            raise ValueError("Abstract resolver must return exactly one outcome per cited node")
        by_node = {node.node_id: node for node in pending}
        records = []
        for item in fetched:
            key = self._cache_key(by_node[item.node_id])
            if key and item.status == "ok" and item.text_variant == "title_abstract" and item.title and item.abstract:
                records.append(
                    {
                        "run_id": CACHE_PARTITION,
                        "cache_key": key,
                        "metadata_json": item.model_dump_json(),
                        "cached_at": datetime.now(UTC),
                    }
                )
        self._merge_records(records, self.full_cache_table_name, CACHE_DDL, ("run_id", "cache_key"))
        logger.info("Bibliography abstract cache: {} hits, {} provider lookups", len(hits), len(pending))
        return sorted([*hits.values(), *fetched], key=lambda item: item.node_id)

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
            _normalize_metadata_timestamps(AbstractMetadata.model_validate(row))
            for row in self._read_rows(self.full_metadata_table_name, run_id)
            if row["snapshot_id"] == snapshot_id
        ]
        if len(records) != snapshots[0].record_count or len({item.node_id for item in records}) != len(records):
            raise ValueError("Bibliography metadata snapshot counts or keys are inconsistent")
        return sorted(records, key=lambda item: item.node_id)

    def load_graph_input(self, run_id: str, snapshot_id: str) -> CitationGraph:
        """Resolve identical documents for embedding and graph stages without publishing a graph."""
        graph = build_citation_graph(run_id, self.load_run_dataset(run_id))
        metadata = {item.node_id: item for item in self.load_metadata(run_id, snapshot_id)}
        if set(metadata) != {node.node_id for node in graph.nodes if node.kind != "selected"}:
            raise ValueError("Bibliography metadata keys do not match this graph")
        graph.bibliography_snapshot_id = snapshot_id
        for node in graph.nodes:
            if node.node_id in metadata:
                item = metadata[node.node_id]
                node.title = item.title or node.title
                node.abstract = item.abstract
                node.text_variant = item.text_variant
                if item.status != "ok":
                    graph.warnings.append(f"{node.node_id}: abstract {item.status}")
        graph.warnings = sorted(set(graph.warnings))
        return graph

    def enrich_bibliography(self, run_id: str, resolver: AbstractResolver | None = None) -> BibliographySnapshot:
        """Retrieve metadata for one-hop cited works; preserve originals and earlier snapshots."""
        graph = build_citation_graph(run_id, self.load_run_dataset(run_id))
        nodes = [node for node in graph.nodes if node.kind != "selected"]
        snapshot = BibliographySnapshot(run_id=run_id, snapshot_id=uuid4().hex)
        self._save_snapshot(snapshot)
        try:
            metadata = self._fetch_cached(nodes, resolver or AbstractResolver())
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
                _normalize_metadata_timestamps(AbstractMetadata.model_validate(row))
                for row in self._read_rows(self.full_metadata_table_name, run_id)
                if row["snapshot_id"] == snapshot.snapshot_id
            ]
            expected = {item.node_id: _normalize_metadata_timestamps(item) for item in metadata}
            actual = {item.node_id: item for item in stored}
            if len(stored) != len(metadata) or actual != expected:
                differences = {
                    node_id: (
                        ["missing_or_unexpected_node"]
                        if node_id not in actual or node_id not in expected
                        else [
                            field
                            for field in AbstractMetadata.model_fields
                            if getattr(actual[node_id], field) != getattr(expected[node_id], field)
                        ]
                    )
                    for node_id in sorted(set(actual) | set(expected))
                    if actual.get(node_id) != expected.get(node_id)
                }
                raise ValueError(f"Persisted bibliography metadata differs from retrieved data: {differences}")
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
