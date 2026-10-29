"""Versioned Delta publication for citation graphs, separate from ingestion status."""

from datetime import UTC, datetime
from uuid import uuid4

from loguru import logger

from autographrs.graph.builder import build_citation_graph
from autographrs.graph.models import CitationGraph, GraphBuild, GraphEdge, GraphNode
from autographrs.storage.delta import DeltaPaperStore, _validate_run_id

GRAPH_NODES_DDL_SCHEMA = (
    "run_id STRING, graph_build_id STRING, node_id STRING, kind STRING, identity_status STRING, "
    "identifiers ARRAY<STRING>, source STRING, paper_id STRING, title STRING, authors ARRAY<STRING>, "
    "publication_year INT, urls ARRAY<STRING>, abstract STRING, summary STRING, tags ARRAY<STRING>, "
    "arxiv_category ARRAY<STRING>, pdf_volume_path STRING"
)
GRAPH_EDGES_DDL_SCHEMA = (
    "run_id STRING, graph_build_id STRING, edge_id STRING, source_id STRING, target_id STRING, relation STRING, "
    "evidence ARRAY<STRUCT<source: STRING, paper_id: STRING, reference_id: STRING, position: INT, "
    "text: STRING, provider: STRING, evidence_url: STRING>>"
)
GRAPH_BUILDS_DDL_SCHEMA = (
    "run_id STRING, graph_build_id STRING, builder_version STRING, status STRING, node_count INT, "
    "edge_count INT, warnings ARRAY<STRING>, error STRING, started_at TIMESTAMP, completed_at TIMESTAMP"
)


class GraphStore(DeltaPaperStore):
    """Reuse run-scoped Delta access while keeping graph builds in independent tables."""

    @property
    def full_nodes_table_name(self) -> str:
        """Return the graph node table in the configured environment."""
        return f"{self.config.full_schema_name}.graph_nodes"

    @property
    def full_edges_table_name(self) -> str:
        """Return the graph edge table in the configured environment."""
        return f"{self.config.full_schema_name}.graph_edges"

    @property
    def full_builds_table_name(self) -> str:
        """Return the graph publication registry."""
        return f"{self.config.full_schema_name}.graph_builds"

    def _save_build(self, build: GraphBuild) -> None:
        self._merge_records(
            [build.model_dump()],
            self.full_builds_table_name,
            GRAPH_BUILDS_DDL_SCHEMA,
            ("run_id", "graph_build_id"),
        )

    def _write_graph(self, graph: CitationGraph, build_id: str) -> None:
        for table, schema, objects, key in (
            (self.full_nodes_table_name, GRAPH_NODES_DDL_SCHEMA, graph.nodes, "node_id"),
            (self.full_edges_table_name, GRAPH_EDGES_DDL_SCHEMA, graph.edges, "edge_id"),
        ):
            self._ensure_table(table, schema)
            records = [{"run_id": graph.run_id, "graph_build_id": build_id, **item.model_dump()} for item in objects]
            self._merge_records(records, table, schema, ("run_id", "graph_build_id", key))

    def _read_graph(self, build: GraphBuild) -> CitationGraph:
        nodes = [
            GraphNode.model_validate(row)
            for row in self._read_rows(self.full_nodes_table_name, build.run_id)
            if row["graph_build_id"] == build.graph_build_id
        ]
        edges = [
            GraphEdge.model_validate(row)
            for row in self._read_rows(self.full_edges_table_name, build.run_id)
            if row["graph_build_id"] == build.graph_build_id
        ]
        if len(nodes) != build.node_count or len(edges) != build.edge_count:
            raise ValueError(f"Graph snapshot counts do not match build {build.graph_build_id}")
        graph = CitationGraph(
            run_id=build.run_id,
            builder_version=build.builder_version,
            nodes=sorted(nodes, key=lambda node: node.node_id),
            edges=sorted(edges, key=lambda edge: (edge.source_id, edge.target_id)),
            warnings=build.warnings,
        )
        graph.validate_integrity()
        return graph

    def load_graph(self, run_id: str, graph_build_id: str) -> CitationGraph:
        """Require an explicit, completed build; never combine partial or historical builds."""
        _validate_run_id(graph_build_id)
        builds = [
            GraphBuild.model_validate(row)
            for row in self._read_rows(self.full_builds_table_name, run_id)
            if row["graph_build_id"] == graph_build_id
        ]
        if len(builds) != 1 or builds[0].status != "COMPLETED":
            raise ValueError(f"Graph build {graph_build_id} is not published")
        return self._read_graph(builds[0])

    def build_graph(self, run_id: str) -> GraphBuild:
        """Build from a completed enrichment run and publish only after read-back validation."""
        dataset = self.load_run_dataset(run_id)
        build = GraphBuild(run_id=run_id, graph_build_id=uuid4().hex)
        self._save_build(build)
        try:
            graph = build_citation_graph(run_id, dataset)
            build.node_count = len(graph.nodes)
            build.edge_count = len(graph.edges)
            build.warnings = graph.warnings
            for warning in graph.warnings:
                logger.warning("Graph {}: {}", build.graph_build_id, warning)
            self._write_graph(graph, build.graph_build_id)
            if self._read_graph(build) != graph:
                raise ValueError("Persisted graph does not match the generated snapshot")
            build.status = "COMPLETED"
            build.completed_at = datetime.now(UTC)
            self._save_build(build)
        except Exception as exc:
            build.status = "FAILED"
            build.error = f"{type(exc).__name__}: {exc}"
            build.completed_at = datetime.now(UTC)
            logger.exception("Graph build {} failed for run {}", build.graph_build_id, run_id)
            self._save_build(build)
            raise
        logger.info(
            "Published graph {} for run {}: {} nodes, {} edges",
            build.graph_build_id,
            run_id,
            build.node_count,
            build.edge_count,
        )
        return build
