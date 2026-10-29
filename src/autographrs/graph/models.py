"""Typed graph snapshots and publication metadata."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

GRAPH_BUILDER_VERSION = "citations-v1"


class GraphNode(BaseModel):
    """A selected paper, identified external work, or source-local reference."""

    node_id: str
    kind: Literal["selected", "external", "unresolved"]
    identity_status: Literal["identified", "missing", "ambiguous"] = "identified"
    identifiers: list[str] = Field(default_factory=list)
    source: str | None = None
    paper_id: str | None = None
    title: str
    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    urls: list[str] = Field(default_factory=list)
    abstract: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    arxiv_category: list[str] = Field(default_factory=list)
    pdf_volume_path: str | None = None


class CitationEvidence(BaseModel):
    """A bibliography occurrence supporting one directed edge."""

    source: str
    paper_id: str
    reference_id: str
    position: int
    text: str
    provider: str | None = None
    evidence_url: str


class GraphEdge(BaseModel):
    """One directed citation, retaining all supporting bibliography occurrences."""

    edge_id: str
    source_id: str
    target_id: str
    relation: Literal["CITES"] = "CITES"
    evidence: list[CitationEvidence] = Field(min_length=1)


class CitationGraph(BaseModel):
    """A reproducible, in-memory graph without storage or network dependencies."""

    run_id: str
    builder_version: str = GRAPH_BUILDER_VERSION
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def validate_integrity(self) -> None:
        """Reject duplicate IDs, dangling endpoints, self-citations and external citing nodes."""
        nodes = {node.node_id: node for node in self.nodes}
        if len(nodes) != len(self.nodes) or len({edge.edge_id for edge in self.edges}) != len(self.edges):
            raise ValueError("Graph contains duplicate node or edge IDs")
        pairs = {(edge.source_id, edge.target_id) for edge in self.edges}
        if len(pairs) != len(self.edges):
            raise ValueError("Graph contains duplicate directed citations")
        for edge in self.edges:
            if edge.source_id not in nodes or edge.target_id not in nodes:
                raise ValueError("Graph contains a dangling edge")
            if edge.source_id == edge.target_id or nodes[edge.source_id].kind != "selected":
                raise ValueError("Graph contains a self-citation or an external citing node")


class GraphBuild(BaseModel):
    """Publication marker; only COMPLETED builds may be consumed."""

    run_id: str
    graph_build_id: str
    builder_version: str = GRAPH_BUILDER_VERSION
    status: Literal["RUNNING", "COMPLETED", "FAILED"] = "RUNNING"
    node_count: int = 0
    edge_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
