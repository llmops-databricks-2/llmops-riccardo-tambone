"""Read-only bibliographic tools, reused directly by the assistant and exposed through MCP."""

import json
import threading
from collections.abc import Callable

from databricks.sdk import WorkspaceClient
from pydantic import BaseModel, ValidationError

from autographrs.assistant.models import (
    AnalyticsArguments,
    AssistantOperationError,
    AssistantSettings,
    BibliographyArguments,
    NeighborArguments,
    PaperArguments,
    RecoverableToolError,
    SearchArguments,
    ToolArgumentsError,
)
from autographrs.assistant.sql import SqlReader, qualified
from autographrs.embeddings.gateway import GatewayEmbedder
from autographrs.embeddings.lakebase import LakebasePgVectorManager
from autographrs.embeddings.models import validate_vector
from autographrs.graph.models import GraphEdge, GraphNode
from autographrs.models import BibliographicReference

MAX_TOOL_RESULT_CHARS = 64000

ARGUMENTS: dict[str, type[BaseModel]] = {
    "search_collection": SearchArguments,
    "get_paper": PaperArguments,
    "get_bibliography": BibliographyArguments,
    "get_neighbors": NeighborArguments,
    "ask_collection_analytics": AnalyticsArguments,
}
DESCRIPTIONS = {
    "search_collection": "Semantic top-k across selected papers, external works and unresolved references. "
    "Check kind/text_variant; results are candidates, not an exhaustive topic filter or citation ranking.",
    "get_paper": "Read exact paper metadata, abstract, identifiers and provenance by node_id.",
    "get_bibliography": "Read original acquired references, including unresolved ones, and provider coverage. Supports pagination.",
    "get_neighbors": "Explore local CITES links: outgoing means references, incoming means citing papers. "
    "Default/max 100 neighbors, depth <=2. Compact node metadata; use get_paper for abstracts. "
    "Not a complete citation count when truncated.",
    "ask_collection_analytics": "Use Genie for exhaustive counts, topic filters, citation rankings and listing "
    "papers. 'Papers' are the selected (acquired) papers. Citation rankings use only local incoming CITES "
    "from selected papers over any node kind; global citation counts are unavailable. Include the topic.",
}


class BibliographicTools:
    """Bind all calls to deployment-approved data, never to scope supplied by an LLM or a client."""

    def __init__(
        self,
        settings: AssistantSettings,
        workspace: WorkspaceClient,
        *,
        sql_workspace: Callable[[], WorkspaceClient] | None = None,
        gateway_workspace: Callable[[], WorkspaceClient] | None = None,
    ) -> None:
        self.settings = settings
        self.workspace = workspace
        self.sql_workspace = sql_workspace
        self.gateway_workspace = gateway_workspace
        self.reader = SqlReader(settings.config, workspace, workspace_provider=sql_workspace)
        self._slots = threading.BoundedSemaphore(2)

    def table(self, suffix: str) -> str:
        """Resolve a deployment-owned view rather than accept table names from callers."""
        return qualified(self.settings.config, f"{self.settings.view_prefix}_{suffix}")

    def invoke(self, name: str, arguments: dict) -> dict:
        """Validate before dispatch, bound concurrency and reject oversized evidence explicitly."""
        if name not in ARGUMENTS:
            raise RecoverableToolError(
                "unknown_tool", "Unknown or unavailable tool", hint=f"Available tools: {', '.join(ARGUMENTS)}"
            )
        try:
            args = ARGUMENTS[name].model_validate(arguments)
        except ValidationError as exc:
            fields = ARGUMENTS[name].model_fields
            issues = [
                {
                    "field": error["loc"][0] if error["loc"] and error["loc"][0] in fields else "unknown",
                    "type": error["type"],
                }
                for error in exc.errors(include_url=False, include_context=False, include_input=False)[:8]
            ]
            raise ToolArgumentsError(name, issues) from None
        if not self._slots.acquire(blocking=False):
            raise AssistantOperationError("tool_capacity", "Tool capacity reached; retry shortly")
        try:
            if isinstance(args, SearchArguments):
                result = self.search(args)
            elif isinstance(args, NeighborArguments):
                result = self.neighbors(args)
            elif isinstance(args, BibliographyArguments):
                result = self.bibliography(args)
            elif isinstance(args, PaperArguments):
                result = {"paper": self.paper(args.node_id).model_dump()}
            elif isinstance(args, AnalyticsArguments):
                result = self.analytics(args)
            else:
                raise TypeError("Unsupported tool argument contract")
            result["scope"] = {
                "run_id": self.settings.graph.run_id,
                "graph_build_id": self.settings.graph.graph_build_id,
                "embedding_snapshot_id": self.settings.snapshot.snapshot_id,
            }
            if len(json.dumps(result, ensure_ascii=False)) > MAX_TOOL_RESULT_CHARS:
                raise RecoverableToolError(
                    "tool_response_too_large",
                    f"Tool response exceeds {MAX_TOOL_RESULT_CHARS} characters",
                    tool=name,
                    hint="Repeat the call with a smaller limit, fewer hops or a narrower query.",
                )
            return result
        finally:
            self._slots.release()

    def paper(self, node_id: str) -> GraphNode:
        """Require an exact node in this collection."""
        rows = self.reader.read(
            f"SELECT to_json(struct(p.*)) AS payload FROM {self.table('nodes')} p WHERE node_id = :node_id LIMIT 1",
            {"node_id": node_id},
            limit=1,
        )
        if len(rows) != 1:
            raise RecoverableToolError(
                "node_not_found",
                "Node not found in the approved collection",
                hint="Use an exact node_id returned by a previous tool result; search_collection finds node_ids.",
            )
        paper = GraphNode.model_validate(rows[0])
        if paper.node_id != node_id:
            raise ValueError("SQL returned a paper outside the requested node identity")
        return paper

    def search(self, args: SearchArguments) -> dict:
        """Use the original embedding space and a verified committed pgvector publication."""
        if not self.settings.snapshot.document_count:
            return {
                "papers": [],
                "metric": "cosine_similarity",
                "coverage": "The approved snapshot contains no indexable documents.",
            }
        from autographrs.enrichment.llm import create_gateway_openai_client

        config = self.settings.config
        # Refresh Gateway OAuth for each operation instead of caching an expiring access token in the App.
        gateway_workspace = self.gateway_workspace() if self.gateway_workspace is not None else self.workspace
        with create_gateway_openai_client(gateway_workspace) as client:
            vectors, response_model = GatewayEmbedder(config, client, timeout_seconds=20, max_retries=0).embed(
                [args.query]
            )
        if response_model != self.settings.response_model:
            raise ValueError("Query model differs from the published collection; rebuild with a new revision")
        validate_vector(vectors[0], self.settings.snapshot.dimension)
        results = LakebasePgVectorManager(
            config.model_copy(update={"search_timeout_seconds": 20}),
            self.workspace,
        ).search_published(self.settings.snapshot, self.settings.fingerprint, vectors[0], args.limit)
        return {
            "papers": [
                {
                    "node_id": row["node_id"],
                    "kind": row["kind"],
                    "title": row["title"],
                    "text_variant": row["text_variant"],
                    "text_preview": row["text"][:800],
                    "text_preview_truncated": len(row["text"]) > 800,
                    "cosine_similarity": row["cosine_similarity"],
                }
                for row in results
            ],
            "metric": "cosine_similarity",
            "coverage": "Semantic top-k across all node kinds, not an exhaustive topic selection. "
            "Unresolved/external nodes may lack abstracts and acquired bibliographies.",
        }

    def bibliography(self, args: BibliographyArguments) -> dict:
        """Return acquired references including unresolved occurrences, provider status and pagination."""
        paper = self.paper(args.node_id)
        if paper.kind != "selected":
            return {"references": [], "has_more": False, "coverage": "not_acquired_for_external_or_unresolved"}
        rows = self.reader.read(
            "SELECT to_json(named_struct("
            f"'references',slice(bibliography,{args.offset + 1},{args.limit}),"
            "'total',coalesce(size(bibliography),0),"
            "'citation_source',citation_source,'citation_status',citation_status,"
            "'citation_error',citation_error,'citation_lookups',citation_lookups)) AS payload "
            f"FROM {self.table('bibliography')} WHERE node_id=:node_id LIMIT 1",
            {"node_id": args.node_id},
            limit=1,
        )
        if len(rows) != 1:
            raise RecoverableToolError(
                "bibliography_not_published",
                "Selected paper has no published bibliography record",
                hint="Answer from get_paper or graph neighbors and state that the bibliography is unavailable.",
            )
        row = rows[0]
        has_more = args.offset + args.limit < row["total"]
        return {
            **row,
            "references": [BibliographicReference.model_validate(ref).model_dump() for ref in row["references"] or []],
            "has_more": has_more,
            "next_offset": args.offset + args.limit if has_more else None,
            "coverage": "acquired_bibliography",
        }

    def neighbors(self, args: NeighborArguments) -> dict:
        """Traverse at most two hops, returning explicit truncation and actual edge evidence."""
        self.paper(args.node_id)
        frontier, visited = {args.node_id}, {args.node_id}
        nodes: list[dict] = []
        edges: dict[str, dict] = {}
        truncated = False
        for _ in range(args.depth):
            parameters = {f"node{index}": value for index, value in enumerate(sorted(frontier))}
            placeholders = ", ".join(f":{key}" for key in parameters)
            clauses = []
            if args.direction in ("outgoing", "both"):
                clauses.append(f"source_id IN ({placeholders})")
            if args.direction in ("incoming", "both"):
                clauses.append(f"target_id IN ({placeholders})")
            rows = self.reader.read(
                f"SELECT to_json(struct(e.*)) AS payload FROM {self.table('citations')} e "
                f"WHERE ({' OR '.join(clauses)}) ORDER BY edge_id LIMIT 100",
                parameters,
                limit=100,
            )
            if len(rows) == 100:
                more = self.reader.read(
                    f"SELECT to_json(struct(e.*)) AS payload FROM {self.table('citations')} e "
                    f"WHERE ({' OR '.join(clauses)}) AND edge_id > :after ORDER BY edge_id LIMIT 1",
                    {**parameters, "after": rows[-1]["edge_id"]},
                    limit=1,
                )
                truncated = truncated or bool(more)
            next_frontier = set()
            for row in rows:
                edge = GraphEdge.model_validate(row)
                candidates = {edge.source_id, edge.target_id} - visited
                if candidates and len(visited) - 1 >= args.limit:
                    truncated = True
                    continue
                for node_id in sorted(candidates):
                    if len(visited) - 1 >= args.limit:
                        truncated = True
                        break
                    visited.add(node_id)
                    next_frontier.add(node_id)
                if edge.source_id in visited and edge.target_id in visited:
                    edges[edge.edge_id] = edge.model_dump()
            frontier = next_frontier
            if frontier:
                node_parameters = {f"id{index}": value for index, value in enumerate(sorted(frontier))}
                placeholders = ", ".join(f":{key}" for key in node_parameters)
                found = self.reader.read(
                    "SELECT to_json(named_struct('node_id',p.node_id,'kind',p.kind,'title',p.title,"
                    "'publication_year',p.publication_year,"
                    "'abstract_available',p.abstract IS NOT NULL AND length(trim(p.abstract))>0)) AS payload "
                    f"FROM {self.table('nodes')} p "
                    f"WHERE node_id IN ({placeholders}) ORDER BY node_id LIMIT 100",
                    node_parameters,
                    limit=100,
                )
                parsed = [GraphNode.model_validate(row) for row in found]
                if {node.node_id for node in parsed} != frontier:
                    raise ValueError("Published graph traversal returned missing or foreign nodes")
                nodes.extend(
                    {
                        **node.model_dump(include={"node_id", "kind", "title", "publication_year"}),
                        "abstract_available": row.get("abstract_available", bool(node.abstract)),
                    }
                    for node, row in zip(parsed, found, strict=True)
                )
            if not frontier:
                break
        return {
            "nodes": nodes,
            "edges": list(edges.values()),
            "truncated": truncated,
            "relation": "CITES",
            "direction": args.direction,
            "coverage": "Local exploration only; outgoing bibliographies were acquired only for selected papers. "
            "Nodes contain compact metadata; use get_paper for abstracts. Not a complete ranking.",
        }

    def analytics(self, args: AnalyticsArguments) -> dict:
        """Use only the configured Genie MCP; never accept an MCP URL or arbitrary SQL from a caller."""
        from autographrs.assistant.genie import ask_genie

        return {
            "genie": ask_genie(
                self.sql_workspace() if self.sql_workspace is not None else self.workspace,
                self.settings.genie_space_id,
                args.question,
            ),
            "scope_note": "Only the configured collection views are curated in Genie.",
        }
