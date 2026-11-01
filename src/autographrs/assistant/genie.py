"""Authenticated, bounded calls to one private Genie MCP, without MLflow wrappers."""

import asyncio
import json
from datetime import timedelta

from databricks.sdk import WorkspaceClient
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool

from autographrs.assistant.models import AssistantOperationError, RecoverableToolError


class GenieContractError(ValueError):
    """Report only structural MCP metadata, never questions, descriptions or query results."""

    def __init__(self, reason: str, tools: list[Tool]) -> None:
        schemas = []
        for tool in tools[:8]:
            required = tool.inputSchema.get("required", [])
            properties = tool.inputSchema.get("properties", {})
            types = [properties.get(name, {}).get("type") for name in required[:8]]
            schemas.append(
                {
                    "required_count": len(required),
                    "property_count": len(properties),
                    "required_types": [
                        value
                        if value in ("string", "object", "array", "integer", "number", "boolean", "null")
                        else "unspecified"
                        for value in types
                    ],
                }
            )
        self.diagnostics = {
            "operation": "genie_mcp",
            "reason": reason,
            "tool_count": len(tools),
            "schemas": schemas,
        }
        super().__init__(reason)


async def _ask(workspace: WorkspaceClient, space_id: str, question: str) -> dict:
    host = (workspace.config.host or "").rstrip("/")
    if not host.startswith("https://"):
        raise ValueError("Genie MCP requires an HTTPS workspace URL")
    async with asyncio.timeout(40):
        async with (
            streamablehttp_client(
                f"{host}/api/2.0/mcp/genie/{space_id}",
                headers=workspace.config.authenticate(),
                timeout=timedelta(seconds=10),
                sse_read_timeout=timedelta(seconds=30),
            ) as (read, write, _),
            ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session,
        ):
            await session.initialize()
            tools = (await session.list_tools()).tools
            candidates: list[tuple[Tool, str]] = []
            for tool in tools:
                required = tool.inputSchema.get("required", [])
                properties = tool.inputSchema.get("properties", {})
                if len(required) == 1 and properties.get(required[0], {}).get("type") == "string":
                    candidates.append((tool, required[0]))
            if not candidates:
                raise GenieContractError("unsupported_question_schema", tools)
            if len(candidates) != 1:
                raise GenieContractError("ambiguous_question_tools", tools)
            tool, argument = candidates[0]
            poll = _poll_tool(tools, tool)
            raw = await _call(session, tool.name, {argument: question})
            delay = 1.0
            while (
                status := str((_payload(raw) or {}).get("status") or "").upper()
            ) and status not in TERMINAL_STATUSES:
                payload = _payload(raw) or {}
                ids = (payload.get("conversationId"), payload.get("messageId"))
                if poll is None or not all(isinstance(value, str) and value for value in ids):
                    raise GenieContractError("unpollable_pending_result", tools)
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 5.0)
                raw = await _call(session, poll[0], {poll[1]: ids[0], poll[2]: ids[1]})
            if status and status != "COMPLETED":
                raise AssistantOperationError(
                    "genie_query_failed",
                    f"Genie finished with status {status}",
                    tool="ask_collection_analytics",
                    details={"genie_status": status},
                )
            return compact_genie_result(raw)


# Genie MCP answers asynchronously: the question tool may return before the message is complete.
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"}


def _poll_tool(tools: list[Tool], question: Tool) -> tuple[str, str, str] | None:
    """Find the status tool taking (conversation id, message id); None when Genie exposes none."""
    for tool in tools:
        if tool.name == question.name:
            continue
        required = tool.inputSchema.get("required", [])
        properties = tool.inputSchema.get("properties", {})
        if len(required) != 2 or any(properties.get(name, {}).get("type") != "string" for name in required):
            continue
        conversation = [name for name in required if "conversation" in name.lower()]
        message = [name for name in required if "message" in name.lower()]
        if len(conversation) == 1 and len(message) == 1 and conversation != message:
            return tool.name, conversation[0], message[0]
    return None


async def _call(session: ClientSession, name: str, arguments: dict) -> dict:
    result = await session.call_tool(name, arguments)
    if result.isError:
        raise AssistantOperationError(
            "genie_query_failed", "Genie MCP reported a query error", tool="ask_collection_analytics"
        )
    return result.model_dump(mode="json")


def _payload(raw: dict) -> dict | None:
    payload = raw.get("structuredContent")
    if isinstance(payload, dict):
        return payload
    for item in raw.get("content") or []:
        if item.get("type") == "text":
            try:
                candidate = json.loads(item.get("text") or "")
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                return candidate
    return None


MAX_GENIE_ROWS = 30
MAX_GENIE_TEXT = 4000


def compact_genie_result(raw: dict) -> dict:
    """Keep Genie's answer text, SQL and bounded rows; drop duplicated transport and statement metadata."""
    payload = _payload(raw)
    if payload is None:
        texts = [item.get("text") or "" for item in raw.get("content") or [] if item.get("type") == "text"]
        return {"text": [text[:MAX_GENIE_TEXT] for text in texts], "queries": []}
    content = payload.get("content") if isinstance(payload.get("content"), dict) else payload
    queries = []
    for attachment in content.get("queryAttachments") or []:
        statement = attachment.get("statement_response") or {}
        manifest = statement.get("manifest") or {}
        columns = [column.get("name") for column in (manifest.get("schema") or {}).get("columns") or []]
        rows = [_row(row) for row in (statement.get("result") or {}).get("data_array") or []]
        total = manifest.get("total_row_count", len(rows))
        queries.append(
            {
                "sql": attachment.get("query"),
                "state": (statement.get("status") or {}).get("state"),
                "columns": columns,
                "rows": rows[:MAX_GENIE_ROWS],
                "row_count": total,
                "truncated": bool(manifest.get("truncated")) or len(rows) > MAX_GENIE_ROWS or total > len(rows),
            }
        )
    return {
        "status": payload.get("status"),
        "text": [str(text)[:MAX_GENIE_TEXT] for text in content.get("textAttachments") or []],
        "queries": queries,
    }


def _row(row: object) -> list:
    if isinstance(row, dict):
        return [
            next(iter(value.values()), None) if isinstance(value, dict) else value for value in row.get("values", [])
        ]
    return list(row) if isinstance(row, list) else [row]


def ask_genie(workspace: WorkspaceClient, space_id: str, question: str) -> dict:
    """Run from a synchronous tool worker, using fresh SDK OAuth headers and bounded transport timeouts."""
    try:
        return asyncio.run(_ask(workspace, space_id, question))
    except* TimeoutError:
        raise RecoverableToolError(
            "genie_timeout",
            "Genie did not finish within 40 seconds",
            tool="ask_collection_analytics",
            hint="Ask a simpler analytics question once, or answer from other tools and say analytics timed out.",
        ) from None
