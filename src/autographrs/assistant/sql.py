"""Bounded SQL Statement Execution for interactive reads; no Spark runtime or model-authored SQL."""

import json
import re
import time
from collections.abc import Callable

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import (
    Disposition,
    ExecuteStatementRequestOnWaitTimeout,
    StatementParameterListItem,
    StatementState,
)

from autographrs.assistant.models import AssistantOperationError
from autographrs.config import ProjectConfig


def identifier(name: str) -> str:
    """Quote configuration-owned identifiers, rejecting anything outside the supported naming contract."""
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", name):
        raise ValueError("Invalid SQL identifier")
    return f"`{name}`"


def qualified(config: ProjectConfig, name: str) -> str:
    """Use only the configured Unity Catalog schema."""
    return ".".join(identifier(part) for part in (config.catalog, config.db_schema, name))


class SqlReader:
    """Execute fixed parameterized SELECT statements with explicit timeout and size barriers."""

    def __init__(
        self,
        config: ProjectConfig,
        workspace: WorkspaceClient,
        *,
        workspace_provider: Callable[[], WorkspaceClient] | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.workspace_provider = workspace_provider

    def read(self, statement: str, parameters: dict[str, str] | None = None, limit: int = 100) -> list[dict]:
        """Return JSON payload rows, never silently accept truncation or partial result chunks."""
        if not statement.lstrip().upper().startswith("SELECT ") or not 1 <= limit <= 100:
            raise ValueError("Interactive SQL requires a bounded SELECT")
        workspace = self.workspace_provider() if self.workspace_provider is not None else self.workspace
        response = workspace.statement_execution.execute_statement(
            statement=statement,
            warehouse_id=self.config.warehouse_id,
            parameters=[
                StatementParameterListItem(name=key, value=value, type="STRING")
                for key, value in (parameters or {}).items()
            ],
            disposition=Disposition.INLINE,
            row_limit=limit,
            byte_limit=1_000_000,
            wait_timeout="10s",
            on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
        )
        deadline = time.monotonic() + 25
        while response.status and response.status.state in (StatementState.PENDING, StatementState.RUNNING):
            if not response.statement_id:
                raise ValueError("SQL execution response has no statement ID")
            if time.monotonic() >= deadline:
                workspace.statement_execution.cancel_execution(response.statement_id)
                raise TimeoutError("SQL read exceeded its interactive timeout and was cancelled")
            time.sleep(0.5)
            response = workspace.statement_execution.get_statement(response.statement_id)
        if not response.status or response.status.state != StatementState.SUCCEEDED:
            raise AssistantOperationError("sql_read_failed", "SQL read failed; check Warehouse and view access")
        if not response.manifest or response.manifest.truncated or response.result is None:
            raise ValueError("SQL read returned missing or truncated results")
        result = response.result
        if result.next_chunk_index is not None or result.next_chunk_internal_link:
            raise ValueError("SQL result exceeds the single-chunk interactive limit; request a smaller page")
        rows = result.data_array or []
        if len(rows) > limit:
            raise ValueError("SQL returned too many rows")
        parsed = []
        for row in rows:
            if len(row) != 1 or row[0] is None:
                raise ValueError("SQL response does not match the JSON payload contract")
            payload = json.loads(row[0])
            if not isinstance(payload, dict):
                raise ValueError("SQL payload must be an object")
            parsed.append(payload)
        return parsed
