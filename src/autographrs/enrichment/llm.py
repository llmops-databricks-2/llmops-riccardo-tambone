"""AI Gateway client helper for Databricks LLMOps course."""

import os

from databricks.sdk import WorkspaceClient
from openai import OpenAI


def create_gateway_openai_client(workspace_client: WorkspaceClient | None = None) -> OpenAI:
    """Create an OpenAI client authenticated against the Databricks AI Gateway.

    Uses `/ai-gateway/mlflow/v1` on the current workspace host and the workspace
    OAuth/PAT token resolved by `databricks-sdk` (or `DATABRICKS_TOKEN` env var).

    Args:
        workspace_client: Optional pre-initialized `WorkspaceClient`

    Returns:
        Configured `openai.OpenAI` client

    """
    ws = workspace_client or WorkspaceClient()
    host = (ws.config.host or os.getenv("DATABRICKS_HOST", "")).rstrip("/")
    headers = ws.config.authenticate()
    bearer = headers.get("Authorization", "").removeprefix("Bearer ").strip()
    api_key = bearer or os.getenv("DATABRICKS_TOKEN", "databricks-token")

    return OpenAI(
        base_url=f"{host}/ai-gateway/mlflow/v1",
        api_key=api_key,
    )
