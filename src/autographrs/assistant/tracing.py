"""Conversation tracing to the App's explicitly provisioned MLflow experiment."""

import os
from collections.abc import Callable, Generator
from contextlib import contextmanager

import mlflow
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.entities import LiveSpan, MlflowExperimentLocation, SpanStatusCode

from autographrs.assistant.models import AssistantOperationError, AssistantSettings, ChatRequest
from autographrs.assistant.trace_permissions import read_writer_credentials
from autographrs.config import ProjectConfig


@contextmanager
def trace_span(
    name: str,
    span_type: str,
    inputs: dict,
    attributes: dict | None = None,
    *,
    experiment_id: str | None = None,
) -> Generator[LiveSpan | None, None, None]:
    """Keep upstream exception text and transport credentials out of recorded errors."""
    if experiment_id is None and mlflow.get_current_active_span() is None:
        yield None
        return
    failure: Exception | None = None
    with mlflow.start_span(
        name=name,
        span_type=span_type,
        attributes=attributes,
        trace_destination=MlflowExperimentLocation(experiment_id) if experiment_id is not None else None,
    ) as span:
        span.set_inputs(inputs)
        try:
            yield span
        except Exception as exc:
            failure = exc
            span.set_outputs({"error": type(exc).__name__, "code": getattr(exc, "code", None)})
            span.set_status(SpanStatusCode.ERROR)
    if failure is not None:
        raise failure


class ConversationTracing:
    """Record each turn once, grouping turns without duplicating history in the root span."""

    def __init__(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id

    def record(self, request: ChatRequest, settings: AssistantSettings, run: Callable[[], dict]) -> dict:
        """Export turns through MLflow without a SQL read-back in the chat request path."""
        trace_id: str | None = None
        try:
            with trace_span(
                "autograph.chat",
                "AGENT",
                {"messages": [request.messages[-1].model_dump()]},
                {
                    "autograph.app_name": settings.app_name,
                    "autograph.run_id": settings.graph.run_id,
                    "autograph.graph_build_id": settings.graph.graph_build_id,
                    "autograph.embedding_snapshot_id": settings.snapshot.snapshot_id,
                    "autograph.model": settings.config.llm_endpoint,
                },
                experiment_id=self.experiment_id,
            ) as span:
                if span is None or not span.trace_id:
                    raise AssistantOperationError("tracing_unavailable", "MLflow tracing is not active")
                trace_id = span.trace_id
                mlflow.update_current_trace(
                    session_id=f"{settings.app_name}:{request.conversation_id}",
                    metadata={"autograph.conversation_id": request.conversation_id},
                    request_preview=request.messages[-1].content,
                )
                result = run()
                span.set_outputs({"messages": [{"role": "assistant", "content": result["answer"]}]})
                span.set_attributes(
                    {
                        "autograph.trace_id": result["trace_id"],
                        "autograph.tool_calls": result["tool_calls"],
                        "autograph.total_tokens": result["tokens"],
                    }
                )
                mlflow.update_current_trace(response_preview=result["answer"])
        except Exception as exc:
            if trace_id:
                logger.info("Assistant failed turn; MLflow trace {} conversation {}", trace_id, request.conversation_id)
                if isinstance(exc, AssistantOperationError):
                    exc.details["mlflow_trace_id"] = trace_id
            raise
        if trace_id is None:
            raise AssistantOperationError("tracing_unavailable", "MLflow tracing did not produce a trace ID")
        logger.info("Assistant turn; MLflow trace {} conversation {}", trace_id, request.conversation_id)
        return {**result, "mlflow_trace_id": trace_id, "conversation_id": request.conversation_id}


def configure_tracing(workspace: WorkspaceClient, config: ProjectConfig) -> ConversationTracing:
    """Initialize MLflow under the course writer, after capturing the App's own SDK client."""
    experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID", "").strip()
    if not experiment_id:
        raise RuntimeError("MLFLOW_EXPERIMENT_ID must reference the Bundle's assistant experiment")
    client_id, client_secret = read_writer_credentials(workspace, config)
    # MLflow initializes and refreshes credentials in its exporter threads. Keep these
    # process defaults stable; Lakebase and delegated operations use explicit clients.
    os.environ.update(
        DATABRICKS_HOST=workspace.config.host,
        DATABRICKS_AUTH_TYPE="oauth-m2m",
        DATABRICKS_CLIENT_ID=client_id,
        DATABRICKS_CLIENT_SECRET=client_secret,
    )
    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.set_experiment(experiment_id=experiment_id)
    if experiment.lifecycle_stage != "active":
        raise RuntimeError("The assistant MLflow experiment is not active")
    logger.info("Assistant conversation tracing enabled; experiment {}", experiment_id)
    return ConversationTracing(experiment_id)
