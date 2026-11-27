"""Real MLflow persistence tests with offline model and tool doubles."""

import json
import os
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import MagicMock, patch
from uuid import uuid4

import mlflow
import pytest
import yaml
from mlflow.tracking import MlflowClient
from pydantic import ValidationError
from starlette.testclient import TestClient
from test_assistant import _completion, _tool_call
from test_assistant import settings as settings
from test_embeddings import ROOT
from test_embeddings import prepared_input as prepared_input

from autographrs.assistant.agent import BibliographicAgent
from autographrs.assistant.models import AssistantOperationError, AssistantSettings, ChatRequest, RecoverableToolError
from autographrs.assistant.tools import BibliographicTools
from autographrs.assistant.tracing import ConversationTracing, configure_tracing
from autographrs.assistant.web import create_app, create_user_app


@pytest.fixture
def tracing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[ConversationTracing, None, None]:
    """Write and read actual traces without credentials or a Databricks connection."""
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    uri = f"sqlite:///{tmp_path / 'traces.db'}"
    mlflow.set_tracking_uri(uri)
    experiment_id = mlflow.create_experiment("assistant-tests", artifact_location=tmp_path.as_uri())
    try:
        yield ConversationTracing(experiment_id)
    finally:
        mlflow.flush_trace_async_logging()
        mlflow.set_tracking_uri(previous_uri)


def _agent(settings: AssistantSettings, tracing: ConversationTracing, responses: list) -> BibliographicAgent:
    tools = MagicMock(spec=BibliographicTools)
    tools.settings = settings
    tools.invoke.return_value = {"paper": {"title": "EEG paper", "node_id": "paper"}}
    client = MagicMock()
    client.__enter__.return_value = client
    client.with_options.return_value.chat.completions.create.side_effect = responses
    return BibliographicAgent(tools, lambda: client, tracing=tracing)


def test_conversation_turns_are_persisted_with_model_and_tool_children(
    settings: AssistantSettings, tracing: ConversationTracing
) -> None:
    """Each user turn survives independently of browser history and shares the conversation session."""
    conversation_id = str(uuid4())
    agent = _agent(
        settings,
        tracing,
        [_completion(tools=[_tool_call()]), _completion("First answer"), _completion("Second answer")],
    )
    first = agent.chat(
        ChatRequest(messages=[{"role": "user", "content": "Find EEG papers"}], conversation_id=conversation_id)
    )
    second = agent.chat(
        ChatRequest(
            messages=[
                {"role": "user", "content": "Find EEG papers"},
                {"role": "assistant", "content": first["answer"]},
                {"role": "user", "content": "Which are similar?"},
            ],
            conversation_id=conversation_id,
        )
    )
    assert first["mlflow_trace_id"] != second["mlflow_trace_id"]
    traces = [MlflowClient().get_trace(result["mlflow_trace_id"], display=False) for result in (first, second)]
    for trace, question, answer in zip(
        traces, ("Find EEG papers", "Which are similar?"), ("First answer", "Second answer"), strict=True
    ):
        root = next(span for span in trace.data.spans if span.parent_id is None)
        assert root.inputs == {"messages": [{"role": "user", "content": question}]}
        assert root.outputs == {"messages": [{"role": "assistant", "content": answer}]}
        assert root.attributes["autograph.graph_build_id"] == settings.graph.graph_build_id
        assert root.status.status_code.value == "OK"
        assert f"{settings.app_name}:{conversation_id}" in trace.info.trace_metadata.values()
        assert trace.info.trace_metadata["autograph.conversation_id"] == conversation_id
    root = next(span for span in traces[0].data.spans if span.parent_id is None)
    children = [span for span in traces[0].data.spans if span.parent_id == root.span_id]
    assert sorted(span.span_type for span in children) == ["LLM", "LLM", "TOOL"]
    assert next(span for span in children if span.span_type == "TOOL").outputs["paper"]["title"] == "EEG paper"
    assert all(
        span.attributes["mlflow.chat.tokenUsage"]["total_tokens"] == 20 for span in children if span.span_type == "LLM"
    )


def test_failed_turn_is_recorded_without_upstream_exception_text(
    settings: AssistantSettings, tracing: ConversationTracing
) -> None:
    """Record failed questions and diagnostic codes, never potentially credential-bearing upstream errors."""
    agent = _agent(settings, tracing, [AssistantOperationError("gateway_failure", "SECRET upstream token")])
    with pytest.raises(AssistantOperationError, match="SECRET") as failure:
        agent.chat(ChatRequest(messages=[{"role": "user", "content": "My failed question"}]))
    trace = MlflowClient().get_trace(failure.value.details["mlflow_trace_id"], display=False)
    assert all(span.status.status_code.value == "ERROR" for span in trace.data.spans)
    assert "SECRET" not in json.dumps(trace.to_dict())
    root = next(span for span in trace.data.spans if span.parent_id is None)
    assert root.inputs["messages"][0]["content"] == "My failed question"
    assert root.outputs["error"] == "AssistantOperationError"


def test_corrected_tool_error_has_failed_child_but_successful_answer(
    settings: AssistantSettings, tracing: ConversationTracing
) -> None:
    """Recoverable tool errors stay visible even when the agent completes the question."""
    agent = _agent(settings, tracing, [_completion(tools=[_tool_call()]), _completion("No matching paper")])
    agent.tools.invoke.side_effect = RecoverableToolError("node_not_found", "SECRET upstream data")
    result = agent.chat(ChatRequest(messages=[{"role": "user", "content": "Find paper"}]))
    trace = MlflowClient().get_trace(result["mlflow_trace_id"], display=False)
    tool = next(span for span in trace.data.spans if span.span_type == "TOOL")
    root = next(span for span in trace.data.spans if span.parent_id is None)
    assert tool.status.status_code.value == "ERROR" and tool.outputs["code"] == "node_not_found"
    assert root.status.status_code.value == "OK"
    assert "SECRET" not in json.dumps(trace.to_dict())


def test_chat_does_not_require_trace_read_access_or_retry_inference(
    settings: AssistantSettings, tracing: ConversationTracing
) -> None:
    """Observability does not add SQL queries or inference retries to the request path."""
    agent = _agent(settings, tracing, [_completion("Answer")])
    with (
        patch.object(MlflowClient, "get_trace", side_effect=RuntimeError("Warehouse unavailable")) as read,
        patch("autographrs.assistant.tracing.mlflow.flush_trace_async_logging") as flush,
    ):
        result = agent.chat(ChatRequest(messages=[{"role": "user", "content": "Question"}]))
    assert result["answer"] == "Answer" and result["mlflow_trace_id"]
    read.assert_not_called()
    flush.assert_not_called()
    assert agent.client_factory().with_options.return_value.chat.completions.create.call_count == 1


def test_exporter_outage_is_logged_without_blocking_answer(
    settings: AssistantSettings, tracing: ConversationTracing, caplog: pytest.LogCaptureFixture
) -> None:
    """The pinned MLflow exporter follows the selected best-effort policy, with explicit error logs."""
    agent = _agent(settings, tracing, [_completion("Answer during outage")])
    with patch(
        "mlflow.tracing.export.mlflow_v3.MlflowV3SpanExporter.export",
        side_effect=RuntimeError("simulated exporter outage"),
    ):
        result = agent.chat(ChatRequest(messages=[{"role": "user", "content": "Question"}]))
    assert result["answer"] == "Answer during outage"
    assert "simulated exporter outage" in caplog.text
    assert agent.client_factory().with_options.return_value.chat.completions.create.call_count == 1


def test_new_conversations_have_distinct_sessions(settings: AssistantSettings, tracing: ConversationTracing) -> None:
    """Older API clients without a conversation ID receive one instead of being grouped with other callers."""
    agent = _agent(settings, tracing, [_completion("One"), _completion("Two")])
    results = [agent.chat(ChatRequest(messages=[{"role": "user", "content": "Question"}])) for _ in range(2)]
    assert results[0]["conversation_id"] != results[1]["conversation_id"]
    for invalid in ("", "user@example.com", "../another-session", "x" * 1000):
        with pytest.raises(ValidationError):
            ChatRequest(messages=[{"role": "user", "content": "Question"}], conversation_id=invalid)


def test_concurrent_turns_do_not_mix_sessions(settings: AssistantSettings, tracing: ConversationTracing) -> None:
    """Worker threads must have separate active roots even while both model calls are in flight."""
    barrier = Barrier(2)

    def run(question: str) -> dict:
        agent = _agent(settings, tracing, [])

        def complete(**_kwargs: object) -> object:
            barrier.wait(timeout=10)
            return _completion(f"Answer to {question}")

        agent.client_factory().with_options.return_value.chat.completions.create.side_effect = complete
        return agent.chat(ChatRequest(messages=[{"role": "user", "content": question}]))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ("One", "Two")))
    for question, result in zip(("One", "Two"), results, strict=True):
        trace = MlflowClient().get_trace(result["mlflow_trace_id"], display=False)
        assert len(trace.data.spans) == 2
        root = next(span for span in trace.data.spans if span.parent_id is None)
        assert root.inputs["messages"][0]["content"] == question
        assert root.outputs["messages"][0]["content"] == f"Answer to {question}"
        assert trace.info.trace_metadata["autograph.conversation_id"] == result["conversation_id"]


def test_chat_http_transports_conversation_id_but_never_oauth_headers(
    settings: AssistantSettings, tracing: ConversationTracing
) -> None:
    """The API passes the browser UUID through to the worker-thread trace without recording headers."""
    agent = _agent(settings, tracing, [_completion("Answer")])
    conversation_id = str(uuid4())
    with TestClient(create_app(settings, agent.tools, agent), base_url=settings.app_url) as client:
        response = client.post(
            "/api/chat",
            headers={"x-forwarded-user": "owner-id", "x-forwarded-access-token": "SECRET OAuth token"},
            json={"messages": [{"role": "user", "content": "Question"}], "conversation_id": conversation_id},
        )
    assert response.status_code == 200 and response.json()["conversation_id"] == conversation_id
    trace = MlflowClient().get_trace(response.json()["mlflow_trace_id"], display=False)
    assert "SECRET" not in json.dumps(trace.to_dict())


def test_user_app_injects_tracing_and_flushes_outside_delegated_context(settings: AssistantSettings) -> None:
    """App lifetime owns tracing; its shutdown must not retain an end user's OAuth client."""
    from autographrs.assistant.auth import USER_WORKSPACE

    settings = settings.model_copy(update={"sql_auth": "user"})
    app_workspace = MagicMock()
    app_workspace.config.host = "https://workspace.example"
    tracing = ConversationTracing("123")
    with (
        patch("autographrs.assistant.web.WorkspaceClient"),
        patch("autographrs.assistant.web.Config"),
        patch("autographrs.assistant.web.load_settings", return_value=settings),
        patch("autographrs.assistant.web.BibliographicAgent") as agents,
        patch("autographrs.assistant.web.mlflow.flush_trace_async_logging") as flush,
    ):
        flush.side_effect = lambda: pytest.fail("OAuth context leaked") if USER_WORKSPACE.get(None) else None
        with TestClient(create_user_app(app_workspace, tracing=tracing), base_url=settings.app_url) as client:
            assert (
                client.get(
                    "/api/collection",
                    headers={"x-forwarded-user": "owner-id", "x-forwarded-access-token": "SECRET"},
                ).status_code
                == 200
            )
            assert agents.call_args.kwargs["tracing"] is tracing
        flush.assert_called_once_with()


def test_configure_tracing_initializes_course_writer_before_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use App credentials only to fetch secrets and initialize MLflow under the course writer."""
    app, config = MagicMock(), MagicMock()
    app.config.host = "https://workspace.example"
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_EXPERIMENT_ID"):
        configure_tracing(app, config)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "123")
    for key in ("HOST", "AUTH_TYPE", "CLIENT_ID", "CLIENT_SECRET"):
        monkeypatch.setenv(f"DATABRICKS_{key}", "app-default")
    with (
        patch(
            "autographrs.assistant.tracing.read_writer_credentials",
            return_value=("course-writer", "synthetic-secret"),
        ) as credentials,
        patch("autographrs.assistant.tracing.mlflow.set_tracking_uri") as tracking,
        patch("autographrs.assistant.tracing.mlflow.set_experiment") as experiment,
    ):

        def verify_writer(uri: str) -> None:
            assert uri == "databricks"
            assert os.environ["DATABRICKS_HOST"] == app.config.host
            assert os.environ["DATABRICKS_AUTH_TYPE"] == "oauth-m2m"
            assert os.environ["DATABRICKS_CLIENT_ID"] == "course-writer"
            assert os.environ["DATABRICKS_CLIENT_SECRET"] == "synthetic-secret"

        tracking.side_effect = verify_writer
        experiment.return_value.lifecycle_stage = "active"
        assert configure_tracing(app, config).experiment_id == "123"
        credentials.assert_called_once_with(app, config)
        tracking.assert_called_once_with("databricks")
        experiment.assert_called_once_with(experiment_id="123")
        experiment.return_value.lifecycle_stage = "deleted"
        with pytest.raises(RuntimeError, match="not active"):
            configure_tracing(app, config)


def test_bundle_references_course_experiments_and_browser_rotates_session() -> None:
    """The bundle refers to existing course resources without granting the App direct trace access."""
    from autographrs.assistant import ui

    bundle = yaml.safe_load((ROOT / "databricks.yml").read_text())
    resources = yaml.safe_load((ROOT / "resources" / "bibliographic_assistant.yml").read_text())["resources"]
    app = resources["apps"]["bibliographic_assistant"]
    assert "experiments" not in resources
    assert {target["variables"]["assistant_experiment_id"] for target in bundle["targets"].values()} == {
        "244896477021790",
        "244896477021794",
        "244896477021798",
    }
    assert {"name": "MLFLOW_EXPERIMENT_ID", "value": "${var.assistant_experiment_id}"} in app["config"]["env"]
    assert not app.get("resources")
    assert bundle["targets"]["dev"]["resources"]["jobs"]["setup_assistant"]["run_as"] == {
        "user_name": "${workspace.current_user.userName}"
    }
    assert "conversation_id: conversationId" in ui.JS and ui.JS.count("crypto.randomUUID()") == 2
    assert "Le conversazioni sono registrate in MLflow" in ui.HTML
