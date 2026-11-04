"""Delegated SQL/Genie/Gateway requests remain isolated from App-authorized Lakebase operations."""

import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import BadRequestError
from starlette.testclient import TestClient
from test_assistant import settings as settings
from test_embeddings import prepared_input as prepared_input

from autographrs.assistant.auth import USER_WORKSPACE, user_workspace
from autographrs.assistant.genie import GenieContractError
from autographrs.assistant.models import AnalyticsArguments, AssistantOperationError, AssistantSettings
from autographrs.assistant.tools import BibliographicTools
from autographrs.assistant.web import create_app, create_user_app, grouped_error_details


def test_missing_delegated_context_never_falls_back_to_app_identity() -> None:
    """Missing OAuth must be explicit, not privilege-dependent fallback behavior."""
    with pytest.raises(PermissionError, match="User OAuth"):
        user_workspace()


@pytest.mark.parametrize("grouped", [False, True])
def test_operational_failure_codes_are_visible_without_provider_messages(
    settings: AssistantSettings, grouped: bool
) -> None:
    """Budget and provider failures stay distinct, including inside MCP task groups."""
    failure = AssistantOperationError(
        "genie_query_failed", "private upstream response", tool="ask_collection_analytics"
    )
    agent = MagicMock()
    agent.chat.side_effect = ExceptionGroup("private", [failure]) if grouped else failure
    with (
        patch("autographrs.assistant.web.logger") as logger,
        TestClient(
            create_app(settings, MagicMock(), agent), base_url=settings.app_url, raise_server_exceptions=False
        ) as client,
    ):
        response = client.post(
            "/api/chat",
            headers={"x-forwarded-user": "owner-id"},
            json={"messages": [{"role": "user", "content": "Rank EEG papers"}]},
        )
        assert response.status_code == 502
        assert "genie_query_failed" in response.text
        assert "ask_collection_analytics" in response.text
        assert "private" not in response.text
        assert "private" not in str(logger.error.call_args_list)


def test_invalid_model_arguments_are_not_reported_as_invalid_user_question(settings: AssistantSettings) -> None:
    """Keep strict tool validation while identifying model errors without exposing argument values."""
    tools = BibliographicTools(settings, MagicMock())
    agent = MagicMock()
    agent.chat.side_effect = lambda _: tools.invoke(
        "search_collection", {"query": "private EEG question", "limit": 100, "private-key": "private token"}
    )
    with TestClient(
        create_app(settings, tools, agent), base_url=settings.app_url, raise_server_exceptions=False
    ) as client:
        result = client.post(
            "/api/chat",
            headers={"x-forwarded-user": "owner-id"},
            json={"messages": [{"role": "user", "content": "dammi il paper che parla di eeg con più citazioni"}]},
        )
        assert result.status_code == 502
        assert result.json()["diagnostics"] == {
            "tool": "search_collection",
            "issues": [
                {"field": "limit", "type": "less_than_equal"},
                {"field": "unknown", "type": "extra_forbidden"},
            ],
        }
        assert "private" not in result.text
        result = client.post(
            "/api/chat",
            headers={"x-forwarded-user": "owner-id"},
            json={"messages": [{"role": "system", "content": "bad"}]},
        )
        assert result.status_code == 400
        assert "Invalid chat request" in result.json()["error"]


def test_nested_group_failure_exposes_leaf_status_without_private_messages(settings: AssistantSettings) -> None:
    """Task-group wrappers must not conceal transport status or leak nested request data."""
    request = httpx.Request(
        "POST",
        "https://workspace.example/api/2.0/mcp/genie/private-space",
        headers={"Authorization": "Bearer private-token"},
    )
    response = httpx.Response(403, request=request, text="private response")
    error = httpx.HTTPStatusError("private transport message", request=request, response=response)
    agent = MagicMock()
    agent.chat.side_effect = ExceptionGroup(
        "private group", [ExceptionGroup("private nested group", [error]), RuntimeError("private question")]
    )
    with (
        patch("autographrs.assistant.web.logger") as logger,
        TestClient(
            create_app(settings, MagicMock(), agent), base_url=settings.app_url, raise_server_exceptions=False
        ) as client,
    ):
        result = client.post(
            "/api/chat",
            headers={"x-forwarded-user": "owner-id"},
            json={"messages": [{"role": "user", "content": "private question"}]},
        )
        assert result.status_code == 502
        assert result.json()["diagnostics"] == {
            "errors": [
                {"type": "HTTPStatusError", "http_status": 403, "operation": "genie_mcp"},
                {"type": "RuntimeError"},
            ],
            "truncated": False,
        }
        assert "Group diagnostics" in result.json()["error"]
        assert "private" not in result.text
        assert "private" not in str(logger.error.call_args_list)


def test_group_diagnostics_bound_output_and_preserve_nested_gateway_metadata() -> None:
    """Large or deep groups remain bounded; wrapped Gateway errors retain existing sanitization."""
    request = httpx.Request("POST", "https://workspace.example/ai-gateway/mlflow/v1/chat/completions")
    response = httpx.Response(400, request=request)
    error = BadRequestError("private message", response=response, body={"param": "tools"})
    details = grouped_error_details(ExceptionGroup("private", [error]))
    assert details["errors"][0]["gateway"] == {"http_status": 400, "operation": "chat", "param": "tools"}
    details = grouped_error_details(ExceptionGroup("private", [ValueError("private")] * 80))
    assert len(details["errors"]) == 16 and details["truncated"]
    group = ExceptionGroup("private", [ValueError("private")])
    for _ in range(80):
        group = ExceptionGroup("private", [group])
    assert grouped_error_details(group) == {"errors": [], "truncated": True}
    error = GenieContractError("unsupported_question_schema", [])
    details = grouped_error_details(ExceptionGroup("private", [error]))
    assert details["errors"][0]["genie"] == {
        "operation": "genie_mcp",
        "reason": "unsupported_question_schema",
        "tool_count": 0,
        "schemas": [],
    }


@pytest.mark.parametrize("operation,path", [("chat", "chat/completions"), ("embedding", "embeddings")])
def test_gateway_failure_exposes_protocol_metadata_not_sensitive_body_or_headers(
    settings: AssistantSettings, operation: str, path: str
) -> None:
    """Keep useful Gateway failure metadata visible to the owner without logging conversations."""
    body = {
        "message": "parallel_tool_calls unsupported; private question and private token",
        "code": "unsupported_parameter",
        "param": "parallel_tool_calls",
        "type": "invalid_request_error",
        "private": "private provider data",
    }
    request = httpx.Request(
        "POST",
        f"https://workspace.example/ai-gateway/mlflow/v1/{path}",
        headers={"Authorization": "Bearer private token"},
    )
    response = httpx.Response(400, request=request, headers={"x-request-id": "provider-id"})
    agent = MagicMock()
    agent.chat.side_effect = BadRequestError("private exception message", response=response, body=body)
    with (
        patch("autographrs.assistant.web.logger") as logger,
        TestClient(
            create_app(settings, MagicMock(), agent), base_url=settings.app_url, raise_server_exceptions=False
        ) as client,
    ):
        result = client.post(
            "/api/chat",
            headers={"x-forwarded-user": "owner-id"},
            json={"messages": [{"role": "user", "content": "private question"}]},
        )
        assert result.status_code == 502
        details = result.json()["gateway"]
        assert details == {
            "http_status": 400,
            "operation": operation,
            "code": "unsupported_parameter",
            "param": "parallel_tool_calls",
            "type": "invalid_request_error",
            "request_id": "provider-id",
            "mentioned_fields": ["parallel_tool_calls"],
        }
        assert "Gateway diagnostics" in result.json()["error"]
        assert "private" not in result.text
        assert "private" not in str(logger.error.call_args_list)


def test_sql_and_genie_use_user_identity_but_gateway_and_lakebase_keep_app_identity(
    settings: AssistantSettings,
) -> None:
    """Changing SQL identity does not silently change other service credentials."""
    app, user = MagicMock(), MagicMock()
    tools = BibliographicTools(settings, app, sql_workspace=user_workspace)
    context = USER_WORKSPACE.set(user)
    try:
        assert tools.workspace is app
        assert tools.reader.workspace_provider() is user
        with patch("autographrs.assistant.genie.ask_genie", return_value={"answer": "one"}) as genie:
            tools.analytics(AnalyticsArguments(question="How many?"))
            assert genie.call_args.args[0] is user
    finally:
        USER_WORKSPACE.reset(context)
    with pytest.raises(PermissionError):
        tools.reader.workspace_provider()


def test_user_clients_are_task_local_and_propagate_to_tool_worker_threads() -> None:
    """Overlapping requests cannot reuse one another's delegated credentials."""

    async def request(client: MagicMock) -> None:
        context = USER_WORKSPACE.set(client)
        try:
            await asyncio.sleep(0)
            assert user_workspace() is client
            assert await asyncio.to_thread(user_workspace) is client
        finally:
            USER_WORKSPACE.reset(context)

    async def run() -> None:
        await asyncio.gather(request(MagicMock()), request(MagicMock()))

    asyncio.run(run())
    with pytest.raises(PermissionError):
        user_workspace()


def test_search_keeps_app_identity_even_inside_a_delegated_sql_request(settings: AssistantSettings) -> None:
    """Embedding and pgvector do not receive the user's SQL token."""
    app, user = MagicMock(), MagicMock()
    tools = BibliographicTools(settings, app, sql_workspace=user_workspace)
    context = USER_WORKSPACE.set(user)
    try:
        with (
            patch("autographrs.enrichment.llm.create_gateway_openai_client") as gateway,
            patch("autographrs.assistant.tools.GatewayEmbedder") as embedder,
            patch("autographrs.assistant.tools.LakebasePgVectorManager") as lakebase,
        ):
            embedder.return_value.embed.return_value = ([[0.1, 0.2, 1.0]], settings.response_model)
            lakebase.return_value.search_published.return_value = []
            assert tools.invoke("search_collection", {"query": "memory"})["papers"] == []
            gateway.assert_called_once_with(app)
            assert lakebase.call_args.args[1] is app
    finally:
        USER_WORKSPACE.reset(context)


def test_search_delegates_gateway_but_keeps_lakebase_on_app_identity(settings: AssistantSettings) -> None:
    """Explicit Gateway delegation never leaks the user credential into PostgreSQL or falls back."""
    app, first, second = MagicMock(), MagicMock(), MagicMock()
    tools = BibliographicTools(settings, app, sql_workspace=user_workspace, gateway_workspace=user_workspace)
    with (
        patch("autographrs.enrichment.llm.create_gateway_openai_client") as gateway,
        patch("autographrs.assistant.tools.GatewayEmbedder") as embedder,
        patch("autographrs.assistant.tools.LakebasePgVectorManager") as lakebase,
    ):
        embedder.return_value.embed.return_value = ([[0.1, 0.2, 1.0]], settings.response_model)
        lakebase.return_value.search_published.return_value = []
        for user in (first, second):
            context = USER_WORKSPACE.set(user)
            try:
                tools.invoke("search_collection", {"query": "memory"})
                assert gateway.call_args.args[0] is user
                assert lakebase.call_args.args[1] is app
            finally:
                USER_WORKSPACE.reset(context)
        with pytest.raises(PermissionError, match="User OAuth"):
            tools.invoke("search_collection", {"query": "memory"})
        assert gateway.call_count == lakebase.call_count == 2
        context = USER_WORKSPACE.set(first)
        try:
            gateway.side_effect = PermissionError("Gateway scope denied")
            with pytest.raises(PermissionError, match="Gateway scope denied"):
                tools.invoke("search_collection", {"query": "memory"})
            assert gateway.call_count == 3
            assert lakebase.call_count == 2
        finally:
            USER_WORKSPACE.reset(context)


def test_chat_gateway_uses_current_request_identity_not_first_user(settings: AssistantSettings) -> None:
    """The cached agent resolves the delegated client for each chat in the worker thread."""
    settings = settings.model_copy(update={"sql_auth": "user"})
    app, first, second = MagicMock(), MagicMock(), MagicMock()
    app.config.host = "https://workspace.example"
    with (
        patch("autographrs.assistant.web.WorkspaceClient", side_effect=[first, second]),
        patch("autographrs.assistant.web.Config"),
        patch("autographrs.assistant.web.load_settings", return_value=settings),
        patch("autographrs.assistant.web.BibliographicAgent") as agents,
        patch("autographrs.assistant.web.create_gateway_openai_client") as gateway,
        TestClient(create_user_app(app), base_url=settings.app_url) as client,
    ):

        def chat(_request: object) -> dict:
            agents.call_args.args[1]()
            return {"answer": "one"}

        agents.return_value.chat.side_effect = chat
        for token in ("first", "second"):
            response = client.post(
                "/api/chat",
                headers={"x-forwarded-user": "owner-id", "x-forwarded-access-token": token},
                json={"messages": [{"role": "user", "content": "q"}]},
            )
            assert response.status_code == 200
        assert [call.args[0] for call in gateway.call_args_list] == [first, second]
        tools = agents.call_args.args[0]
        assert tools.workspace is app
        assert tools.gateway_workspace is user_workspace
    with pytest.raises(PermissionError):
        agents.call_args.args[1]()
    assert gateway.call_count == 2


def test_app_startup_does_not_read_sql_and_each_request_uses_its_own_token(settings: AssistantSettings) -> None:
    """No user token is cached in the App, and no token means no SQL or fallback."""
    settings = settings.model_copy(update={"sql_auth": "user"})
    app_workspace = MagicMock()
    app_workspace.config.host = "https://workspace.example"
    user_one, user_two = MagicMock(), MagicMock()
    with (
        patch("autographrs.assistant.web.WorkspaceClient", side_effect=[user_one, user_two]) as clients,
        patch("autographrs.assistant.web.Config") as config,
        patch("autographrs.assistant.web.load_settings", return_value=settings) as load,
        TestClient(create_user_app(app_workspace), base_url=settings.app_url) as client,
    ):
        load.assert_not_called()
        response = client.get("/api/collection")
        assert response.status_code == 401
        clients.assert_not_called()
        headers = {"x-forwarded-user": "owner-id", "x-forwarded-access-token": "first-example-token"}
        assert client.get("/api/collection", headers=headers).status_code == 200
        load.assert_called_once_with(user_one)
        headers["x-forwarded-access-token"] = "second-example-token"
        assert client.get("/api/collection", headers=headers).status_code == 200
        assert [call.kwargs["token"] for call in config.call_args_list] == [
            "first-example-token",
            "second-example-token",
        ]
        assert client.get("/api/collection", headers={"x-forwarded-user": "owner-id"}).status_code == 401
        headers["x-forwarded-user"] = "other-user"
        clients.side_effect = None
        assert client.get("/api/collection", headers=headers).status_code == 403
        load.assert_called_once()
    with pytest.raises(PermissionError):
        user_workspace()


def test_chat_and_mcp_workers_receive_delegated_sql_context(settings: AssistantSettings) -> None:
    """Both transports see request-local OAuth, without persisting it in shared tools."""
    settings = settings.model_copy(update={"sql_auth": "user"})
    app, user = MagicMock(), MagicMock()
    app.config.host = "https://workspace.example"
    observed = []

    def chat(_request: object) -> dict:
        observed.append(user_workspace())
        return {"answer": "one", "evidence": [], "tool_calls": 0, "tokens": 1}

    def invoke(_name: str, _arguments: dict) -> dict:
        observed.append(user_workspace())
        return {"paper": {"title": "one"}}

    with (
        patch("autographrs.assistant.web.WorkspaceClient", return_value=user),
        patch("autographrs.assistant.web.Config"),
        patch("autographrs.assistant.web.load_settings", return_value=settings),
        patch("autographrs.assistant.web.BibliographicAgent") as agents,
        patch("autographrs.assistant.web.BibliographicTools") as tools,
        TestClient(create_user_app(app), base_url=settings.app_url) as client,
    ):
        agents.return_value.chat.side_effect = chat
        tools.return_value.invoke.side_effect = invoke
        headers = {
            "x-forwarded-user": "owner-id",
            "x-forwarded-access-token": "example",
            "Accept": "application/json, text/event-stream",
        }
        response = client.post("/api/chat", headers=headers, json={"messages": [{"role": "user", "content": "q"}]})
        assert response.status_code == 200
        response = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "get_paper", "arguments": {"node_id": "paper"}},
            },
        )
        assert response.status_code == 200
        assert not response.json()["result"].get("isError")
        assert observed == [user, user]
        assert tools.call_args.args[1] is app
    with pytest.raises(PermissionError):
        user_workspace()


def test_user_manifest_errors_fail_explicitly_without_caching_bad_initialization(settings: AssistantSettings) -> None:
    """Denied SQL and absent setup do not masquerade as an empty collection or trigger App SQL fallback."""
    app_workspace = MagicMock()
    app_workspace.config.host = "https://workspace.example"
    with (
        patch("autographrs.assistant.web.WorkspaceClient"),
        patch("autographrs.assistant.web.Config"),
        patch("autographrs.assistant.web.load_settings", side_effect=PermissionError("private details")) as load,
        TestClient(create_user_app(app_workspace), base_url=settings.app_url) as client,
    ):
        headers = {"x-forwarded-user": "owner-id", "x-forwarded-access-token": "example"}
        response = client.get("/api/collection", headers=headers)
        assert response.status_code == 502 and "private details" not in response.text
        client.get("/api/collection", headers=headers)
        assert load.call_count == 2


@pytest.mark.parametrize("identity_header", ["x-forwarded-email", "x-forwarded-preferred-username"])
def test_proxy_owner_identity_can_differ_from_workspace_id(settings: AssistantSettings, identity_header: str) -> None:
    """IdP subject IDs need not equal SCIM IDs; all requests still require the configured owner."""
    settings = settings.model_copy(update={"sql_auth": "user", "allowed_subjects": ["owner-id", "owner@example.test"]})
    workspace = MagicMock()
    workspace.config.host = "https://workspace.example"
    with (
        patch("autographrs.assistant.web.WorkspaceClient"),
        patch("autographrs.assistant.web.Config"),
        patch("autographrs.assistant.web.load_settings", return_value=settings),
        TestClient(create_user_app(workspace), base_url=settings.app_url) as client,
    ):
        headers = {
            "x-forwarded-user": "idp-subject-different-from-scim",
            "x-forwarded-access-token": "example",
            identity_header: "other@example.test",
        }
        assert client.get("/api/collection", headers=headers).status_code == 403
        headers[identity_header] = "owner@example.test"
        assert client.get("/api/collection", headers=headers).status_code == 200
        assert client.get("/", headers=headers).status_code == 200
        headers[identity_header] = "other@example.test"
        assert client.get("/api/collection", headers=headers).status_code == 403
        headers[identity_header] = "owner@example.test"
        del headers["x-forwarded-user"]
        assert client.get("/api/collection", headers=headers).status_code == 401
