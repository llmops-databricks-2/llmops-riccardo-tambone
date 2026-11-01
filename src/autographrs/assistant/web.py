"""Private Databricks App: stateless chat API and MCP backed by the same collection-scoped tools."""

import asyncio
import json
import os
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import Context
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from databricks.sdk import WorkspaceClient
from databricks.sdk.config import Config
from loguru import logger
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from openai import APIStatusError
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from autographrs.assistant import ui
from autographrs.assistant.agent import BibliographicAgent
from autographrs.assistant.auth import USER_WORKSPACE, is_allowed_owner, user_workspace
from autographrs.assistant.genie import GenieContractError
from autographrs.assistant.models import (
    AssistantOperationError,
    AssistantSettings,
    ChatRequest,
    RecoverableToolError,
    ToolArgumentsError,
)
from autographrs.assistant.sql import SqlReader, qualified
from autographrs.assistant.tools import BibliographicTools
from autographrs.config import load_config
from autographrs.enrichment.llm import create_gateway_openai_client


def gateway_error_details(exc: APIStatusError) -> dict:
    """Expose bounded protocol metadata, never raw provider messages, bodies or request headers."""
    details: dict = {"http_status": exc.status_code}
    path = exc.request.url.path.rstrip("/")
    if path.endswith("/chat/completions"):
        details["operation"] = "chat"
    elif path.endswith("/embeddings"):
        details["operation"] = "embedding"
    for key, value in (("code", exc.code), ("type", exc.type), ("param", exc.param), ("request_id", exc.request_id)):
        if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_.:\[\]-]{1,128}", value):
            details[key] = value
    body = exc.body
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            message = error["message"]
            fields = (
                "model",
                "messages",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                "max_tokens",
                "max_completion_tokens",
                "input",
                "additionalProperties",
                "temperature",
                "stream",
            )
            mentioned = [field for field in fields if re.search(rf"\b{re.escape(field)}\b", message)]
            if mentioned:
                details["mentioned_fields"] = mentioned
    return details


def grouped_error_details(exc: BaseExceptionGroup) -> dict:
    """Inspect bounded group leaves without serializing exception text or transport credentials."""
    pending: list[BaseException] = [exc]
    errors: list[dict] = []
    inspected = 0
    truncated = False
    while pending and inspected < 64 and len(errors) < 16:
        current = pending.pop()
        inspected += 1
        if isinstance(current, BaseExceptionGroup):
            truncated = truncated or len(current.exceptions) > 64
            pending.extend(reversed(current.exceptions[:64]))
            continue
        detail: dict = {"type": type(current).__name__}
        if isinstance(current, GenieContractError):
            detail["genie"] = current.diagnostics
        if isinstance(current, AssistantOperationError):
            detail["code"] = current.code
            if current.tool:
                detail["tool"] = current.tool
        traceback = current.__traceback__
        while traceback is not None:
            module = traceback.tb_frame.f_globals.get("__name__", "")
            if isinstance(module, str) and module.startswith("autographrs."):
                detail["origin"] = module
            traceback = traceback.tb_next
        if isinstance(current, APIStatusError):
            detail["gateway"] = gateway_error_details(current)
        elif isinstance(current, httpx.HTTPStatusError):
            detail["http_status"] = current.response.status_code
            if "/api/2.0/mcp/genie/" in current.request.url.path:
                detail["operation"] = "genie_mcp"
        errors.append(detail)
    return {"errors": errors, "truncated": truncated or bool(pending)}


class PrivateAccess:
    """Enforce proxy identity, same-origin browser access and bounded bodies before either transport parses input."""

    def __init__(self, app: ASGIApp, settings: AssistantSettings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Fail closed for HTTP; Databricks Apps is the required authenticated reverse proxy."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        if not is_allowed_owner(request.headers, self.settings.allowed_subjects):
            await JSONResponse({"error": "Access denied"}, status_code=403)(scope, receive, send)
            return
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != self.settings.app_url.rstrip("/"):
            await JSONResponse({"error": "Origin denied"}, status_code=403)(scope, receive, send)
            return
        body = bytearray()
        if scope["method"] in ("POST", "PUT", "PATCH"):
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > 64000:
                    await JSONResponse({"error": "Request exceeds 64000 bytes"}, status_code=413)(scope, receive, send)
                    return
                if not message.get("more_body", False):
                    break
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed and scope["method"] in ("POST", "PUT", "PATCH"):
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        async def secured_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = list(message.get("headers", [])) + [
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                    (
                        b"content-security-policy",
                        b"default-src 'self'; script-src 'self'; style-src 'self'; "
                        b"connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
                    ),
                ]
            await send(message)

        await self.app(scope, replay, secured_send)


def load_settings(workspace: WorkspaceClient) -> AssistantSettings:
    """Fail closed until the explicit bootstrap job has approved a collection for this App."""
    config = load_config("project_config.yml", env=os.environ["AUTOGRAPHRS_ENV"])
    app_name = os.environ["AUTOGRAPHRS_APP_NAME"]
    rows = SqlReader(config, workspace).read(
        f"SELECT payload FROM {qualified(config, 'assistant_settings')} "
        "WHERE run_id = :run AND app_name = :app LIMIT 2",
        {"run": "assistant-settings-v1", "app": app_name},
        limit=2,
    )
    if len(rows) != 1:
        raise ValueError("Run setup_assistant with an explicit published collection before starting this App")
    settings = AssistantSettings.model_validate(rows[0])
    if settings.app_name != app_name or settings.config.full_schema_name != config.full_schema_name:
        raise ValueError("Assistant settings do not belong to this App/environment")
    return settings


def create_app(settings: AssistantSettings, tools: BibliographicTools, agent: BibliographicAgent) -> Starlette:
    """Compose transports; Databricks authenticates the edge, then the App enforces its owner allowlist."""
    host = urlparse(settings.app_url).netloc
    if not host or not settings.app_url.startswith("https://"):
        raise ValueError("A valid Databricks App HTTPS URL is required")
    mcp = FastMCP(
        "AutoGraphRS",
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(
            allowed_hosts=[host],
            allowed_origins=[settings.app_url.rstrip("/")],
        ),
    )
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    def invoke(name: str, arguments: dict) -> dict:
        try:
            return tools.invoke(name, arguments)
        except Exception as exc:
            reference = uuid4().hex
            logger.error("MCP tool {} failed: {} [{}]", name, type(exc).__name__, reference)
            diagnostic = ""
            if isinstance(exc, AssistantOperationError):
                diagnostic = f"; code {exc.code}"
                if isinstance(exc, RecoverableToolError) and exc.hint:
                    diagnostic += f"; {exc.hint}"
                logger.error("MCP application failure: {} [{}]", exc.code, reference)
            elif isinstance(exc, BaseExceptionGroup):
                diagnostic = f"; Group diagnostics: {json.dumps(grouped_error_details(exc))}"
                logger.error("MCP grouped failure: {} [{}]", diagnostic, reference)
            raise ToolError(f"Tool failed ({type(exc).__name__}); reference {reference}{diagnostic}") from None

    @mcp.tool(annotations=annotations)
    def search_collection(query: str, limit: int = 5) -> dict:
        """Find semantically similar papers in this approved collection."""
        return invoke("search_collection", {"query": query, "limit": limit})

    @mcp.tool(annotations=annotations)
    def get_paper(node_id: str) -> dict:
        """Read exact paper metadata and provenance."""
        return invoke("get_paper", {"node_id": node_id})

    @mcp.tool(annotations=annotations)
    def get_bibliography(node_id: str, limit: int = 10, offset: int = 0) -> dict:
        """Read acquired bibliography evidence, not generated references."""
        return invoke("get_bibliography", {"node_id": node_id, "limit": limit, "offset": offset})

    @mcp.tool(annotations=annotations)
    def get_neighbors(node_id: str, direction: str = "outgoing", depth: int = 1, limit: int = 100) -> dict:
        """Traverse published CITES evidence within bounded depth and result count."""
        return invoke("get_neighbors", {"node_id": node_id, "direction": direction, "depth": depth, "limit": limit})

    @mcp.tool(annotations=annotations)
    def ask_collection_analytics(question: str) -> dict:
        """Ask the private Genie Agent an analytical question about this collection."""
        return invoke("ask_collection_analytics", {"question": question})

    slots = asyncio.Semaphore(2)

    async def collection(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "run_id": settings.graph.run_id,
                "graph_build_id": settings.graph.graph_build_id,
                "embedding_snapshot_id": settings.snapshot.snapshot_id,
                "node_count": settings.graph.node_count,
                "edge_count": settings.graph.edge_count,
            }
        )

    async def chat(request: Request) -> JSONResponse:
        try:
            parsed = ChatRequest.model_validate_json(await request.body())
        except ValidationError:
            logger.warning("Rejected invalid chat request")
            return JSONResponse(
                {"error": "Invalid chat request; start a new conversation and check message limits"},
                status_code=400,
            )
        if slots.locked():
            return JSONResponse({"error": "Assistant busy; retry shortly"}, status_code=429)
        async with slots:
            return JSONResponse(await run_in_threadpool(agent.chat, parsed))

    async def index(_request: Request) -> HTMLResponse:
        return HTMLResponse(ui.HTML)

    async def script(_request: Request) -> PlainTextResponse:
        return PlainTextResponse(ui.JS, media_type="application/javascript")

    async def style(_request: Request) -> PlainTextResponse:
        return PlainTextResponse(ui.CSS, media_type="text/css")

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncGenerator[None]:
        async with mcp.session_manager.run():
            yield

    async def invalid_request(_request: Request, _exc: Exception) -> JSONResponse:
        reference = uuid4().hex
        logger.error("Assistant data validation failed [{}]", reference)
        return JSONResponse({"error": f"Returned data failed validation; reference {reference}"}, status_code=502)

    async def operation_failed(_request: Request, exc: Exception) -> JSONResponse:
        reference = uuid4().hex
        logger.error("Assistant request failed: {} [{}]", type(exc).__name__, reference)
        payload: dict = {"error": f"Operation failed ({type(exc).__name__}); reference {reference}"}
        if isinstance(exc, APIStatusError):
            details = gateway_error_details(exc)
            logger.error("Gateway failure metadata: {} [{}]", json.dumps(details), reference)
            payload["gateway"] = details
            payload["error"] += f"; Gateway diagnostics: {json.dumps(details)}"
        elif isinstance(exc, BaseExceptionGroup):
            details = grouped_error_details(exc)
            logger.error("Grouped failure metadata: {} [{}]", json.dumps(details), reference)
            payload["diagnostics"] = details
            payload["error"] += f"; Group diagnostics: {json.dumps(details)}"
        elif isinstance(exc, GenieContractError):
            logger.error("Genie contract metadata: {} [{}]", json.dumps(exc.diagnostics), reference)
            payload["diagnostics"] = exc.diagnostics
            payload["error"] += f"; Genie diagnostics: {json.dumps(exc.diagnostics)}"
        elif isinstance(exc, ToolArgumentsError):
            details = {"tool": exc.tool, "issues": exc.issues}
            logger.error("Model tool argument validation failed: {} [{}]", json.dumps(details), reference)
            payload["diagnostics"] = details
            payload["error"] = (
                f"The model generated invalid tool arguments; reference {reference}; "
                f"Tool diagnostics: {json.dumps(details)}"
            )
        elif isinstance(exc, AssistantOperationError):
            details = {"code": exc.code}
            if exc.tool:
                details["tool"] = exc.tool
            details.update(exc.details)
            logger.error("Application failure metadata: {} [{}]", json.dumps(details), reference)
            payload["diagnostics"] = details
            payload["error"] += f"; Application diagnostics: {json.dumps(details)}"
        return JSONResponse(
            payload,
            status_code=502,
        )

    return Starlette(
        routes=[
            Route("/", index),
            Route("/ui.js", script),
            Route("/ui.css", style),
            Route("/api/collection", collection),
            Route("/api/chat", chat, methods=["POST"]),
            Mount("/", mcp.streamable_http_app()),
        ],
        lifespan=lifespan,
        middleware=[Middleware(PrivateAccess, settings=settings)],
        exception_handlers={
            ValidationError: invalid_request,
            json.JSONDecodeError: invalid_request,
            Exception: operation_failed,
        },
    )


def create_user_app(workspace: WorkspaceClient) -> Starlette:
    """Load the collection on first consented request, retaining no user client or token between requests."""
    inner: Starlette | None = None
    lock = asyncio.Lock()
    shutdown = asyncio.Event()
    runner: asyncio.Task[None] | None = None

    async def run_lifespan(candidate: Starlette, ready: asyncio.Future[None]) -> None:
        try:
            async with candidate.router.lifespan_context(candidate):
                ready.set_result(None)
                await shutdown.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncGenerator[None]:
        try:
            yield
        finally:
            shutdown.set()
            if runner is not None:
                await runner

    async def dispatch(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal inner, runner
        if scope["type"] != "http":
            return
        request = Request(scope)
        token = request.headers.get("x-forwarded-access-token", "").strip()
        if not request.headers.get("x-forwarded-user") or not token:
            await JSONResponse(
                {"error": "User OAuth required: open the Databricks App and consent to SQL, Genie and Gateway access"},
                status_code=401,
            )(scope, receive, send)
            return
        delegated = WorkspaceClient(
            config=Config(
                host=workspace.config.host,
                token=token,
                auth_type="pat",
                http_timeout_seconds=15,
                retry_timeout_seconds=20,
            )
        )
        context = USER_WORKSPACE.set(delegated)
        try:
            async with lock:
                if inner is None:
                    try:
                        settings = await run_in_threadpool(load_settings, delegated)
                        if settings.sql_auth != "user":
                            raise ValueError("Run setup_assistant with sql_auth=user before starting this App")
                        if not is_allowed_owner(request.headers, settings.allowed_subjects):
                            await JSONResponse({"error": "Access denied"}, status_code=403)(scope, receive, send)
                            return
                        tools = BibliographicTools(
                            settings, workspace, sql_workspace=user_workspace, gateway_workspace=user_workspace
                        )
                        agent = BibliographicAgent(tools, lambda: create_gateway_openai_client(user_workspace()))
                        candidate = create_app(settings, tools, agent)
                        ready = asyncio.get_running_loop().create_future()
                        runner = asyncio.create_task(run_lifespan(candidate, ready), context=Context())
                        await ready
                        inner = candidate
                    except Exception as exc:
                        reference = uuid4().hex
                        logger.error(
                            "User-authorized App initialization failed: {} [{}]", type(exc).__name__, reference
                        )
                        await JSONResponse(
                            {
                                "error": f"Collection initialization failed ({type(exc).__name__}); reference {reference}. "
                                "Check setup_assistant, OAuth consent and your SQL permissions."
                            },
                            status_code=502,
                        )(scope, receive, send)
                        return
            await inner(scope, receive, send)
        finally:
            USER_WORKSPACE.reset(context)

    return Starlette(routes=[Mount("/", app=dispatch)], lifespan=lifespan)


def main() -> None:
    """Start only inside the private Databricks App deployment, with platform-injected OAuth credentials."""
    import uvicorn

    if not os.getenv("DATABRICKS_APP_NAME"):
        raise RuntimeError(
            "Run this entrypoint inside Databricks Apps; local tests use create_app with injected doubles"
        )
    workspace = WorkspaceClient(config=Config(http_timeout_seconds=15, retry_timeout_seconds=20))
    uvicorn.run(
        create_user_app(workspace),
        host="0.0.0.0",
        port=int(os.environ.get("DATABRICKS_APP_PORT", "8000")),
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
