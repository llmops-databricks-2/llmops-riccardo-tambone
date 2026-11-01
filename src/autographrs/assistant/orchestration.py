"""Job-only collection handoff and controlled App lifecycle; never invoked by assistant tools."""

import time
from datetime import timedelta
from typing import Literal
from urllib.parse import urlparse

import httpx
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import (
    AppDeployment,
    AppDeploymentMode,
    ApplicationState,
    ComputeState,
    EnvVar,
)
from databricks.sdk.service.jobs import RunResultState
from loguru import logger
from pydantic import Field, field_validator

from autographrs.assistant.models import StrictModel
from autographrs.storage.delta import _validate_run_id


class CollectionOutput(StrictModel):
    """Small versioned notebook exit payload for a successfully published collection."""

    version: Literal[1] = 1
    env: Literal["dev", "acc", "prd"]
    run_id: str = Field(min_length=1, max_length=160)
    graph_build_id: str = Field(min_length=1, max_length=160)

    @field_validator("run_id", "graph_build_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        """Use the same identifier contract as the snapshot stores."""
        return _validate_run_id(value)


def resolve_collection(
    workspace: WorkspaceClient, pipeline_task_run_id: int, pipeline_job_id: int, env: str
) -> tuple[int, CollectionOutput]:
    """Resolve the exact Run Job task's child and its successful graph task, never the latest run."""
    if pipeline_task_run_id <= 0 or pipeline_job_id <= 0:
        raise ValueError("Positive Databricks task run and pipeline job IDs are required")
    output = workspace.jobs.get_run_output(pipeline_task_run_id)
    if output.error or not output.run_job_output or not output.run_job_output.run_id:
        raise ValueError("Pipeline task has no successful child run output; deploy the updated pipeline first")
    child_id = output.run_job_output.run_id
    token = None
    seen_tokens: set[str] = set()
    graph_tasks = []
    while True:
        child = workspace.jobs.get_run(child_id, page_token=token)
        if (
            child.run_id != child_id
            or child.job_id != pipeline_job_id
            or not child.state
            or child.state.result_state != RunResultState.SUCCESS
        ):
            raise ValueError("Referenced pipeline execution is not a successful run of the expected job")
        graph_tasks.extend(task for task in (child.tasks or []) if task.task_key == "build_graph")
        token = child.next_page_token
        if not token:
            break
        if token in seen_tokens:
            raise ValueError("Pipeline enumeration returned a repeated pagination token")
        seen_tokens.add(token)
    if len(graph_tasks) != 1:
        raise ValueError("Expected exactly one build_graph task in the child execution")
    task = graph_tasks[0]
    if not task.run_id or not task.state or task.state.result_state != RunResultState.SUCCESS:
        raise ValueError("Child graph task is not successful")
    graph_output = workspace.jobs.get_run_output(task.run_id)
    notebook = graph_output.notebook_output
    if graph_output.error or not notebook or notebook.truncated or not notebook.result:
        raise ValueError("Child graph output is missing, failed or truncated")
    if len(notebook.result) > 4096:
        raise ValueError("Child graph output exceeds its 4096-character contract")
    collection = CollectionOutput.model_validate_json(notebook.result)
    if collection.env != env:
        raise ValueError("Child collection belongs to a different environment")
    logger.info(
        "Resolved pipeline execution {} to collection {} / {}", child_id, collection.run_id, collection.graph_build_id
    )
    return child_id, collection


def stop_assistant(workspace: WorkspaceClient, app_name: str) -> None:
    """Wait for confirmed stopped compute before any setup can modify grants or Genie."""
    app = workspace.apps.get(app_name)
    if app.pending_deployment:
        raise ValueError("App has a pending deployment; finish it before changing collection")
    if app.compute_status and app.compute_status.state == ComputeState.STOPPED:
        logger.info("Assistant {} is already stopped", app_name)
        return
    stopped = workspace.apps.stop(app_name).result(timeout=timedelta(minutes=20))
    if not stopped.compute_status or stopped.compute_status.state != ComputeState.STOPPED:
        raise RuntimeError("App stop did not confirm stopped compute; setup must not run")
    logger.info("Assistant {} stopped; collection setup may proceed", app_name)


def _check_collection_url(url: str) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.endswith(".databricksapps.com")
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Readiness requires a platform-provided Databricks App HTTPS URL")


def verify_assistant(workspace: WorkspaceClient, app_name: str, collection: CollectionOutput) -> None:
    """Require running compute/application and an authenticated HTTP response for the exact approved collection."""
    deadline = time.monotonic() + 180
    while True:
        app = workspace.apps.get(app_name)
        compute = app.compute_status.state if app.compute_status else None
        application = app.app_status.state if app.app_status else None
        if compute in (ComputeState.ERROR, ComputeState.STOPPED) or application == ApplicationState.CRASHED:
            raise RuntimeError("Assistant failed to start; inspect App logs and repair the start task")
        if compute == ComputeState.ACTIVE and application == ApplicationState.RUNNING:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Assistant did not reach running state within 180 seconds")
        time.sleep(3)
    url = app.url or ""
    _check_collection_url(url)
    # Use owner OAuth through the real Apps proxy; never forge X-Forwarded-User.
    with httpx.Client(timeout=20, follow_redirects=False) as client:
        response = client.get(
            f"{url.rstrip('/')}/api/collection",
            headers=workspace.config.authenticate(),
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or (
        payload.get("run_id") != collection.run_id or payload.get("graph_build_id") != collection.graph_build_id
    ):
        raise ValueError("Running App does not expose the collection approved by this orchestration")
    logger.info("Assistant {} is responsive on the approved collection", app_name)


def start_assistant(
    workspace: WorkspaceClient,
    app_name: str,
    collection: CollectionOutput,
    source_code_path: str,
    command: list[str],
) -> None:
    """Restart the existing deployment, or deploy Bundle sources once after the first successful setup."""
    app = workspace.apps.get(app_name)
    if app.pending_deployment:
        raise ValueError("App has a pending deployment; wait before starting it")
    if app.active_deployment:
        if not app.compute_status or app.compute_status.state != ComputeState.ACTIVE:
            workspace.apps.start(app_name).result(timeout=timedelta(minutes=20))
    else:
        if (
            not source_code_path.startswith("/Workspace/")
            or not command
            or not all(isinstance(part, str) and part for part in command)
        ):
            raise ValueError("First App deployment requires Bundle workspace sources and a nonempty command")
        workspace.apps.deploy(
            app_name,
            AppDeployment(
                source_code_path=source_code_path,
                mode=AppDeploymentMode.SNAPSHOT,
                command=command,
                env_vars=[
                    EnvVar(name="UV_NO_DEV", value="true"),
                    EnvVar(name="AUTOGRAPHRS_ENV", value=collection.env),
                    EnvVar(name="AUTOGRAPHRS_APP_NAME", value=app_name),
                ],
            ),
        ).result(timeout=timedelta(minutes=20))
    verify_assistant(workspace, app_name, collection)
