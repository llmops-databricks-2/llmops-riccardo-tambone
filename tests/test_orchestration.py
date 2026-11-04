"""Exact child-run handoff, safe App cutover and native Bundle job dependencies."""

import json
import runpy
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml
from databricks.sdk.service.apps import (
    App,
    AppDeployment,
    ApplicationState,
    ApplicationStatus,
    ComputeState,
    ComputeStatus,
)
from databricks.sdk.service.jobs import (
    NotebookOutput,
    Run,
    RunJobOutput,
    RunOutput,
    RunResultState,
    RunState,
    RunTask,
)
from pydantic import ValidationError
from test_embeddings import ROOT

from autographrs.assistant.orchestration import (
    CollectionOutput,
    resolve_collection,
    start_assistant,
    stop_assistant,
    verify_assistant,
)

APP = "autographrs-test-dev"
URL = f"https://{APP}.databricksapps.com"
COLLECTION = CollectionOutput(env="dev", run_id="collection-123", graph_build_id="build-456")
SUCCESS = RunState(result_state=RunResultState.SUCCESS)


def _workspace() -> MagicMock:
    workspace = MagicMock()
    workspace.jobs.get_run_output.side_effect = [
        RunOutput(
            error="Please refer to the logs for this run on the triggered run details page.",
            metadata=Run(state=SUCCESS),
            run_job_output=RunJobOutput(run_id=200),
        ),
        RunOutput(notebook_output=NotebookOutput(result=COLLECTION.model_dump_json(), truncated=False)),
    ]
    workspace.jobs.get_run.return_value = Run(
        job_id=10,
        run_id=200,
        state=SUCCESS,
        tasks=[RunTask(task_key="build_graph", run_id=300, state=SUCCESS)],
    )
    workspace.apps.get.return_value = App(
        name=APP,
        url=URL,
        active_deployment=AppDeployment(deployment_id="existing"),
        compute_status=ComputeStatus(state=ComputeState.ACTIVE),
        app_status=ApplicationStatus(state=ApplicationState.RUNNING),
    )
    workspace.config.authenticate.return_value = {"Authorization": "Bearer example"}
    return workspace


def test_handoff_resolves_exact_successful_child_and_graph_output() -> None:
    """Wrapper ID, child-job ID and collection ID remain distinct, including paginated task lookup."""
    workspace = _workspace()
    graph_page = workspace.jobs.get_run.return_value
    workspace.jobs.get_run.side_effect = [
        Run(job_id=10, run_id=200, state=SUCCESS, tasks=[], next_page_token="page2"),
        graph_page,
    ]
    child_id, collection = resolve_collection(workspace, 100, 10, "dev")
    assert child_id == 200 and collection == COLLECTION
    assert [call.args[0] for call in workspace.jobs.get_run_output.call_args_list] == [100, 300]
    assert [call.kwargs["page_token"] for call in workspace.jobs.get_run.call_args_list] == [None, "page2"]
    workspace.jobs.list_runs.assert_not_called()
    workspace.apps.stop.assert_not_called()


@pytest.mark.parametrize(
    "mode",
    [
        "invalid_id",
        "missing_child",
        "child_error",
        "wrong_job",
        "wrong_run",
        "failed_child",
        "missing_graph",
        "duplicate_graph",
        "failed_graph",
        "missing_notebook",
        "truncated",
        "malformed",
        "extra_scope",
        "wrong_env",
        "oversized",
        "invalid_collection_id",
        "pagination_loop",
    ],
)
def test_handoff_rejects_incomplete_foreign_or_invalid_outputs(mode: str) -> None:
    """No fallback to a global latest run or a partial child execution is allowed."""
    workspace = _workspace()
    child = workspace.jobs.get_run.return_value
    wrapper = RunOutput(run_job_output=RunJobOutput(run_id=200))
    graph_output = RunOutput(notebook_output=NotebookOutput(result=COLLECTION.model_dump_json()))
    if mode == "missing_child":
        wrapper.run_job_output = None
    elif mode == "child_error":
        wrapper.error = "child failed"
    elif mode == "wrong_job":
        child.job_id = 99
    elif mode == "wrong_run":
        child.run_id = 99
    elif mode == "failed_child":
        child.state = RunState(result_state=RunResultState.FAILED)
    elif mode == "missing_graph":
        child.tasks = []
    elif mode == "duplicate_graph":
        child.tasks = [*child.tasks, *child.tasks]
    elif mode == "failed_graph":
        child.tasks[0].state = RunState(result_state=RunResultState.FAILED)
    elif mode == "missing_notebook":
        graph_output.notebook_output = None
    elif mode == "truncated":
        graph_output.notebook_output.truncated = True
    elif mode == "malformed":
        graph_output.notebook_output.result = "not json"
    elif mode == "extra_scope":
        graph_output.notebook_output.result = json.dumps({**COLLECTION.model_dump(), "table": "other"})
    elif mode == "wrong_env":
        graph_output.notebook_output.result = COLLECTION.model_copy(update={"env": "prd"}).model_dump_json()
    elif mode == "oversized":
        graph_output.notebook_output.result = "x" * 4097
    elif mode == "invalid_collection_id":
        graph_output.notebook_output.result = COLLECTION.model_copy(update={"run_id": "../other"}).model_dump_json()
    elif mode == "pagination_loop":
        child.next_page_token = "same"
    workspace.jobs.get_run_output.side_effect = [wrapper, graph_output]
    with pytest.raises((ValueError, ValidationError)):
        resolve_collection(workspace, -1 if mode == "invalid_id" else 100, 10, "dev")
    workspace.apps.stop.assert_not_called()


@pytest.mark.parametrize("already_stopped", [False, True])
def test_stop_waits_for_confirmed_compute_stop_and_is_repeatable(already_stopped: bool) -> None:
    """A repaired stop task does not need another stop operation if compute is already stopped."""
    workspace = _workspace()
    stopped = App(name=APP, compute_status=ComputeStatus(state=ComputeState.STOPPED))
    if already_stopped:
        workspace.apps.get.return_value = stopped
    workspace.apps.stop.return_value.result.return_value = stopped
    stop_assistant(workspace, APP)
    assert workspace.apps.stop.call_count == (0 if already_stopped else 1)


def test_lifecycle_rejects_pending_deployments_and_unconfirmed_stop() -> None:
    """Cutover must not race a code deployment or continue after a failed stop."""
    workspace = _workspace()
    workspace.apps.get.return_value.pending_deployment = AppDeployment(deployment_id="pending")
    with pytest.raises(ValueError, match="pending"):
        stop_assistant(workspace, APP)
    with pytest.raises(ValueError, match="pending"):
        start_assistant(workspace, APP, COLLECTION, "/Workspace/bundle/files", ["python"])
    workspace.apps.get.return_value.pending_deployment = None
    workspace.apps.stop.return_value.result.return_value = App(
        name=APP,
        compute_status=ComputeStatus(state=ComputeState.ACTIVE),
    )
    with pytest.raises(RuntimeError, match="stop"):
        stop_assistant(workspace, APP)


@pytest.mark.parametrize("existing", [True, False])
def test_start_reuses_deployment_or_bootstraps_bundle_sources_with_target_env(existing: bool) -> None:
    """Existing code is restarted, not replaced; first deployment uses the Bundle's sources and pinned App command."""
    workspace = _workspace()
    app = workspace.apps.get.return_value
    app.compute_status.state = ComputeState.STOPPED
    if not existing:
        app.active_deployment = None
    command = yaml.safe_load((ROOT / "app.yaml").read_text())["command"]
    with patch("autographrs.assistant.orchestration.verify_assistant") as verify:
        start_assistant(workspace, APP, COLLECTION, "/Workspace/bundle/files", command)
    if existing:
        workspace.apps.start.assert_called_once_with(APP)
        workspace.apps.start.return_value.result.assert_called_once()
        workspace.apps.deploy.assert_not_called()
    else:
        workspace.apps.start.assert_not_called()
        deployment = workspace.apps.deploy.call_args.args[1]
        assert deployment.source_code_path == "/Workspace/bundle/files"
        assert deployment.command == command
        assert {var.name: var.value for var in deployment.env_vars} == {
            "UV_NO_DEV": "true",
            "AUTOGRAPHRS_ENV": "dev",
            "AUTOGRAPHRS_APP_NAME": APP,
        }
        workspace.apps.deploy.return_value.result.assert_called_once()
    verify.assert_called_once_with(workspace, APP, COLLECTION)


def test_repaired_start_checks_running_app_without_creating_a_second_deployment() -> None:
    """If start already succeeded remotely, a retry verifies scope instead of duplicating a deployment."""
    workspace = _workspace()
    with patch("autographrs.assistant.orchestration.verify_assistant") as verify:
        start_assistant(workspace, APP, COLLECTION, "/Workspace/bundle/files", ["python"])
    workspace.apps.start.assert_not_called()
    workspace.apps.deploy.assert_not_called()
    verify.assert_called_once()


@pytest.mark.parametrize("mode", ["ok", "wrong_collection", "forbidden", "crashed", "timeout", "foreign_url"])
def test_readiness_requires_running_status_and_authenticated_exact_collection(mode: str) -> None:
    """Compute ACTIVE alone is insufficient; HTTP must prove the manifest was loaded through the platform proxy."""
    workspace = _workspace()
    if mode == "crashed":
        workspace.apps.get.return_value.app_status.state = ApplicationState.CRASHED
    elif mode == "timeout":
        workspace.apps.get.return_value.app_status.state = ApplicationState.DEPLOYING
    elif mode == "foreign_url":
        workspace.apps.get.return_value.url = "https://other.example"
    payload = COLLECTION.model_dump()
    if mode == "wrong_collection":
        payload["graph_build_id"] = "different"
    response = httpx.Response(
        403 if mode == "forbidden" else 200,
        json=payload,
        request=httpx.Request("GET", f"{URL}/api/collection"),
    )
    with (
        patch("autographrs.assistant.orchestration.httpx.Client") as clients,
        patch("autographrs.assistant.orchestration.time.monotonic", side_effect=[0, 181]),
    ):
        client = clients.return_value.__enter__.return_value
        client.get.return_value = response
        if mode == "ok":
            verify_assistant(workspace, APP, COLLECTION)
            client.get.assert_called_once_with(
                f"{URL}/api/collection",
                headers=workspace.config.authenticate.return_value,
            )
        else:
            with pytest.raises((ValueError, RuntimeError, TimeoutError, httpx.HTTPStatusError)):
                verify_assistant(workspace, APP, COLLECTION)
        if mode == "foreign_url":
            workspace.config.authenticate.assert_not_called()


def test_orchestrator_bundle_reuses_jobs_and_only_stops_after_handoff() -> None:
    """The native DAG makes failed pipeline/handoff/setup skip all unsafe downstream operations."""
    jobs = yaml.safe_load((ROOT / "resources" / "assistant_orchestrator.job.yml").read_text())["resources"]["jobs"]
    job = jobs["refresh_assistant"]
    assert job["max_concurrent_runs"] == 1 and job["queue"] == {"enabled": True}
    assert job["environments"][0]["spec"]["environment_version"] == "5"
    tasks = job["tasks"]
    assert [task["task_key"] for task in tasks] == [
        "pipeline",
        "resolve_collection",
        "stop_assistant",
        "configure_assistant",
        "start_assistant",
    ]
    for previous, task in zip(tasks[:-1], tasks[1:], strict=True):
        assert task["depends_on"] == [{"task_key": previous["task_key"]}]
        assert task.get("run_if", "ALL_SUCCESS") == "ALL_SUCCESS"
    assert tasks[0]["run_job_task"]["job_id"] == "${resources.jobs.ingest_papers.id}"
    assert tasks[3]["run_job_task"]["job_id"] == "${resources.jobs.setup_assistant.id}"
    assert tasks[0]["max_retries"] == tasks[3]["max_retries"] == 0
    assert tasks[1]["notebook_task"]["base_parameters"]["pipeline_task_run_id"] == "{{tasks.pipeline.run_id}}"
    handoff = tasks[3]["run_job_task"]["job_parameters"]
    start = tasks[4]["notebook_task"]["base_parameters"]
    for key in ("run_id", "graph_build_id"):
        assert handoff[key] == start[key] == f"{{{{tasks.resolve_collection.values.{key}}}}}"
    pipeline = yaml.safe_load((ROOT / "resources" / "autograph_pipeline.job.yml").read_text())["resources"]["jobs"][
        "ingest_papers"
    ]
    assert tasks[0]["run_job_task"]["job_parameters"] == {
        param["name"]: f"{{{{job.parameters.{param['name']}}}}}" for param in pipeline["parameters"]
    }


def test_shutdown_bundle_runs_only_stop_at_italian_local_hours() -> None:
    """Keep both daily local-time triggers active even in development mode, without setup or restart."""
    jobs = yaml.safe_load((ROOT / "resources" / "assistant_shutdown.job.yml").read_text())["resources"]["jobs"]
    job = jobs["shutdown_assistant"]
    assert job["schedule"] == {
        "quartz_cron_expression": "0 0 13,20 * * ?",
        "timezone_id": "Europe/Rome",
        "pause_status": "UNPAUSED",
    }
    assert job["max_concurrent_runs"] == 1
    assert job["environments"][0]["spec"] == {
        "environment_version": "5",
        "dependencies": ["../dist/*.whl"],
    }
    assert job["tasks"] == [
        {
            "task_key": "stop_assistant",
            "environment_key": "default",
            "timeout_seconds": 1500,
            "notebook_task": {
                "notebook_path": "deployment_notebooks/assistant_lifecycle.py",
                "base_parameters": {
                    "action": "stop",
                    "app_name": "${resources.apps.bibliographic_assistant.name}",
                },
            },
        }
    ]


def test_handoff_notebook_publishes_values_and_lifecycle_notebook_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise deployed notebook wrappers, not just their helper functions."""
    dbutils = MagicMock()
    runtime = ModuleType("databricks.sdk.runtime")
    runtime.dbutils = dbutils
    values = {"env": "dev", "pipeline_task_run_id": "100", "pipeline_job_id": "10"}
    dbutils.widgets.get.side_effect = values.__getitem__
    with (
        patch.dict(sys.modules, {"databricks.sdk.runtime": runtime}),
        patch("databricks.sdk.WorkspaceClient"),
        patch("autographrs.assistant.orchestration.resolve_collection", return_value=(200, COLLECTION)),
    ):
        runpy.run_path(str(ROOT / "resources" / "deployment_notebooks" / "resolve_assistant_collection.py"))
    assert {call.kwargs["key"]: call.kwargs["value"] for call in dbutils.jobs.taskValues.set.call_args_list} == {
        "pipeline_run_id": 200,
        "run_id": COLLECTION.run_id,
        "graph_build_id": COLLECTION.graph_build_id,
    }
    values = {
        "action": "stop",
        "app_name": APP,
        "env": "dev",
        "run_id": COLLECTION.run_id,
        "graph_build_id": COLLECTION.graph_build_id,
        "source_code_path": "/Workspace/bundle/files",
    }
    dbutils.widgets.get.side_effect = values.__getitem__
    monkeypatch.chdir(ROOT / "resources" / "deployment_notebooks")
    with (
        patch.dict(sys.modules, {"databricks.sdk.runtime": runtime}),
        patch("databricks.sdk.WorkspaceClient"),
        patch("databricks.sdk.config.Config"),
        patch("autographrs.assistant.orchestration.stop_assistant") as stop,
        patch("autographrs.assistant.orchestration.start_assistant") as start,
    ):
        runpy.run_path(str(ROOT / "resources" / "deployment_notebooks" / "assistant_lifecycle.py"))
        values["action"] = "start"
        runpy.run_path(str(ROOT / "resources" / "deployment_notebooks" / "assistant_lifecycle.py"))
        assert start.call_args.args[1:4] == (APP, COLLECTION, "/Workspace/bundle/files")
        assert start.call_args.args[4] == yaml.safe_load((ROOT / "app.yaml").read_text())["command"]
        values["action"] = "invalid"
        with pytest.raises(ValueError, match="action"):
            runpy.run_path(str(ROOT / "resources" / "deployment_notebooks" / "assistant_lifecycle.py"))
    assert stop.call_args.args[1] == APP
