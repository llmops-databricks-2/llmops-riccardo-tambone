# Databricks notebook source
# ruff: noqa: ERA001
"""Bridge exact child-job notebook outputs into parent-job task values."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils

from autographrs.assistant.orchestration import resolve_collection

for name in ("env", "pipeline_task_run_id", "pipeline_job_id"):
    dbutils.widgets.text(name, "")
values = {name: dbutils.widgets.get(name).strip() for name in ("env", "pipeline_task_run_id", "pipeline_job_id")}
if not all(values.values()):
    raise ValueError("Collection handoff requires env, pipeline_task_run_id and pipeline_job_id")
child_id, collection = resolve_collection(
    WorkspaceClient(),
    int(values["pipeline_task_run_id"]),
    int(values["pipeline_job_id"]),
    values["env"],
)
for key, value in (
    ("pipeline_run_id", child_id),
    ("run_id", collection.run_id),
    ("graph_build_id", collection.graph_build_id),
):
    dbutils.jobs.taskValues.set(key=key, value=value)
print(f"Resolved pipeline execution {child_id}: collection {collection.run_id}, graph {collection.graph_build_id}")
