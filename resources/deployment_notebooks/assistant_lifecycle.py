# Databricks notebook source
# ruff: noqa: ERA001
"""Stop only after ingestion; start only after successful setup and verify the approved collection."""

from pathlib import Path

import yaml
from databricks.sdk import WorkspaceClient
from databricks.sdk.config import Config
from databricks.sdk.runtime import dbutils

from autographrs.assistant.orchestration import CollectionOutput, start_assistant, stop_assistant

names = ("action", "app_name", "env", "run_id", "graph_build_id", "source_code_path")
for name in names:
    dbutils.widgets.text(name, "")
values = {name: dbutils.widgets.get(name).strip() for name in names}
if not values["app_name"]:
    raise ValueError("App lifecycle requires an explicit app_name")
workspace = WorkspaceClient(config=Config(http_timeout_seconds=30, retry_timeout_seconds=60))
if values["action"] == "stop":
    stop_assistant(workspace, values["app_name"])
elif values["action"] == "start":
    collection = CollectionOutput.model_validate(
        {
            "env": values["env"],
            "run_id": values["run_id"],
            "graph_build_id": values["graph_build_id"],
        }
    )
    command = yaml.safe_load(Path("../../app.yaml").read_text())["command"]
    start_assistant(workspace, values["app_name"], collection, values["source_code_path"], command)
else:
    raise ValueError("App lifecycle action must be stop or start")
