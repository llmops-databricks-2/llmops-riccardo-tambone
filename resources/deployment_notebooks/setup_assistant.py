# Databricks notebook source
# ruff: noqa: ERA001, E402
"""Approve one published collection, prepare private Genie and provision read-only App access."""

import os

# Select before bootstrap imports psycopg: the binary extension aborts this notebook's runtime.
os.environ["PSYCOPG_IMPL"] = "python"
print("Assistant setup: loading psycopg Python implementation with system libpq", flush=True)

import psycopg

if psycopg.pq.__impl__ != "python":
    raise RuntimeError("Restart the Python session: psycopg was already loaded with a different implementation")
print(f"Assistant setup: psycopg implementation=python, libpq={psycopg.pq.version()}", flush=True)

from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils, spark

from autographrs.assistant.bootstrap import configure_assistant
from autographrs.assistant.trace_permissions import CLIENT_ID, grant_trace_access
from autographrs.config import load_config

for name, default in (
    ("env", "dev"),
    ("app_name", ""),
    ("run_id", ""),
    ("graph_build_id", ""),
    ("parent_path", ""),
    ("sql_auth", "user"),
):
    dbutils.widgets.text(name, default)
values = {
    name: dbutils.widgets.get(name).strip()
    for name in ("env", "app_name", "run_id", "graph_build_id", "parent_path", "sql_auth")
}
if not all(values.values()):
    raise ValueError("Assistant setup requires env, app_name, run_id, graph_build_id and parent_path")
if values["env"] not in ("dev", "acc", "prd") or values["app_name"] != f"autographrs-rtambone-{values['env']}":
    raise ValueError("Assistant setup may grant credentials only to the AutoGraph App for this environment")

config = load_config("../../project_config.yml", env=values["env"])
workspace = WorkspaceClient()
settings = configure_assistant(
    config=config,
    app_name=values["app_name"],
    run_id=values["run_id"],
    graph_build_id=values["graph_build_id"],
    spark=spark,
    workspace=workspace,
    parent_path=values["parent_path"],
    sql_auth=values["sql_auth"],
)

# COMMAND ----------

writer_client_id = dbutils.secrets.get(catalog=config.catalog, schema=config.db_schema, key=CLIENT_ID)
grant_trace_access(workspace, config, values["app_name"], writer_client_id)
dbutils.jobs.taskValues.set(key="genie_space_id", value=settings.genie_space_id)
print(
    f"Approved collection {settings.graph.graph_build_id}; Genie {settings.genie_space_id}. Now deploy/start the App."
)
