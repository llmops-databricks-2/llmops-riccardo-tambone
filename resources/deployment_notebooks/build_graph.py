# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Build a citation graph from one published ingestion/enrichment run."""

from databricks.sdk.runtime import dbutils

from autographrs.config import load_config
from autographrs.graph import GraphStore

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("bibliography_snapshot_id", "")
env = dbutils.widgets.get("env")
run_id = dbutils.widgets.get("run_id").strip()
bibliography_snapshot_id = dbutils.widgets.get("bibliography_snapshot_id").strip() or None
if not run_id:
    raise ValueError("The graph task requires the published ingestion run_id")

# COMMAND ----------

store = GraphStore(load_config("../../project_config.yml", env=env))
build = store.build_graph(run_id, bibliography_snapshot_id=bibliography_snapshot_id)
graph = store.load_graph(run_id, build.graph_build_id)
dbutils.jobs.taskValues.set(key="run_id", value=run_id)
dbutils.jobs.taskValues.set(key="graph_build_id", value=build.graph_build_id)
print(
    f"Graph {build.graph_build_id} for run {run_id}: "
    f"{len(graph.nodes)} nodes, {len(graph.edges)} CITES edges, {len(graph.warnings)} warnings. "
    f"Tables: {store.full_nodes_table_name}, {store.full_edges_table_name}, {store.full_builds_table_name}"
)
