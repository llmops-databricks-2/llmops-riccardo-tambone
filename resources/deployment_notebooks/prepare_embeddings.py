# Databricks notebook source
# ruff: noqa: ERA001
"""Generate versioned embeddings through the course Gateway, without creating search compute."""

from databricks.sdk.runtime import dbutils

from autographrs.config import load_config
from autographrs.embeddings.storage import EmbeddingStore

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("bibliography_snapshot_id", "")
run_id = dbutils.widgets.get("run_id").strip()
bibliography_snapshot_id = dbutils.widgets.get("bibliography_snapshot_id").strip()
if not run_id or not bibliography_snapshot_id:
    raise ValueError("Embedding preparation requires explicit run_id and bibliography_snapshot_id")

store = EmbeddingStore(load_config("../../project_config.yml", env=dbutils.widgets.get("env")))
snapshot = store.prepare(run_id, bibliography_snapshot_id)
store.load_snapshot(run_id, snapshot.snapshot_id)
dbutils.jobs.taskValues.set(key="embedding_snapshot_id", value=snapshot.snapshot_id)
print(
    f"Embedding snapshot {snapshot.snapshot_id}: {snapshot.document_count} documents, "
    f"{snapshot.dimension} dimensions, {snapshot.skipped_count} skipped. Source: {snapshot.source_table}"
)
