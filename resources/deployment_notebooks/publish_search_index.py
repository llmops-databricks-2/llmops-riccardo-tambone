# Databricks notebook source
# ruff: noqa: ERA001
"""Prepare Lakebase prerequisites, then transactionally publish embedding snapshots and HNSW indexes."""

from databricks.sdk.runtime import dbutils

from autographrs.config import load_config
from autographrs.embeddings.search import SearchStore

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("embedding_snapshot_id", "")
run_id = dbutils.widgets.get("run_id").strip()
snapshot_id = dbutils.widgets.get("embedding_snapshot_id").strip()
if not run_id or not snapshot_id:
    raise ValueError("Index publication requires explicit run_id and embedding_snapshot_id")

store = SearchStore(load_config("../../project_config.yml", env=dbutils.widgets.get("env")))
publication = store.publish(run_id, snapshot_id)
store.load_publication(run_id, snapshot_id, publication.publication_id)
dbutils.jobs.taskValues.set(key="search_publication_id", value=publication.publication_id)
print(
    f"Search publication {publication.publication_id}: {publication.document_count} documents, {publication.index_name}"
)
