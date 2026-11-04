# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Retrieve one-hop cited-work abstracts without changing the original bibliography."""

from databricks.sdk.runtime import dbutils

from autographrs.config import load_config
from autographrs.enrichment.bibliography import BibliographyStore

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "")
env = dbutils.widgets.get("env")
run_id = dbutils.widgets.get("run_id").strip()
if not run_id:
    raise ValueError("Bibliography enrichment requires a published ingestion run_id")

# COMMAND ----------

store = BibliographyStore(load_config("../../project_config.yml", env=env))
snapshot = store.enrich_bibliography(run_id)
metadata = store.load_metadata(run_id, snapshot.snapshot_id)
dbutils.jobs.taskValues.set(key="bibliography_snapshot_id", value=snapshot.snapshot_id)
print(
    f"Bibliography snapshot {snapshot.snapshot_id} for run {run_id}: {len(metadata)} cited nodes; "
    f"{sum(item.text_variant == 'title_abstract' for item in metadata)} title_abstract, "
    f"{sum(item.text_variant == 'title_only' for item in metadata)} title_only, "
    f"{sum(item.text_variant is None for item in metadata)} without usable text. "
    f"Warnings: {len(snapshot.warnings)}"
)
