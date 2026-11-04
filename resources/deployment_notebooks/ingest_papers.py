# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Ingest papers from arXiv into Unity Catalog Volume and Delta Table."""

# COMMAND ----------

from databricks.sdk.runtime import dbutils

from autographrs.config import load_config
from autographrs.pipeline import run_ingestion
from autographrs.storage.delta import DeltaPaperStore

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("topic", "neuroscience, peririnal cortex")
dbutils.widgets.text("num_papers", "10")
dbutils.widgets.dropdown("ranking_criteria", "relevance", ["relevance", "submittedDate", "lastUpdatedDate"])
dbutils.widgets.text("criteria_date", "")
dbutils.widgets.text("categories", "")
dbutils.widgets.text("run_id", "")

env = dbutils.widgets.get("env")
topic = dbutils.widgets.get("topic")
num_papers = int(dbutils.widgets.get("num_papers"))
ranking_criteria = dbutils.widgets.get("ranking_criteria")
criteria_date = dbutils.widgets.get("criteria_date") or None
raw_categories = dbutils.widgets.get("categories")
categories = [c.strip() for c in raw_categories.split(",") if c.strip()] if raw_categories else None
run_id = dbutils.widgets.get("run_id").strip() or None

# COMMAND ----------

result = run_ingestion(
    topic=topic,
    num_papers=num_papers,
    ranking_criteria=ranking_criteria,
    criteria_date=criteria_date,
    categories=categories,
    env=env,
    config_path="../../project_config.yml",
    run_id=run_id,
)

# COMMAND ----------

store = DeltaPaperStore(load_config("../../project_config.yml", env=env))
published = store.load_run(result.run_id)
dataset = store.load_run_dataset(result.run_id)
assert len(dataset) == published.paper_count == published.enrichment_count
print(f"Verified published run {published.run_id}: status={published.status}, warnings={published.warnings}")
references = [reference for _, enrichment in dataset for reference in (enrichment.bibliography or [])]
print(
    "Retrieved references, including external works, are in bibliography. "
    "Citation relationships will be built separately by the graph module."
)
print("Citation retrieval: Scholar batch first, OpenAlex for empty results; per-provider outcomes in citation_lookups.")
print(
    f"Bibliography: {len(references)} entries, "
    f"{sum(bool(reference.title) for reference in references)} with cited-work titles, "
    f"{sum(reference.match_method == 'unresolved' for reference in references)} unresolved. "
    "External references are retained in paper_enrichment.bibliography."
)

print(
    f"Run {result.run_id}: stored {len(result.papers)} papers in {result.delta_table}, "
    f"{len(result.enrichments)} enrichments in {result.enrichment_table}, and PDFs in {result.volume_dir}"
)

dbutils.jobs.taskValues.set(key="run_id", value=result.run_id)
