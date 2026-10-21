# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Ingest papers from arXiv into Unity Catalog Volume and Delta Table."""

# COMMAND ----------

from databricks.sdk.runtime import dbutils

from autographrs.pipeline import run_ingestion

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("topic", "large language models graph rag")
dbutils.widgets.text("num_papers", "10")
dbutils.widgets.dropdown("ranking_criteria", "relevance", ["relevance", "submittedDate", "lastUpdatedDate"])
dbutils.widgets.text("criteria_date", "")
dbutils.widgets.text("categories", "")

env = dbutils.widgets.get("env")
topic = dbutils.widgets.get("topic")
num_papers = int(dbutils.widgets.get("num_papers"))
ranking_criteria = dbutils.widgets.get("ranking_criteria")
criteria_date = dbutils.widgets.get("criteria_date") or None
raw_categories = dbutils.widgets.get("categories")
categories = [c.strip() for c in raw_categories.split(",") if c.strip()] if raw_categories else None

# COMMAND ----------

result = run_ingestion(
    topic=topic,
    num_papers=num_papers,
    ranking_criteria=ranking_criteria,
    criteria_date=criteria_date,
    categories=categories,
    env=env,
    config_path="../../project_config.yml",
)

print(
    f"Run {result.run_id}: stored {len(result.papers)} papers in {result.delta_table} and PDFs in {result.volume_dir}"
)
