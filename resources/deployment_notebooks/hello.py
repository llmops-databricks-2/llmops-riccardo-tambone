# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Check the deployed wheel, configuration and schema write access."""

# MAGIC %md
# MAGIC # Hello
# MAGIC
# MAGIC The first job of your course repo. It proves three things on the first deploy: your
# MAGIC `arxiv_curator` wheel installs on serverless, `project_config.yml` loads, and you can
# MAGIC write to your own schema. It appends one row to the table `hello` there.

# COMMAND ----------

from datetime import UTC, datetime

from databricks.sdk.runtime import dbutils
from pyspark.sql import SparkSession

from arxiv_curator.config import load_config

# The environment comes from the job's base parameters; "dev" when run by hand.
dbutils.widgets.text("env", "dev")
env = dbutils.widgets.get("env")

cfg = load_config("../../project_config.yml", env=env)
print(f"Configuration loaded for {cfg.full_schema_name}")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

table = f"{cfg.full_schema_name}.hello"
row = spark.createDataFrame(
    [("hello from arxiv_curator", datetime.now(UTC))],
    "message STRING, written_at TIMESTAMP",
)
row.write.mode("append").saveAsTable(table)
print(f"Wrote one row to {table}")
