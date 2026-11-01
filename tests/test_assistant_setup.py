"""Verify psycopg implementation selection occurs before setup imports the aborting native extension."""

import runpy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from test_embeddings import ROOT


@pytest.mark.parametrize("implementation", ["python", "binary"])
def test_setup_selects_python_libpq_before_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    implementation: str,
) -> None:
    """Setup uses the supported system-libpq wrapper and rejects an already loaded binary wrapper."""
    monkeypatch.delenv("PSYCOPG_IMPL", raising=False)
    psycopg = ModuleType("psycopg")
    psycopg.pq = SimpleNamespace(__impl__=implementation, version=lambda: 170000)
    runtime = ModuleType("databricks.sdk.runtime")
    runtime.dbutils = MagicMock()
    runtime.spark = MagicMock()
    values = {
        "env": "dev",
        "app_name": "autographrs-rtambone-dev",
        "run_id": "run",
        "graph_build_id": "build",
        "parent_path": "/Workspace/private",
        "sql_auth": "user",
    }
    runtime.dbutils.widgets.get.side_effect = values.__getitem__
    bootstrap = ModuleType("autographrs.assistant.bootstrap")
    bootstrap.configure_assistant = MagicMock()
    with (
        patch.dict(
            sys.modules,
            {
                "psycopg": psycopg,
                "databricks.sdk.runtime": runtime,
                "autographrs.assistant.bootstrap": bootstrap,
            },
        ),
        patch("databricks.sdk.WorkspaceClient"),
        patch("autographrs.config.load_config"),
    ):
        if implementation == "binary":
            with pytest.raises(RuntimeError, match="Restart the Python session"):
                runpy.run_path(str(ROOT / "resources" / "deployment_notebooks" / "setup_assistant.py"))
            bootstrap.configure_assistant.assert_not_called()
        else:

            def configure(**kwargs: object) -> MagicMock:
                import os

                assert os.environ["PSYCOPG_IMPL"] == "python"
                assert kwargs["sql_auth"] == "user"
                return MagicMock()

            bootstrap.configure_assistant.side_effect = configure
            runpy.run_path(str(ROOT / "resources" / "deployment_notebooks" / "setup_assistant.py"))
            bootstrap.configure_assistant.assert_called_once()
