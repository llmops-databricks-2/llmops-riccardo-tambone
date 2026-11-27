"""Verify the explicit boundary between the App principal and the MLflow writer."""

from unittest.mock import MagicMock, call

import pytest
from databricks.sdk.errors import PermissionDenied
from databricks.sdk.service.catalog import PermissionsChange, Privilege
from test_embeddings import _config

from autographrs.assistant.trace_permissions import grant_trace_access, read_writer_credentials

APP_ID = "00000000-0000-4000-8000-000000000001"
WRITER_ID = "00000000-0000-4000-8000-000000000002"


def test_grants_are_explicit_and_target_separate_principals() -> None:
    """Only the writer receives table grants; only the App receives access to startup secrets."""
    workspace, config = MagicMock(), _config()
    workspace.apps.get.return_value.service_principal_client_id = APP_ID
    grant_trace_access(workspace, config, "autograph-test", WRITER_ID)

    namespace = f"{config.catalog}.{config.db_schema}"
    workspace.apps.get.assert_called_once_with(name="autograph-test")
    assert workspace.api_client.do.call_args_list == [
        call(
            "PATCH",
            f"/api/2.1/unity-catalog/permissions/{kind}/{name}",
            body={"changes": [{"principal": APP_ID, "add": [privilege]}]},
        )
        for kind, name, privilege in (
            ("schema", namespace, "USE_SCHEMA"),
            ("secret", f"{namespace}.databricks_client_id", "READ_SECRET"),
            ("secret", f"{namespace}.databricks_client_secret", "READ_SECRET"),
        )
    ]
    assert workspace.grants.update.call_args_list == [
        call(
            securable_type="table",
            full_name=f"{namespace}.{name}",
            changes=[
                PermissionsChange(
                    principal=WRITER_ID,
                    add=[Privilege.SELECT, Privilege.MODIFY] if name.startswith("llmops_otel_") else [Privilege.SELECT],
                ),
            ],
        )
        for name in (
            "llmops_otel_spans",
            "llmops_otel_logs",
            "llmops_otel_annotations",
            "llmops_otel_metrics",
            "llmops_trace_metadata",
            "llmops_trace_unified",
        )
    ]


def test_schema_denial_stops_setup() -> None:
    """The notebook cannot bypass a missing grantor privilege."""
    workspace = MagicMock()
    workspace.apps.get.return_value.service_principal_client_id = APP_ID
    workspace.api_client.do.side_effect = PermissionDenied("Denied")
    with pytest.raises(PermissionDenied, match="cannot grant USE_SCHEMA"):
        grant_trace_access(workspace, _config(), "autograph-test", WRITER_ID)
    assert workspace.api_client.do.call_count == 1
    workspace.grants.update.assert_not_called()


def test_reader_uses_uc_value_endpoint_without_returning_metadata() -> None:
    """Fetch credential values with the existing App-authenticated client."""
    workspace, config = MagicMock(), _config()
    workspace.api_client.do.side_effect = [{"effective_value": WRITER_ID}, {"effective_value": "synthetic-secret"}]
    assert read_writer_credentials(workspace, config) == (WRITER_ID, "synthetic-secret")
    assert workspace.api_client.do.call_args_list == [
        call(
            "GET",
            f"/api/2.1/unity-catalog/secrets/{config.catalog}.{config.db_schema}.{key}",
            query={"include_value": "true"},
        )
        for key in ("databricks_client_id", "databricks_client_secret")
    ]
    workspace.api_client.do.side_effect = [{"effective_value": WRITER_ID}, {}]
    with pytest.raises(RuntimeError, match="READ_SECRET"):
        read_writer_credentials(workspace, config)
