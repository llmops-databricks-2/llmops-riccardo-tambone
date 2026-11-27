"""UC permissions and credential lookup for the course MLflow writer."""

from urllib.parse import quote
from uuid import UUID

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import PermissionDenied
from databricks.sdk.service.catalog import PermissionsChange, Privilege

from autographrs.config import ProjectConfig

CLIENT_ID = "databricks_client_id"
CLIENT_SECRET = "databricks_client_secret"
TRACE_TABLES = ("llmops_otel_spans", "llmops_otel_logs", "llmops_otel_annotations", "llmops_otel_metrics")
TRACE_VIEWS = ("llmops_trace_metadata", "llmops_trace_unified")


def _namespace(config: ProjectConfig) -> str:
    return f"{config.catalog}.{config.db_schema}"


def grant_trace_access(workspace: WorkspaceClient, config: ProjectConfig, app_name: str, writer_client_id: str) -> None:
    """Grant schema and secret access to the App, and trace access to the course writer."""
    app_id = workspace.apps.get(name=app_name).service_principal_client_id
    if not app_id:
        raise ValueError("Deploy the App resource before configuring MLflow permissions")
    try:
        writer_id = str(UUID(writer_client_id.strip()))
    except ValueError:
        raise ValueError("The course client ID must be a UUID") from None
    if writer_id == app_id.lower():
        raise ValueError("The course writer must be distinct from the generated App principal")

    namespace = _namespace(config)
    for kind, name, privilege in (
        ("schema", namespace, "USE_SCHEMA"),
        ("secret", f"{namespace}.{CLIENT_ID}", "READ_SECRET"),
        ("secret", f"{namespace}.{CLIENT_SECRET}", "READ_SECRET"),
    ):
        try:
            # The pinned SDK does not expose READ_SECRET in its Privilege enum.
            workspace.api_client.do(
                "PATCH",
                f"/api/2.1/unity-catalog/permissions/{kind}/{quote(name, safe='')}",
                body={"changes": [{"principal": app_id, "add": [privilege]}]},
            )
        except PermissionDenied:
            raise PermissionDenied(
                f"Assistant setup cannot grant {privilege} on {name}; "
                "check the job's Run as identity or ask a course administrator to apply this grant"
            ) from None

    for name in (*TRACE_TABLES, *TRACE_VIEWS):
        privileges = [Privilege.SELECT, Privilege.MODIFY] if name in TRACE_TABLES else [Privilege.SELECT]
        try:
            workspace.grants.update(
                securable_type="table",
                full_name=f"{namespace}.{name}",
                changes=[PermissionsChange(principal=writer_id, add=privileges)],
            )
        except PermissionDenied:
            raise PermissionDenied(
                f"Assistant setup cannot grant trace access on {namespace}.{name}; "
                "check the job's Run as identity or ask a course administrator to apply this grant"
            ) from None


def read_writer_credentials(workspace: WorkspaceClient, config: ProjectConfig) -> tuple[str, str]:
    """Fetch only the course's two UC secrets using the generated App identity."""
    values = []
    for key in (CLIENT_ID, CLIENT_SECRET):
        name = f"{_namespace(config)}.{key}"
        # The pinned SDK's get_secret method does not accept include_value yet.
        payload = workspace.api_client.do(
            "GET",
            f"/api/2.1/unity-catalog/secrets/{quote(name, safe='')}",
            query={"include_value": "true"},
        )
        value = payload.get("effective_value")
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"No value returned for UC secret {name}; check the App's READ_SECRET grant")
        values.append(value.strip())
    return values[0], values[1]
