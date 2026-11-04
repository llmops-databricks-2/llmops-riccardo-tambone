"""Request-local delegated SQL, Genie and Gateway identity; Lakebase uses App OAuth."""

from collections.abc import Mapping
from contextvars import ContextVar

from databricks.sdk import WorkspaceClient

USER_WORKSPACE: ContextVar[WorkspaceClient] = ContextVar("assistant_user_workspace")


def is_allowed_owner(headers: Mapping[str, str], allowed_subjects: list[str]) -> bool:
    """Match owner identifiers supplied only by the authenticated Databricks Apps proxy."""
    return bool(headers.get("x-forwarded-user")) and any(
        headers.get(header) in allowed_subjects
        for header in ("x-forwarded-user", "x-forwarded-email", "x-forwarded-preferred-username")
    )


def user_workspace() -> WorkspaceClient:
    """Require a proxy-forwarded user credential in the current request, never fall back to App credentials."""
    try:
        return USER_WORKSPACE.get()
    except LookupError:
        raise PermissionError(
            "User OAuth is required for SQL, Genie and Gateway; open the App and grant consent"
        ) from None
