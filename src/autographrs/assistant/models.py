"""Strict request contracts and an explicit, deployment-approved collection binding."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autographrs.config import ProjectConfig
from autographrs.embeddings.models import EmbeddingSnapshot
from autographrs.graph.models import GraphBuild


class StrictModel(BaseModel):
    """Reject extra arguments rather than let the model broaden a tool's scope."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class ToolArgumentsError(ValueError):
    """Distinguish model-generated invalid tool arguments from a malformed browser request."""

    def __init__(self, tool: str, issues: list[dict]) -> None:
        self.tool = tool
        self.issues = issues
        super().__init__("The model generated invalid tool arguments")


class AssistantOperationError(RuntimeError):
    """Expose an application-owned failure code without serializing upstream messages or data."""

    def __init__(self, code: str, message: str, *, tool: str | None = None, details: dict | None = None) -> None:
        self.code = code
        self.tool = tool
        # Only sanitized metadata (codes, counts, enum values); never prompts, tool payloads or upstream text.
        self.details = details or {}
        super().__init__(message)


class RecoverableToolError(AssistantOperationError):
    """A well-formed tool call whose request cannot be satisfied; the model may correct it."""

    def __init__(self, code: str, message: str, *, tool: str | None = None, hint: str = "") -> None:
        self.hint = hint
        super().__init__(code, message, tool=tool)


class SearchArguments(StrictModel):
    """Semantic retrieval within the configured collection."""

    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=5, ge=1, le=20)


class PaperArguments(StrictModel):
    """An exact node identifier, not a title-based identity lookup."""

    node_id: str = Field(min_length=1, max_length=512)


class BibliographyArguments(PaperArguments):
    """A bounded page of acquired citation evidence."""

    limit: int = Field(default=10, ge=1, le=30)
    offset: int = Field(default=0, ge=0, le=10000)


class NeighborArguments(PaperArguments):
    """Bounded traversal of the published CITES graph."""

    direction: Literal["outgoing", "incoming", "both"] = "outgoing"
    depth: int = Field(default=1, ge=1, le=2)
    limit: int = Field(default=100, ge=1, le=100)


class AnalyticsArguments(StrictModel):
    """A natural-language question for the collection-specific Genie Agent."""

    question: str = Field(min_length=1, max_length=4000)


class ChatMessage(StrictModel):
    """Clients may supply conversation text, never system prompts or forged tool results."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(StrictModel):
    """Bound conversation size before contacting any model."""

    messages: list[ChatMessage] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_history(self) -> "ChatRequest":
        """Require a user turn and bound the whole history, not just individual messages."""
        if self.messages[-1].role != "user" or sum(len(message.content) for message in self.messages) > 16000:
            raise ValueError("End with a user message and keep conversation text within 16000 characters")
        return self


class AssistantSettings(BaseModel):
    """Non-secret runtime manifest; modified by bootstrap only, never by the agent."""

    config: ProjectConfig
    app_name: str
    app_url: str
    allowed_subjects: list[str] = Field(min_length=1)
    graph: GraphBuild
    snapshot: EmbeddingSnapshot
    fingerprint: str
    response_model: str | None
    genie_space_id: str
    view_prefix: str
    sql_auth: Literal["app", "user"] = "app"

    @model_validator(mode="after")
    def validate_binding(self) -> "AssistantSettings":
        """Reject mixed, incomplete or unapproved data lineage at startup."""
        if (
            self.graph.status != "COMPLETED"
            or self.snapshot.status != "COMPLETED"
            or self.graph.run_id != self.snapshot.run_id
            or self.graph.embedding_snapshot_id != self.snapshot.snapshot_id
            or self.graph.bibliography_snapshot_id != self.snapshot.bibliography_snapshot_id
            or not self.graph.search_publication_id
            or bool(self.snapshot.document_count) != bool(self.response_model)
        ):
            raise ValueError("Assistant settings require one coherent completed graph/embedding publication")
        return self
