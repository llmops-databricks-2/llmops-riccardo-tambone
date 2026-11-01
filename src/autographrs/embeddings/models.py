"""Embedding document identities and immutable publication records."""

import hashlib
import json
import math
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

TEXT_VERSION = "title-abstract-v1"


def digest(*values: str) -> str:
    """Hash length-delimited values independently from display titles and node IDs."""
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_vector(vector: list[float], dimension: int | None = None) -> None:
    """Reject malformed vectors instead of publishing unusable search data."""
    if not vector or (dimension is not None and len(vector) != dimension):
        raise ValueError("Embedding vector has an unexpected dimension")
    if not all(math.isfinite(value) for value in vector) or not any(vector):
        raise ValueError("Embedding vector must be finite and nonzero")


class EmbeddingDocument(BaseModel):
    """One document occurrence in an explicit run/snapshot, ready for search publication."""

    run_id: str
    snapshot_id: str
    document_id: str
    node_id: str
    kind: str
    title: str
    text: str
    text_variant: Literal["title_only", "title_abstract"]
    text_hash: str
    model_key: str
    embedding_key: str
    model_endpoint: str
    model_revision: str
    response_model: str = Field(min_length=1)
    dimension: int = Field(ge=1)
    embedding: list[float]


class EmbeddingSnapshot(BaseModel):
    """Only COMPLETED snapshots may be published to retrieval consumers."""

    run_id: str
    snapshot_id: str
    bibliography_snapshot_id: str
    builder_version: str
    model_key: str
    model_endpoint: str
    model_revision: str
    source_table: str
    status: Literal["RUNNING", "COMPLETED", "FAILED"] = "RUNNING"
    dimension: int = Field(default=0, ge=0)
    document_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


class SearchPublication(BaseModel):
    """Registry barrier between index updates and snapshot-scoped retrieval."""

    run_id: str
    snapshot_id: str
    publication_id: str
    backend: Literal["vector_search", "lakebase_pgvector"] = "vector_search"
    endpoint_name: str
    index_name: str
    database_name: str | None = None
    table_name: str | None = None
    status: Literal["RUNNING", "COMPLETED", "FAILED"] = "RUNNING"
    document_count: int = Field(default=0, ge=0)
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
