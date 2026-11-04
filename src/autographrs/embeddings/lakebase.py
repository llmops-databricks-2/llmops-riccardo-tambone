"""OAuth Lakebase connections and transactionally published, snapshot-local pgvector indexes."""

import json
import re
import time
from collections.abc import Generator
from contextlib import contextmanager

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import EndpointType
from loguru import logger
from psycopg import sql
from psycopg.rows import dict_row

from autographrs.config import ProjectConfig
from autographrs.embeddings.models import EmbeddingDocument, EmbeddingSnapshot, digest, validate_vector


def lakebase_schema(config: ProjectConfig) -> str:
    """Require an explicit PostgreSQL schema to keep environments isolated."""
    if not config.lakebase_schema or not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", config.lakebase_schema):
        raise ValueError("Configure a valid lakebase_schema for this environment")
    return config.lakebase_schema


def snapshot_table(model_key: str, snapshot_id: str) -> str:
    """Give each immutable snapshot its own HNSW index, avoiding ANN post-filter underfilling."""
    if not re.fullmatch(r"[a-f0-9]{64}", model_key) or not re.fullmatch(r"[a-f0-9]{32}", snapshot_id):
        raise ValueError("Invalid model key or embedding snapshot ID")
    return f"e_{model_key[:16]}_{snapshot_id}"


def document_fingerprint(docs: list[EmbeddingDocument]) -> str:
    """Include the full payload, not just identifiers, in the PostgreSQL publication barrier."""
    return digest(*(doc.model_dump_json() for doc in sorted(docs, key=lambda doc: doc.document_id)))


class LakebasePgVectorManager:
    """Manage pgvector storage; never create projects, branches, computes, or AI Search resources."""

    def __init__(self, config: ProjectConfig, workspace: WorkspaceClient | None = None) -> None:
        self.config = config
        self._workspace = workspace

    @property
    def workspace(self) -> WorkspaceClient:
        """Resolve credentials only when connecting, not during configuration/read-only Delta operations."""
        if self._workspace is None:
            self._workspace = WorkspaceClient()
        return self._workspace

    def resolve_endpoint(self) -> str:
        """Discover the existing read/write endpoint by its exact configured host, never by branch display name."""
        if not self.config.lakebase_instance or not self.config.lakebase_host:
            raise ValueError("Configure lakebase_instance and lakebase_host")
        parent = self.config.lakebase_instance
        if not parent.startswith("projects/"):
            parent = f"projects/{parent}"
        if self.config.lakebase_endpoint:
            if not self.config.lakebase_endpoint.startswith(parent + "/branches/"):
                raise ValueError("lakebase_endpoint must belong to the configured Lakebase project")
            endpoints = [self.workspace.postgres.get_endpoint(self.config.lakebase_endpoint)]
        else:
            endpoints = [
                endpoint
                for branch in self.workspace.postgres.list_branches(parent=parent)
                if branch.name
                for endpoint in self.workspace.postgres.list_endpoints(parent=branch.name)
            ]
        matches = [
            endpoint.name
            for endpoint in endpoints
            if endpoint.name
            and endpoint.status
            and endpoint.status.hosts
            and endpoint.status.hosts.host == self.config.lakebase_host
            and endpoint.status.endpoint_type == EndpointType.ENDPOINT_TYPE_READ_WRITE
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one read/write Lakebase endpoint matching lakebase_host; set lakebase_endpoint explicitly"
            )
        return matches[0]

    @contextmanager
    def connect(self) -> Generator[psycopg.Connection]:
        """Refresh short-lived OAuth credentials on each connection; bounded retry for transport startup only."""
        endpoint = self.resolve_endpoint()
        user = self.config.lakebase_user or self.workspace.current_user.me().user_name
        if not user:
            raise ValueError("Cannot determine the OAuth PostgreSQL role; configure lakebase_user")
        deadline = time.monotonic() + min(self.config.search_timeout_seconds, 60)
        connection = None
        for attempt in range(3):
            credential = self.workspace.postgres.generate_database_credential(endpoint=endpoint)
            if not credential.token:
                raise ValueError("Lakebase credential API returned no token")
            try:
                connection = psycopg.connect(
                    host=self.config.lakebase_host,
                    dbname=self.config.lakebase_database,
                    user=user,
                    password=credential.token,
                    sslmode="require",
                    connect_timeout=10,
                    row_factory=dict_row,
                    options=f"-c statement_timeout={self.config.search_timeout_seconds * 1000} -c lock_timeout=30000",
                )
                break
            except psycopg.OperationalError as exc:
                # Authentication/authorization failures need a real fix, not repeated logins.
                if not exc.sqlstate or (not exc.sqlstate.startswith("08") and exc.sqlstate != "57P03"):
                    raise
                if attempt == 2 or time.monotonic() >= deadline:
                    raise
                logger.warning("Lakebase connection unavailable; retrying startup ({}/3)", attempt + 1)
                time.sleep(2 * (attempt + 1))
        if connection is None:
            raise RuntimeError("Lakebase connection was not established")
        with connection:
            yield connection

    @property
    def registry(self) -> sql.Identifier:
        """Return the environment-local PostgreSQL publication registry."""
        return sql.Identifier(lakebase_schema(self.config), "embedding_publications")

    def setup(self) -> str:
        """Install standard pgvector if available and create the environment schema/registry."""
        schema = lakebase_schema(self.config)
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", ("autographrs-pgvector-setup",))
            cur.execute("SELECT name FROM pg_available_extensions WHERE name = 'vector'")
            if cur.fetchone() is None:
                raise ValueError("Standard pgvector extension 'vector' is not available in this Lakebase database")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
            cur.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} (run_id TEXT NOT NULL, snapshot_id TEXT NOT NULL, "
                    "model_key TEXT NOT NULL, table_name TEXT NOT NULL, dimension INTEGER NOT NULL, "
                    "document_count INTEGER NOT NULL, fingerprint TEXT NOT NULL, "
                    "PRIMARY KEY (run_id, snapshot_id))"
                ).format(self.registry)
            )
        logger.info("Lakebase schema {} ready with standard pgvector; no Lakebase Search enabled", schema)
        return schema

    @staticmethod
    def _vector_schema(cur: psycopg.Cursor) -> str:
        cur.execute(
            "SELECT n.nspname AS schema_name FROM pg_extension e JOIN pg_namespace n "
            "ON n.oid = e.extnamespace WHERE e.extname = 'vector'"
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError("pgvector is not installed; publish a snapshot to prepare Lakebase before searching")
        return row["schema_name"]

    def publish(self, snapshot: EmbeddingSnapshot, docs: list[EmbeddingDocument]) -> str:
        """Atomically write/verify an immutable snapshot, its HNSW index and its PostgreSQL marker."""
        if len(docs) != snapshot.document_count or not 1 <= snapshot.dimension <= 2000:
            raise ValueError("pgvector HNSW vector indexing requires 1-2000 dimensions and consistent document counts")
        if snapshot.status != "COMPLETED" or len({doc.document_id for doc in docs}) != len(docs):
            raise ValueError("Publish requires a completed snapshot with unique documents")
        for doc in docs:
            validate_vector(doc.embedding, snapshot.dimension)
            if (
                doc.run_id != snapshot.run_id
                or doc.snapshot_id != snapshot.snapshot_id
                or doc.model_key != snapshot.model_key
                or doc.dimension != snapshot.dimension
            ):
                raise ValueError("Embedding document does not belong to the requested snapshot/model")
        schema = lakebase_schema(self.config)
        table_name = snapshot_table(snapshot.model_key, snapshot.snapshot_id)
        table = sql.Identifier(schema, table_name)
        fingerprint = document_fingerprint(docs)
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"{schema}.{table_name}",))
            vector_schema = self._vector_schema(cur)
            vector_type = sql.SQL("{}({})").format(
                sql.Identifier(vector_schema, "vector"), sql.Literal(snapshot.dimension)
            )
            cur.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} (document_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, "
                    "snapshot_id TEXT NOT NULL, payload JSONB NOT NULL, embedding {} NOT NULL)"
                ).format(table, vector_type)
            )
            cur.execute(
                "SELECT a.atttypmod AS dimension FROM pg_attribute a "
                "JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace "
                "JOIN pg_type t ON t.oid = a.atttypid JOIN pg_namespace tn ON tn.oid = t.typnamespace "
                "WHERE n.nspname = %s AND c.relname = %s AND a.attname = 'embedding' "
                "AND NOT a.attisdropped AND t.typname = 'vector' AND tn.nspname = %s",
                (schema, table_name, vector_schema),
            )
            dimension = cur.fetchone()
            if dimension is None or dimension["dimension"] != snapshot.dimension:
                raise ValueError("Existing PostgreSQL vector column is incompatible")
            insert = sql.SQL(
                "INSERT INTO {} (document_id, run_id, snapshot_id, payload, embedding) "
                "VALUES (%s, %s, %s, %s::jsonb, %s::{}) ON CONFLICT (document_id) DO NOTHING"
            ).format(table, sql.Identifier(vector_schema, "vector"))
            cur.executemany(
                insert,
                [
                    (doc.document_id, doc.run_id, doc.snapshot_id, doc.model_dump_json(), json.dumps(doc.embedding))
                    for doc in docs
                ],
            )
            cur.execute(
                sql.SQL(
                    "SELECT document_id, run_id, snapshot_id, payload, embedding::text AS vector_text FROM {}"
                ).format(table)
            )
            expected = {doc.document_id: doc for doc in docs}
            rows = cur.fetchall()
            if len(rows) != len(expected):
                raise ValueError("PostgreSQL snapshot document count differs from Delta")
            for row in rows:
                doc = expected.get(row["document_id"])
                if (
                    doc is None
                    or row["run_id"] != snapshot.run_id
                    or row["snapshot_id"] != snapshot.snapshot_id
                    or EmbeddingDocument.model_validate(row["payload"]) != doc
                    or self._float32(json.loads(row["vector_text"])) != self._float32(doc.embedding)
                ):
                    raise ValueError("PostgreSQL snapshot data differs from Delta")
            index_name = f"{table_name}_hnsw"
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding {})").format(
                    sql.Identifier(index_name), table, sql.Identifier(vector_schema, "vector_cosine_ops")
                )
            )
            cur.execute(
                "SELECT am.amname AS method, op.opcname AS opclass, i.indisvalid AS valid "
                "FROM pg_index i JOIN pg_class idx ON idx.oid = i.indexrelid "
                "JOIN pg_namespace n ON n.oid = idx.relnamespace JOIN pg_am am ON am.oid = idx.relam "
                "JOIN pg_opclass op ON op.oid = i.indclass[0] "
                "WHERE n.nspname = %s AND idx.relname = %s AND i.indrelid = %s::regclass",
                (schema, index_name, f'"{schema}"."{table_name}"'),
            )
            index = cur.fetchone()
            if not index or index != {"method": "hnsw", "opclass": "vector_cosine_ops", "valid": True}:
                raise ValueError("Existing PostgreSQL HNSW index is incompatible or invalid")
            marker = (
                snapshot.run_id,
                snapshot.snapshot_id,
                snapshot.model_key,
                table_name,
                snapshot.dimension,
                len(docs),
                fingerprint,
            )
            cur.execute(
                sql.SQL(
                    "INSERT INTO {} (run_id, snapshot_id, model_key, table_name, dimension, document_count, fingerprint) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (run_id, snapshot_id) DO NOTHING"
                ).format(self.registry),
                marker,
            )
            self._check_marker(cur, snapshot, docs)
            if not self._query(cur, snapshot, docs[0].embedding, 1):
                raise ValueError("Published PostgreSQL snapshot returned no search results")
        return f"{schema}.{table_name}"

    @staticmethod
    def _float32(vector: list[float]) -> list[float]:
        from array import array

        return array("f", vector).tolist()

    def _check_marker(self, cur: psycopg.Cursor, snapshot: EmbeddingSnapshot, docs: list[EmbeddingDocument]) -> None:
        self._check_fingerprint(cur, snapshot, document_fingerprint(docs))

    def _check_fingerprint(self, cur: psycopg.Cursor, snapshot: EmbeddingSnapshot, fingerprint: str) -> None:
        cur.execute(
            sql.SQL(
                "SELECT model_key, table_name, dimension, document_count, fingerprint FROM {} "
                "WHERE run_id = %s AND snapshot_id = %s"
            ).format(self.registry),
            (snapshot.run_id, snapshot.snapshot_id),
        )
        expected = {
            "model_key": snapshot.model_key,
            "table_name": snapshot_table(snapshot.model_key, snapshot.snapshot_id),
            "dimension": snapshot.dimension,
            "document_count": snapshot.document_count,
            "fingerprint": fingerprint,
        }
        if cur.fetchone() != expected:
            raise ValueError("PostgreSQL publication marker is missing or differs from the Delta snapshot")

    def _query(
        self,
        cur: psycopg.Cursor,
        snapshot: EmbeddingSnapshot,
        vector: list[float],
        limit: int,
    ) -> list[dict[str, object]]:
        validate_vector(vector, snapshot.dimension)
        vector_schema = self._vector_schema(cur)
        # A table contains only one immutable snapshot: HNSW has no selective run filter to underfill results.
        cur.execute(sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(max(100, limit))))
        distance = sql.SQL("embedding OPERATOR({}.<=>) %s::{}").format(
            sql.Identifier(vector_schema),
            sql.Identifier(vector_schema, "vector"),
        )
        cur.execute(
            sql.SQL(
                "SELECT payload, 1 - ({distance}) AS cosine_similarity FROM {table} ORDER BY {distance} LIMIT %s"
            ).format(
                distance=distance,
                table=sql.Identifier(
                    lakebase_schema(self.config), snapshot_table(snapshot.model_key, snapshot.snapshot_id)
                ),
            ),
            (json.dumps(vector), json.dumps(vector), limit),
        )
        results = []
        for row in cur.fetchall():
            doc = EmbeddingDocument.model_validate(row["payload"])
            if (
                doc.run_id != snapshot.run_id
                or doc.snapshot_id != snapshot.snapshot_id
                or doc.model_key != snapshot.model_key
            ):
                raise ValueError("PostgreSQL returned a document outside the requested snapshot")
            results.append(
                {
                    **doc.model_dump(exclude={"embedding"}),
                    "cosine_similarity": row["cosine_similarity"],
                }
            )
        expected_count = min(limit, snapshot.document_count)
        if len(results) != expected_count:
            raise ValueError(f"pgvector returned {len(results)} results; expected {expected_count} for this snapshot")
        return results

    def search(
        self,
        snapshot: EmbeddingSnapshot,
        docs: list[EmbeddingDocument],
        vector: list[float],
        limit: int,
    ) -> list[dict[str, object]]:
        """Require a committed PostgreSQL marker before retrieving snapshot-local nearest neighbors."""
        if not 1 <= limit <= 100:
            raise ValueError("Search limit must be between 1 and 100")
        with self.connect() as conn, conn.cursor() as cur:
            self._check_marker(cur, snapshot, docs)
            return self._query(cur, snapshot, vector, limit)

    def search_published(
        self,
        snapshot: EmbeddingSnapshot,
        fingerprint: str,
        vector: list[float],
        limit: int,
    ) -> list[dict[str, object]]:
        """Query a bootstrap-verified publication without loading a Delta snapshot through Spark."""
        if not 1 <= limit <= 100:
            raise ValueError("Search limit must be between 1 and 100")
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            self._check_fingerprint(cur, snapshot, fingerprint)
            return self._query(cur, snapshot, vector, limit)
