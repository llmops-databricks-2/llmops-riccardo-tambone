"""Lakebase pgvector publication with Delta provenance and snapshot-scoped retrieval."""

from datetime import UTC, datetime
from uuid import uuid4

from databricks.sdk import WorkspaceClient
from loguru import logger

from autographrs.embeddings.gateway import GatewayEmbedder
from autographrs.embeddings.lakebase import LakebasePgVectorManager, lakebase_schema, snapshot_table
from autographrs.embeddings.models import SearchPublication, validate_vector
from autographrs.embeddings.storage import EmbeddingStore
from autographrs.storage.delta import _validate_run_id

PUBLICATION_DDL = (
    "run_id STRING, snapshot_id STRING, publication_id STRING, backend STRING, endpoint_name STRING, index_name STRING, "
    "database_name STRING, table_name STRING, status STRING, document_count INT, "
    "error STRING, started_at TIMESTAMP, completed_at TIMESTAMP"
)


class SearchStore(EmbeddingStore):
    """Keep immutable publication attempts and reject historical Vector Search publications."""

    @property
    def full_publications_table_name(self) -> str:
        """Return the Delta index publication registry."""
        return f"{self.config.full_schema_name}.search_publications"

    def _table(self, snapshot_id: str) -> str:
        return f"{lakebase_schema(self.config)}.{snapshot_table(self.model_key, snapshot_id)}"

    def _save_publication(self, publication: SearchPublication) -> None:
        self._merge_records(
            [publication.model_dump()],
            self.full_publications_table_name,
            PUBLICATION_DDL,
            ("run_id", "publication_id"),
        )

    def load_publication(self, run_id: str, snapshot_id: str, publication_id: str) -> SearchPublication:
        """Require a completed pgvector publication matching the configured database/schema/host."""
        _validate_run_id(publication_id)
        matches = [
            SearchPublication.model_validate(row)
            for row in self._read_rows(self.full_publications_table_name, run_id)
            if row["publication_id"] == publication_id
        ]
        if len(matches) != 1 or matches[0].status != "COMPLETED" or matches[0].snapshot_id != snapshot_id:
            raise ValueError("Search snapshot is not published")
        publication = matches[0]
        snapshot, _ = self.load_snapshot(run_id, snapshot_id)
        if (
            publication.backend != "lakebase_pgvector"
            or publication.endpoint_name != self.config.lakebase_host
            or publication.database_name != self.config.lakebase_database
            or publication.table_name != self._table(snapshot_id)
            or publication.index_name != f"{self._table(snapshot_id)}_hnsw"
            or publication.document_count != snapshot.document_count
        ):
            raise ValueError("Search publication backend/configuration/counts are inconsistent; republish on Lakebase")
        return publication

    def publish(
        self,
        run_id: str,
        snapshot_id: str,
        workspace: WorkspaceClient | None = None,
    ) -> SearchPublication:
        """Publish PostgreSQL atomically first; a failed Delta marker can be repaired without re-embedding."""
        snapshot, docs = self.load_snapshot(run_id, snapshot_id)
        if not self.config.lakebase_host:
            raise ValueError("Configure lakebase_host")
        publication = SearchPublication(
            run_id=run_id,
            snapshot_id=snapshot_id,
            publication_id=uuid4().hex,
            backend="lakebase_pgvector",
            endpoint_name=self.config.lakebase_host,
            database_name=self.config.lakebase_database,
            table_name=self._table(snapshot_id),
            index_name=f"{self._table(snapshot_id)}_hnsw",
            document_count=len(docs),
        )
        self._save_publication(publication)
        try:
            if docs:
                manager = LakebasePgVectorManager(self.config, workspace)
                manager.setup()
                manager.publish(snapshot, docs)
            else:
                logger.warning("Empty embedding snapshot {}; no vector table/index created", snapshot_id)
            publication.status = "COMPLETED"
            publication.completed_at = datetime.now(UTC)
            self._save_publication(publication)
        except Exception as exc:
            publication.status = "FAILED"
            publication.error = f"{type(exc).__name__}: {exc}"
            publication.completed_at = datetime.now(UTC)
            logger.error("Lakebase search publication failed for snapshot {}: {}", snapshot_id, publication.error)
            self._save_publication(publication)
            raise
        return publication

    def search(
        self,
        query: str,
        run_id: str,
        snapshot_id: str,
        publication_id: str,
        limit: int = 5,
        workspace: WorkspaceClient | None = None,
        embedder: GatewayEmbedder | None = None,
    ) -> list[dict[str, object]]:
        """Embed a query with the original model and retrieve neighbors in exactly one published snapshot."""
        if not query.strip() or not 1 <= limit <= 100:
            raise ValueError("Search requires nonempty text and a limit between 1 and 100")
        self.load_publication(run_id, snapshot_id, publication_id)
        snapshot, docs = self.load_snapshot(run_id, snapshot_id)
        if not docs:
            return []
        vectors, response_model = (embedder or GatewayEmbedder(self.config)).embed([query])
        validate_vector(vectors[0], snapshot.dimension)
        if response_model != docs[0].response_model:
            raise ValueError("Query embedding model differs from stored documents; rebuild with a new revision")
        results = LakebasePgVectorManager(self.config, workspace).search(snapshot, docs, vectors[0], limit)
        expected = {doc.document_id for doc in docs}
        if any(row["document_id"] not in expected for row in results):
            raise ValueError("Lakebase returned an unknown snapshot document")
        return results
