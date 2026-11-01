"""Persist embedding snapshots before publishing them to an external search index."""

from array import array
from datetime import UTC, datetime
from uuid import uuid4

from loguru import logger

from autographrs.embeddings.gateway import GatewayEmbedder
from autographrs.embeddings.models import TEXT_VERSION, EmbeddingDocument, EmbeddingSnapshot, digest, validate_vector
from autographrs.enrichment.bibliography import BibliographyStore
from autographrs.graph.models import GRAPH_BUILDER_VERSION
from autographrs.storage.delta import DeltaPaperStore, _validate_run_id

DOCUMENT_DDL = (
    "run_id STRING, snapshot_id STRING, document_id STRING, node_id STRING, kind STRING, title STRING, "
    "text STRING, text_variant STRING, text_hash STRING, model_key STRING, embedding_key STRING, "
    "model_endpoint STRING, model_revision STRING, response_model STRING, dimension INT, embedding ARRAY<FLOAT>"
)
SNAPSHOT_DDL = (
    "run_id STRING, snapshot_id STRING, bibliography_snapshot_id STRING, builder_version STRING, model_key STRING, "
    "model_endpoint STRING, model_revision STRING, source_table STRING, status STRING, dimension INT, "
    "document_count INT, skipped_count INT, warnings ARRAY<STRING>, error STRING, "
    "started_at TIMESTAMP, completed_at TIMESTAMP"
)


class EmbeddingStore(DeltaPaperStore):
    """Reuse unchanged vectors within a run, including successfully persisted failed-attempt batches."""

    @property
    def model_key(self) -> str:
        """Separate incompatible model routing revisions and text preparation rules."""
        return digest(self.config.embedding_service, self.config.embedding_revision, TEXT_VERSION)

    @property
    def full_documents_table_name(self) -> str:
        """Use a separate source table for each embedding model configuration."""
        return f"{self.config.full_schema_name}.paper_embeddings_{self.model_key[:16]}"

    @property
    def full_snapshots_table_name(self) -> str:
        """Return the embedding publication registry."""
        return f"{self.config.full_schema_name}.embedding_snapshots"

    def _save_snapshot(self, snapshot: EmbeddingSnapshot) -> None:
        self._merge_records(
            [snapshot.model_dump()], self.full_snapshots_table_name, SNAPSHOT_DDL, ("run_id", "snapshot_id")
        )

    def _documents(self, run_id: str) -> list[EmbeddingDocument]:
        return [
            EmbeddingDocument.model_validate(row) for row in self._read_rows(self.full_documents_table_name, run_id)
        ]

    def load_snapshot(self, run_id: str, snapshot_id: str) -> tuple[EmbeddingSnapshot, list[EmbeddingDocument]]:
        """Validate exact snapshot identity, contents and model before any downstream consumption."""
        _validate_run_id(snapshot_id)
        matches = [
            EmbeddingSnapshot.model_validate(row)
            for row in self._read_rows(self.full_snapshots_table_name, run_id)
            if row["snapshot_id"] == snapshot_id
        ]
        if len(matches) != 1 or matches[0].status != "COMPLETED":
            raise ValueError("Embedding snapshot is not published")
        snapshot = matches[0]
        if (
            snapshot.model_key != self.model_key
            or snapshot.builder_version != GRAPH_BUILDER_VERSION
            or snapshot.source_table != self.full_documents_table_name
            or snapshot.model_endpoint != self.config.embedding_service
            or snapshot.model_revision != self.config.embedding_revision
        ):
            raise ValueError("Embedding snapshot configuration is incompatible")
        docs = [doc for doc in self._documents(run_id) if doc.snapshot_id == snapshot_id]
        if len(docs) != snapshot.document_count or len({doc.node_id for doc in docs}) != len(docs):
            raise ValueError("Embedding snapshot counts or keys are inconsistent")
        if bool(docs) != bool(snapshot.dimension):
            raise ValueError("Embedding snapshot dimension is inconsistent with its document count")
        for doc in docs:
            self._validate_document(doc, snapshot.dimension)
        if len({doc.response_model for doc in docs}) > 1:
            raise ValueError("Embedding snapshot mixes response models")
        return snapshot, sorted(docs, key=lambda doc: doc.node_id)

    def _validate_document(self, doc: EmbeddingDocument, dimension: int | None = None) -> None:
        validate_vector(doc.embedding, dimension or doc.dimension)
        if (
            doc.model_key != self.model_key
            or doc.model_endpoint != self.config.embedding_service
            or doc.model_revision != self.config.embedding_revision
            or doc.dimension != len(doc.embedding)
            or doc.text_hash != digest(doc.text)
            or doc.embedding_key != digest(self.model_key, doc.text_hash)
            or doc.document_id != digest(doc.run_id, doc.snapshot_id, doc.node_id)
        ):
            raise ValueError("Stored embedding document integrity check failed")

    def prepare(
        self, run_id: str, bibliography_snapshot_id: str, embedder: GatewayEmbedder | None = None
    ) -> EmbeddingSnapshot:
        """Embed all usable titles/abstracts without modifying bibliography or graph snapshots."""
        graph = BibliographyStore(self.config, spark=self.spark).load_graph_input(run_id, bibliography_snapshot_id)
        snapshot = EmbeddingSnapshot(
            run_id=run_id,
            snapshot_id=uuid4().hex,
            bibliography_snapshot_id=bibliography_snapshot_id,
            builder_version=GRAPH_BUILDER_VERSION,
            model_key=self.model_key,
            model_endpoint=self.config.embedding_service,
            model_revision=self.config.embedding_revision,
            source_table=self.full_documents_table_name,
            warnings=list(graph.warnings),
        )
        self._save_snapshot(snapshot)
        try:
            self._ensure_table(self.full_documents_table_name, DOCUMENT_DDL)
            cache: dict[str, EmbeddingDocument] = {}
            for doc in self._documents(run_id):
                self._validate_document(doc)
                previous = cache.get(doc.embedding_key)
                if previous and (previous.embedding != doc.embedding or previous.response_model != doc.response_model):
                    raise ValueError("Conflicting cached embeddings; change embedding_revision after model changes")
                cache[doc.embedding_key] = doc
            requests = []
            for node in graph.nodes:
                if node.text_variant is None:
                    snapshot.skipped_count += 1
                    snapshot.warnings.append(f"{node.node_id}: skipped embedding without a usable title")
                    continue
                title = node.title.strip()
                abstract = (node.abstract or "").strip()
                text = f"{title}\n\n{abstract}" if abstract else title
                if not title or len(text) > self.config.embedding_max_chars:
                    raise ValueError(f"{node.node_id}: text empty or exceeds embedding_max_chars; no truncation")
                text_hash = digest(text)
                requests.append((node, text, text_hash, digest(self.model_key, text_hash)))
            pending = {key: text for _, text, _, key in requests if key not in cache}
            engine = embedder
            expected_documents: list[EmbeddingDocument] = []
            for offset in range(0, len(requests), self.config.embedding_batch_size):
                batch = requests[offset : offset + self.config.embedding_batch_size]
                missing = list(dict.fromkeys(key for _, _, _, key in batch if key in pending))
                if missing:
                    engine = engine or GatewayEmbedder(self.config)
                    vectors, response_model = engine.embed([pending[key] for key in missing])
                    if len(vectors) != len(missing):
                        raise ValueError("Embedder returned an incomplete batch")
                    generated = dict(zip(missing, vectors, strict=True))
                else:
                    generated = {}
                    response_model = ""
                records = []
                for node, text, text_hash, key in batch:
                    old = cache.get(key)
                    vector = old.embedding if old else array("f", generated[key]).tolist()
                    doc = EmbeddingDocument(
                        run_id=run_id,
                        snapshot_id=snapshot.snapshot_id,
                        document_id=digest(run_id, snapshot.snapshot_id, node.node_id),
                        node_id=node.node_id,
                        kind=node.kind,
                        title=node.title.strip(),
                        text=text,
                        text_variant="title_abstract" if (node.abstract or "").strip() else "title_only",
                        text_hash=text_hash,
                        model_key=self.model_key,
                        embedding_key=key,
                        model_endpoint=self.config.embedding_service,
                        model_revision=self.config.embedding_revision,
                        response_model=old.response_model if old else response_model,
                        dimension=len(vector),
                        embedding=vector,
                    )
                    self._validate_document(doc)
                    expected_documents.append(doc)
                    records.append(doc.model_dump())
                    cache[key] = doc
                    pending.pop(key, None)
                used = list(cache.values())
                if len({(doc.dimension, doc.response_model) for doc in used}) > 1:
                    raise ValueError("Gateway model/dimension changed; bump embedding_revision and rebuild")
                self._merge_records(
                    records, self.full_documents_table_name, DOCUMENT_DDL, ("run_id", "snapshot_id", "node_id")
                )
            stored = [doc for doc in self._documents(run_id) if doc.snapshot_id == snapshot.snapshot_id]
            if {doc.node_id for doc in stored} != {node.node_id for node, *_ in requests} or len(stored) != len(
                requests
            ):
                raise ValueError("Persisted embedding document coverage is inconsistent")
            for doc in stored:
                self._validate_document(doc)
            if sorted(stored, key=lambda doc: doc.node_id) != sorted(expected_documents, key=lambda doc: doc.node_id):
                raise ValueError("Persisted embedding data differs from generated documents")
            snapshot.document_count = len(stored)
            snapshot.dimension = stored[0].dimension if stored else 0
            snapshot.status = "COMPLETED"
            snapshot.completed_at = datetime.now(UTC)
            for warning in snapshot.warnings:
                logger.warning("Embedding {}: {}", snapshot.snapshot_id, warning)
            self._save_snapshot(snapshot)
        except Exception as exc:
            snapshot.status = "FAILED"
            snapshot.error = f"{type(exc).__name__}: {exc}"
            snapshot.completed_at = datetime.now(UTC)
            logger.exception("Embedding preparation failed for run {}", run_id)
            self._save_snapshot(snapshot)
            raise
        return snapshot
