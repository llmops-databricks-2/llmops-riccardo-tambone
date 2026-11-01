"""Verify OAuth, safe SQL, transactional barriers and snapshot-local HNSW retrieval without a live database."""

import json
from copy import deepcopy
from unittest.mock import MagicMock, patch

import psycopg
import pytest
from databricks.sdk.errors import PermissionDenied
from databricks.sdk.service.postgres import Endpoint, EndpointHosts, EndpointStatus, EndpointType
from test_embeddings import _config, _engine
from test_embeddings import prepared_input as prepared_input

from autographrs.embeddings.lakebase import (
    LakebasePgVectorManager,
    document_fingerprint,
    lakebase_schema,
    snapshot_table,
)
from autographrs.embeddings.models import EmbeddingDocument, EmbeddingSnapshot
from autographrs.embeddings.search import SearchStore

RESOURCE = "projects/lb-llmops-riccardo-tambone/branches/br-technical-id/endpoints/ep-technical-id"


def _workspace() -> MagicMock:
    workspace = MagicMock()
    branch = MagicMock()
    branch.name = RESOURCE.rsplit("/endpoints/", 1)[0]
    workspace.postgres.list_branches.return_value = [branch]
    workspace.postgres.list_endpoints.return_value = [
        Endpoint(
            name=RESOURCE,
            status=EndpointStatus(
                hosts=EndpointHosts(host=_config().lakebase_host),
                endpoint_type=EndpointType.ENDPOINT_TYPE_READ_WRITE,
            ),
        ),
    ]
    workspace.postgres.get_endpoint.return_value = workspace.postgres.list_endpoints.return_value[0]
    workspace.current_user.me.return_value.user_name = "job@example.com"
    workspace.postgres.generate_database_credential.return_value.token = "not-a-real-credential"
    return workspace


def _connection() -> tuple[MagicMock, MagicMock]:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    return conn, cur


def _sql(cur: MagicMock) -> list[str]:
    return [
        call.args[0] if isinstance(call.args[0], str) else call.args[0].as_string(None)
        for call in cur.execute.call_args_list
    ]


def test_endpoint_discovery_uses_exact_host_not_display_name() -> None:
    """Resolve technical resources by host and reject ambiguous or read-only endpoints."""
    workspace = _workspace()
    manager = LakebasePgVectorManager(_config(), workspace)
    assert manager.resolve_endpoint() == RESOURCE
    workspace.postgres.list_branches.assert_called_once_with(parent="projects/lb-llmops-riccardo-tambone")
    workspace.postgres.list_endpoints.assert_called_once_with(parent=RESOURCE.rsplit("/endpoints/", 1)[0])
    explicit = LakebasePgVectorManager(_config(lakebase_endpoint=RESOURCE), workspace)
    assert explicit.resolve_endpoint() == RESOURCE
    workspace.postgres.get_endpoint.assert_called_once_with(RESOURCE)
    workspace.postgres.list_endpoints.return_value *= 2
    with pytest.raises(ValueError, match="Expected one"):
        manager.resolve_endpoint()
    workspace.postgres.list_endpoints.return_value[0].status.endpoint_type = EndpointType.ENDPOINT_TYPE_READ_ONLY
    workspace.postgres.list_endpoints.return_value = workspace.postgres.list_endpoints.return_value[:1]
    with pytest.raises(ValueError, match="read/write"):
        manager.resolve_endpoint()
    with pytest.raises(ValueError, match="belong"):
        LakebasePgVectorManager(
            _config(lakebase_endpoint="projects/other/branches/x/endpoints/y"), workspace
        ).resolve_endpoint()


def test_connections_refresh_oauth_use_tls_and_close() -> None:
    """New connections acquire fresh credentials and have bounded lifetimes/timeouts."""
    workspace = _workspace()
    manager = LakebasePgVectorManager(_config(), workspace)
    conn, _ = _connection()
    with patch("psycopg.connect", return_value=conn) as connect:
        for _ in range(2):
            with manager.connect() as returned:
                assert returned is conn
    assert workspace.postgres.generate_database_credential.call_count == 2
    workspace.postgres.generate_database_credential.assert_called_with(endpoint=RESOURCE)
    assert connect.call_args.kwargs["sslmode"] == "require"
    assert connect.call_args.kwargs["user"] == "job@example.com"
    assert connect.call_args.kwargs["connect_timeout"] == 10
    assert "statement_timeout=" in connect.call_args.kwargs["options"]
    assert conn.__exit__.call_count == 2


def test_cold_start_retries_bounded_but_authorization_does_not_retry() -> None:
    """Only recognized transient SQL states are retried; unknown failures propagate."""
    workspace = _workspace()
    manager = LakebasePgVectorManager(_config(), workspace)
    conn, _ = _connection()
    with patch("psycopg.connect", side_effect=[psycopg.errors.CannotConnectNow("waking"), conn]) as connect:
        with patch("autographrs.embeddings.lakebase.time.sleep") as sleep, manager.connect():
            pass
        assert connect.call_count == 2
        sleep.assert_called_once_with(2)
    with patch("psycopg.connect", side_effect=psycopg.errors.InvalidPassword("role missing")) as connect:
        with pytest.raises(psycopg.errors.InvalidPassword), manager.connect():
            pass
        assert connect.call_count == 1
    workspace.postgres.generate_database_credential.side_effect = PermissionDenied("cannot authenticate")
    with patch("psycopg.connect") as connect, pytest.raises(PermissionDenied), manager.connect():
        pass
    connect.assert_not_called()
    workspace.postgres.generate_database_credential.side_effect = None
    with patch("psycopg.connect", side_effect=psycopg.OperationalError("unknown failure")) as connect:
        with pytest.raises(psycopg.OperationalError), manager.connect():
            pass
        assert connect.call_count == 1
    with patch("psycopg.connect", side_effect=psycopg.errors.CannotConnectNow("starting")) as connect:
        with (
            patch("autographrs.embeddings.lakebase.time.sleep"),
            pytest.raises(psycopg.OperationalError),
            manager.connect(),
        ):
            pass
        assert connect.call_count == 3


def test_setup_uses_standard_vector_only_and_rolls_back_on_unavailable_extension() -> None:
    """Never enable Lakebase Search or continue when standard vector is unavailable."""
    conn, cur = _connection()
    manager = LakebasePgVectorManager(_config(), _workspace())
    with patch("psycopg.connect", return_value=conn):
        cur.fetchone.return_value = {"name": "vector"}
        assert manager.setup() == "autographrs_dev"
        statements = _sql(cur)
        assert "CREATE EXTENSION IF NOT EXISTS vector" in statements
        assert any('"autographrs_dev"."embedding_publications"' in statement for statement in statements)
        assert not any("lakebase_vector" in statement or "lakebase_ann" in statement for statement in statements)
        cur.reset_mock()
        cur.fetchone.return_value = None
        with pytest.raises(ValueError, match="not available"):
            manager.setup()
    assert not any("CREATE EXTENSION" in statement for statement in _sql(cur))
    assert conn.__exit__.call_args.args[0] is ValueError


def _published_cursor(cur: MagicMock, snapshot: EmbeddingSnapshot, docs: list[EmbeddingDocument]) -> None:
    cur.fetchone.side_effect = [
        {"schema_name": "public"},
        {"dimension": snapshot.dimension},
        {"method": "hnsw", "opclass": "vector_cosine_ops", "valid": True},
        {
            "model_key": snapshot.model_key,
            "table_name": snapshot_table(snapshot.model_key, snapshot.snapshot_id),
            "dimension": snapshot.dimension,
            "document_count": len(docs),
            "fingerprint": document_fingerprint(docs),
        },
        {"schema_name": "public"},
    ]
    cur.fetchall.side_effect = [
        [
            {
                "document_id": doc.document_id,
                "run_id": doc.run_id,
                "snapshot_id": doc.snapshot_id,
                "payload": doc.model_dump(),
                "vector_text": json.dumps(doc.embedding),
            }
            for doc in docs
        ],
        [{"payload": docs[0].model_dump(), "cosine_similarity": 1.0}],
    ]


def test_publish_validates_every_document_and_commits_marker_with_hnsw(
    prepared_input: tuple[SearchStore, str, str],
) -> None:
    """Verify full snapshot contents and the index before committing, including repeated publications."""
    store, run_id, bib = prepared_input
    snapshot = store.prepare(run_id, bib, _engine(store.config))
    _, docs = store.load_snapshot(run_id, snapshot.snapshot_id)
    conn, cur = _connection()
    manager = LakebasePgVectorManager(store.config, _workspace())
    with patch("psycopg.connect", return_value=conn):
        for _ in range(2):
            _published_cursor(cur, snapshot, docs)
            assert manager.publish(snapshot, docs).startswith("autographrs_dev.e_")
    statements = _sql(cur)
    assert sum("USING hnsw" in statement for statement in statements) == 2
    assert any('"public"."vector_cosine_ops"' in statement for statement in statements)
    assert any('"public"."vector"(3)' in statement for statement in statements)
    assert any('OPERATOR("public".<=>)' in statement for statement in statements)
    assert all(len(snapshot_table(snapshot.model_key, snapshot.snapshot_id) + suffix) <= 63 for suffix in ("", "_hnsw"))
    assert cur.executemany.call_count == 2
    assert "ON CONFLICT (document_id) DO NOTHING" in cur.executemany.call_args.args[0].as_string(None)
    assert conn.__exit__.call_args.args == (None, None, None)


@pytest.mark.parametrize("corruption", ["payload", "vector", "extra", "index", "dimension", "marker"])
def test_corrupt_existing_data_or_index_is_not_overwritten_or_published(
    prepared_input: tuple[SearchStore, str, str],
    corruption: str,
) -> None:
    """Mismatch aborts the transaction instead of repairing immutable data by overwrite."""
    store, run_id, bib = prepared_input
    snapshot = store.prepare(run_id, bib, _engine(store.config))
    _, docs = store.load_snapshot(run_id, snapshot.snapshot_id)
    conn, cur = _connection()
    _published_cursor(cur, snapshot, docs)
    ones = list(cur.fetchone.side_effect)
    alls = list(cur.fetchall.side_effect)
    if corruption == "payload":
        alls[0][0]["payload"]["title"] = "corruption"
    elif corruption == "vector":
        alls[0][0]["vector_text"] = "[9,9,9]"
    elif corruption == "extra":
        alls[0].append(deepcopy(alls[0][0]))
    elif corruption == "index":
        ones[2]["valid"] = False
    elif corruption == "dimension":
        ones[1]["dimension"] += 1
    else:
        ones[3]["fingerprint"] = "corruption"
    cur.fetchone.side_effect, cur.fetchall.side_effect = ones, alls
    with patch("psycopg.connect", return_value=conn), pytest.raises(ValueError):
        LakebasePgVectorManager(store.config, _workspace()).publish(snapshot, docs)
    assert conn.__exit__.call_args.args[0] is ValueError
    assert not any(statement.startswith("UPDATE") or statement.startswith("DELETE") for statement in _sql(cur))


def test_query_uses_snapshot_table_and_validates_marker(
    prepared_input: tuple[SearchStore, str, str],
) -> None:
    """Check committed provenance before querying the exact snapshot table."""
    store, run_id, bib = prepared_input
    snapshot = store.prepare(run_id, bib, _engine(store.config))
    _, docs = store.load_snapshot(run_id, snapshot.snapshot_id)
    conn, cur = _connection()
    _published_cursor(cur, snapshot, docs)
    ones = list(cur.fetchone.side_effect)
    cur.fetchone.side_effect = [ones[3], ones[0]]
    cur.fetchall.side_effect = [[{"payload": doc.model_dump(), "cosine_similarity": 0.8} for doc in docs]]
    manager = LakebasePgVectorManager(store.config, _workspace())
    with patch("psycopg.connect", return_value=conn):
        result = manager.search(snapshot, docs, docs[0].embedding, 2)
        assert result[0]["cosine_similarity"] == 0.8 and "embedding" not in result[0]
        sql_text = _sql(cur)[-1]
        assert snapshot_table(snapshot.model_key, snapshot.snapshot_id) in sql_text
        assert run_id not in sql_text
        assert cur.execute.call_args.args[1][-1] == 2
        cur.fetchone.side_effect = [None]
        with pytest.raises(ValueError, match="marker"):
            manager.search(snapshot, docs, docs[0].embedding, 2)
        foreign = docs[0].model_dump()
        foreign["run_id"] = "foreign"
        cur.fetchone.side_effect = [ones[3], ones[0]]
        cur.fetchall.side_effect = [[{"payload": foreign, "cosine_similarity": 0.8}]]
        with pytest.raises(ValueError, match="outside"):
            manager.search(snapshot, docs, docs[0].embedding, 2)
        cur.fetchone.side_effect = [ones[3], ones[0]]
        cur.fetchall.side_effect = [[{"payload": docs[0].model_dump(), "cosine_similarity": 0.8}]]
        with pytest.raises(ValueError, match="expected 2"):
            manager.search(snapshot, docs, docs[0].embedding, 2)
        cur.fetchone.side_effect = [ones[3], ones[0]]
        cur.fetchall.side_effect = [[{"payload": doc.model_dump(), "cosine_similarity": 0.8} for doc in docs]]
        assert len(manager.search_published(snapshot, document_fingerprint(docs), docs[0].embedding, 2)) == 2
        assert "SET TRANSACTION READ ONLY" in _sql(cur)
        cur.fetchone.side_effect = [ones[3]]
        with pytest.raises(ValueError, match="marker"):
            manager.search_published(snapshot, "wrong-fingerprint", docs[0].embedding, 2)


def test_identifiers_and_dimension_limits_are_explicit(prepared_input: tuple[SearchStore, str, str]) -> None:
    """Reject malformed identifiers and unsupported dimensions before connecting."""
    with pytest.raises(ValueError, match="schema"):
        lakebase_schema(_config(lakebase_schema='a"; DROP SCHEMA public'))
    with pytest.raises(ValueError, match="snapshot"):
        snapshot_table("0" * 64, '"; DROP TABLE')
    store, run_id, bib = prepared_input
    snapshot = store.prepare(run_id, bib, _engine(store.config))
    _, docs = store.load_snapshot(run_id, snapshot.snapshot_id)
    snapshot.dimension = 2001
    with patch("psycopg.connect") as connect, pytest.raises(ValueError, match="2000"):
        LakebasePgVectorManager(store.config, _workspace()).publish(snapshot, docs)
    connect.assert_not_called()
