"""Batch bibliography abstract retrieval, text classification and snapshot regression tests."""

import io
import json
import urllib.error
from copy import deepcopy
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import pytest
from test_graph import _pair, _publish_run, _ref
from test_ingestion import FakeSparkSession

from autographrs.config import ProjectConfig
from autographrs.enrichment.abstracts import AbstractResolver, reconstruct_abstract
from autographrs.enrichment.bibliography import BibliographyStore
from autographrs.enrichment.citations import SemanticScholarCitationResolver
from autographrs.enrichment.openalex import OpenAlexCitationResolver
from autographrs.graph import GraphStore, build_citation_graph
from autographrs.graph.models import GraphNode


def _nodes() -> list[GraphNode]:
    graph = build_citation_graph(
        "run-1",
        [
            _pair(
                "2401.00001",
                [
                    _ref("scholar:s1"),
                    _ref("scholar:s1", 2),
                    _ref("doi:10.1000/a", 3),
                    _ref("ref:unknown", 4),
                    _ref("openalex:W2", 5),
                ],
            )
        ],
    )
    return [node for node in graph.nodes if node.kind != "selected"]


def test_scholar_batch_and_openalex_fallback_preserve_provenance_and_variants() -> None:
    """Only exactly identified works are queried, once per deduplicated identity."""
    resolver = AbstractResolver(
        scholar=SemanticScholarCitationResolver(min_interval=0),
        openalex=OpenAlexCitationResolver(min_interval=0),
    )
    with (
        patch.object(
            resolver.scholar,
            "_request_json",
            return_value=[
                {"paperId": "doi-a", "title": "Article A", "abstract": "   "},
                {"paperId": "s1", "title": "Article S", "abstract": "Abstract S"},
            ],
        ) as scholar,
        patch.object(resolver.openalex, "_request") as openalex,
    ):
        from autographrs.enrichment.openalex import _Page

        openalex.side_effect = [
            _Page.model_validate(
                {
                    "results": [
                        {
                            "id": "https://openalex.org/W2",
                            "title": "Article W",
                            "abstract_inverted_index": {"world": [1], "Hello": [0]},
                        },
                    ]
                }
            ),
            _Page.model_validate(
                {
                    "results": [
                        {"id": "https://openalex.org/W3", "doi": "https://doi.org/10.1000/a", "title": "Article A"},
                    ]
                }
            ),
        ]
        original = _nodes()
        before = deepcopy(original)
        results = {item.node_id: item for item in resolver.fetch(original)}
    assert original == before
    assert scholar.call_count == 1
    assert scholar.call_args.kwargs["body"]["ids"] == ["DOI:10.1000/a", "s1"]
    assert "references" not in scholar.call_args.args[0]
    assert openalex.call_count == 2
    assert results["scholar:s1"].text_variant == "title_abstract"
    assert results["openalex:W2"].abstract == "Hello world"
    assert results["openalex:W2"].provider == "openalex"
    assert results["doi:10.1000/a"].text_variant == "title_only"
    assert len(results["doi:10.1000/a"].attempts) == 2
    unresolved = next(item for item in results.values() if item.status == "unresolved")
    assert unresolved.attempts == []


@pytest.mark.parametrize("index", [{"A": [1]}, {"A": [-1]}, {"A": [0], "B": [0]}])
def test_invalid_inverted_indexes_are_rejected(index: dict[str, list[int]]) -> None:
    """Malformed positions must not create fabricated or reordered abstracts."""
    with pytest.raises(ValueError, match="positions"):
        reconstruct_abstract(index)


def test_abstract_reconstruction_handles_repetitions_and_absence() -> None:
    """Positions, not dictionary order, determine the reconstructed text."""
    assert reconstruct_abstract({"B": [1], "A": [0, 2]}) == "A B A"
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None


def test_rate_limit_stops_remaining_scholar_chunks() -> None:
    """An exhausted provider is not retried afresh for every work."""
    resolver = AbstractResolver(scholar=SemanticScholarCitationResolver(min_interval=0, batch_size=1))
    limited = urllib.error.HTTPError("https://example.org", 429, "", Message(), None)
    graph = build_citation_graph("run-1", [_pair("2401.00001", [_ref("scholar:s1"), _ref("scholar:s2", 2)])])
    with (
        patch.object(resolver.scholar, "_request_json", side_effect=limited) as request,
        patch.object(resolver.openalex, "_request") as fallback,
    ):
        results = resolver.fetch([node for node in graph.nodes if node.kind != "selected"])
    assert request.call_count == 1
    fallback.assert_not_called()
    assert all(item.status == "rate_limited" and item.text_variant == "title_only" for item in results)


def test_missing_title_is_not_replaced_by_identifier() -> None:
    """An API missing title cannot produce a title-only embedding from a placeholder."""
    resolver = AbstractResolver()
    reference = _ref("scholar:s1")
    reference.title = None
    graph = build_citation_graph("run-1", [_pair("2401.00001", [reference])])
    with patch.object(resolver.scholar, "_request_json", return_value=[None]):
        result = resolver.fetch([node for node in graph.nodes if node.kind != "selected"])[0]
    assert result.title is None and result.text_variant is None
    assert result.status == "missing"


def test_hundreds_of_cited_works_use_bounded_batches() -> None:
    """Two hundred and one unique works are queried in 100/100/1 batches, not per paper."""
    resolver = AbstractResolver(scholar=SemanticScholarCitationResolver(min_interval=0))
    nodes = [
        GraphNode(node_id=f"scholar:s{i}", kind="external", identifiers=[f"scholar:s{i}"], title=f"Paper {i}")
        for i in range(201)
    ]

    def respond(url: str, body: dict[str, list[str]]) -> list[dict[str, str]]:
        return [{"paperId": pid, "title": f"Paper {pid}", "abstract": "Abstract"} for pid in body["ids"]]

    with patch.object(resolver.scholar, "_request_json", side_effect=respond) as request:
        results = resolver.fetch(nodes)
    assert [len(call.kwargs["body"]["ids"]) for call in request.call_args_list] == [100, 100, 1]
    assert len(results) == 201 and all(item.text_variant == "title_abstract" for item in results)


def test_malformed_batch_and_unrelated_openalex_results_are_not_assumed_matches() -> None:
    """Report malformed responses and never attach metadata from an unrelated DOI."""
    from autographrs.enrichment.openalex import _Page

    resolver = AbstractResolver()
    graph = build_citation_graph("run-1", [_pair("2401.00001", [_ref("doi:10.1000/a")])])
    with (
        patch.object(resolver.scholar, "_request_json", return_value=[]),
        patch.object(
            resolver.openalex,
            "_request",
            return_value=_Page.model_validate(
                {
                    "results": [
                        {
                            "id": "https://openalex.org/W99",
                            "doi": "https://doi.org/10.1000/unrelated",
                            "title": "Wrong title",
                            "abstract_inverted_index": {"Wrong": [0]},
                        }
                    ]
                }
            ),
        ),
    ):
        result = resolver.fetch([node for node in graph.nodes if node.kind != "selected"])[0]
    assert result.abstract is None and result.title == "An external paper"
    assert result.status == "error"
    assert result.attempts[0].status == "error"


def test_http_wire_fields_and_openalex_abstract_parsing() -> None:
    """Existing provider HTTP helpers validate real JSON shapes without fetching references."""
    resolver = AbstractResolver(
        scholar=SemanticScholarCitationResolver(min_interval=0),
        openalex=OpenAlexCitationResolver(min_interval=0),
    )
    graph = build_citation_graph("run-1", [_pair("2401.00001", [_ref("openalex:W2")])])
    response = io.BytesIO(
        json.dumps(
            {"results": [{"id": "https://openalex.org/W2", "title": "Title", "abstract_inverted_index": {"Text": [0]}}]}
        ).encode()
    )
    with patch("urllib.request.urlopen", return_value=response) as request:
        result = resolver.fetch([node for node in graph.nodes if node.kind != "selected"])[0]
    assert request.call_count == 1
    assert result.abstract == "Text" and result.text_variant == "title_abstract"


def test_snapshots_and_graph_consume_exact_metadata_without_rewriting_originals(tmp_path: Path) -> None:
    """Published abstract snapshots are run-scoped, immutable, and referenced by graph builds."""
    config = ProjectConfig.from_yaml(str(Path(__file__).resolve().parents[1] / "project_config.yml"), "dev")
    spark = FakeSparkSession()
    graph_store = GraphStore(config, spark=spark, volume_root_override=tmp_path)  # type: ignore[arg-type]
    store = BibliographyStore(config, spark=spark, volume_root_override=tmp_path)  # type: ignore[arg-type]
    run_id = _publish_run(graph_store)
    row = spark.tables[store.full_enrichment_table_name][0]
    row["bibliography"] = [_ref("scholar:s1").model_dump()]
    before = deepcopy(row)
    resolver = AbstractResolver(scholar=SemanticScholarCitationResolver(min_interval=0))
    with patch.object(
        resolver.scholar,
        "_request_json",
        return_value=[{"paperId": "s1", "title": "Cited article", "abstract": "An abstract"}],
    ):
        snapshot = store.enrich_bibliography(run_id, resolver)
    assert row == before
    metadata = store.load_metadata(run_id, snapshot.snapshot_id)
    assert metadata[0].abstract == "An abstract"
    build = graph_store.build_graph(run_id, snapshot.snapshot_id)
    graph = graph_store.load_graph(run_id, build.graph_build_id)
    assert graph.bibliography_snapshot_id == snapshot.snapshot_id
    node = next(node for node in graph.nodes if node.kind == "external")
    assert node.abstract == "An abstract" and node.text_variant == "title_abstract"
    with pytest.raises(ValueError, match="published"):
        store.load_metadata("another-run", snapshot.snapshot_id)
    with patch.object(resolver, "fetch", side_effect=RuntimeError("test failure")), pytest.raises(RuntimeError):
        store.enrich_bibliography(run_id, resolver)
    failed = next(row for row in spark.tables[store.full_snapshots_table_name] if row["status"] == "FAILED")
    with pytest.raises(ValueError, match="published"):
        store.load_metadata(run_id, failed["snapshot_id"])
    assert store.load_metadata(run_id, snapshot.snapshot_id) == metadata
    spark.tables[store.full_metadata_table_name][0]["abstract"] = None
    spark.tables[store.full_metadata_table_name].clear()
    with pytest.raises(ValueError, match="counts"):
        store.load_metadata(run_id, snapshot.snapshot_id)
