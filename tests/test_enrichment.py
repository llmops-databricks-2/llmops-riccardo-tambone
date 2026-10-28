"""Unit tests for LLM enrichment and bibliography retrieval."""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_ingestion import FakeArxivClient, FakeSparkSession, _fake_arxiv_result

from autographrs.config import ProjectConfig
from autographrs.enrichment.citations import CitationLookup, fetch_bibliographies
from autographrs.enrichment.tagger import (
    TOOL_NAME,
    PaperTagger,
    fetch_arxiv_category_taxonomy,
    humanize_category,
    merge_tags,
    normalize_tag,
)
from autographrs.models import Paper
from autographrs.pipeline import run_ingestion
from autographrs.sources.arxiv import ArxivSource
from autographrs.storage.delta import (
    PAPER_ENRICHMENT_DDL_SCHEMA,
    DeltaPaperStore,
    enrich_papers,
    enrichments_to_records,
)

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "project_config.yml"


class FakeChatCompletions:
    """Stub `client.chat.completions` returning a tool call payload."""

    def __init__(self, tags: list[str], summary: str, should_fail: bool = False) -> None:
        self.tags = tags
        self.summary = summary
        self.should_fail = should_fail
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        """Record call arguments and return a fake tool-calling response."""
        self.calls.append(kwargs)
        if self.should_fail:
            raise RuntimeError("Simulated AI Gateway error")

        tool_call = SimpleNamespace(
            function=SimpleNamespace(
                name=TOOL_NAME,
                arguments=json.dumps({"tags": self.tags, "summary": self.summary}),
            )
        )
        message = SimpleNamespace(tool_calls=[tool_call], content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAIClient:
    """Minimal OpenAI client double exposing `.chat.completions`."""

    def __init__(self, completions: FakeChatCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


class FakeCitationProvider:
    """Deterministic stub for `CitationProviderProtocol`."""

    def __init__(self, mapping: dict[str, tuple[set[str], set[str]]]) -> None:
        self._mapping = mapping

    def fetch_references_and_citations(self, paper_id: str) -> CitationLookup:
        """Return preset `(references, citations)` for `paper_id`."""
        refs, cites = self._mapping.get(paper_id, (set(), set()))
        return CitationLookup(references=refs, citations=cites, source="test_provider")


def _make_paper(
    paper_id: str,
    title: str,
    abstract: str = "An abstract about large language models.",
) -> Paper:
    now = datetime(2024, 6, 1, tzinfo=UTC)
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=["Alice Smith"],
        abstract=abstract,
        published_at=now,
        updated_at=now,
        categories=["cs.CL", "cs.AI"],
        primary_category="cs.CL",
        html_url=f"http://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}.pdf",
    )


FAKE_TAXONOMY: dict[str, str] = {
    "cs.cl": "computation-and-language",
    "cs.ai": "artificial-intelligence",
    "cs.lg": "machine-learning",
    "stat.ml": "machine-learning",
}


def test_fetch_arxiv_category_taxonomy_parses_official_html() -> None:
    """Official arXiv taxonomy HTML is parsed into code -> kebab-case label map."""
    sample_html = (
        "<div><h4>cs.LG <span>(Machine Learning)</span></h4>"
        "<h4>cs.CL <span>(Computation and Language)</span></h4></div>"
    )

    class _FakeResp:
        def __enter__(self) -> "_FakeResp":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self) -> bytes:
            return sample_html.encode("utf-8")

    fetch_arxiv_category_taxonomy.cache_clear()
    with patch("urllib.request.urlopen", return_value=_FakeResp()):
        parsed = fetch_arxiv_category_taxonomy()

    assert parsed == {
        "cs.lg": "machine-learning",
        "cs.cl": "computation-and-language",
    }
    fetch_arxiv_category_taxonomy.cache_clear()


def test_normalize_and_merge_tags() -> None:
    """Tags are normalized to lowercase kebab-case and arXiv categories become human-readable."""
    assert normalize_tag("#Graph RAG / LLMs!", taxonomy=FAKE_TAXONOMY) == "graph-rag-llms"
    assert humanize_category("cs.LG", taxonomy=FAKE_TAXONOMY) == "machine-learning"
    assert humanize_category("cs.CL", taxonomy=FAKE_TAXONOMY) == "computation-and-language"
    assert humanize_category("stat.ML", taxonomy=FAKE_TAXONOMY) == "machine-learning"
    merged = merge_tags(
        ["Graph RAG", "Knowledge Graph", "cs.CL"],
        ["cs.CL", "cs.AI", "cs.LG"],
        taxonomy=FAKE_TAXONOMY,
    )
    assert merged == [
        "graph-rag",
        "knowledge-graph",
        "computation-and-language",
        "artificial-intelligence",
        "machine-learning",
    ]


def test_paper_tagger_tool_calling_and_fallback() -> None:
    """PaperTagger extracts separate LLM tags, humanized arXiv categories, and summary via tool calling."""
    cfg = ProjectConfig.from_yaml(str(PROJECT_CONFIG), "dev")
    paper = _make_paper("2401.00001", "Graph RAG Systems")

    completions = FakeChatCompletions(
        tags=["Graph RAG", "Knowledge Graphs"],
        summary="This paper introduces a graph-based retrieval system for LLMs.",
    )
    with patch("autographrs.enrichment.tagger.fetch_arxiv_category_taxonomy", return_value=FAKE_TAXONOMY):
        tagger = PaperTagger(config=cfg, client=FakeOpenAIClient(completions))
        tags, arxiv_cats, summary = tagger.extract_tags_and_summary(paper)

        assert tags == ["graph-rag", "knowledge-graphs"]
        assert arxiv_cats == ["computation-and-language", "artificial-intelligence"]
        assert summary == "This paper introduces a graph-based retrieval system for LLMs."
        assert len(completions.calls) == 1

        # Fallback when gateway fails
        failing_tagger = PaperTagger(
            config=cfg,
            client=FakeOpenAIClient(FakeChatCompletions(tags=[], summary="", should_fail=True)),
        )
        fb_tags, fb_cats, fb_summary = failing_tagger.extract_tags_and_summary(paper)
        assert fb_tags == []
        assert fb_cats == ["computation-and-language", "artificial-intelligence"]
        assert fb_summary == paper.abstract


def test_bibliography_retrieval_rejects_duplicate_ids() -> None:
    """Reject ambiguous source-local keys before calling a provider."""
    paper = _make_paper("2401.00001", "Paper A")
    with pytest.raises(ValueError, match="unique paper_id"):
        fetch_bibliographies([paper, paper], FakeCitationProvider({}))


def test_run_ingestion_with_enrichment_saves_to_delta(tmp_path: Path) -> None:
    """Full run_ingestion with enrich=True writes both `papers` and `paper_enrichment` Delta tables."""
    cfg = ProjectConfig.from_yaml(str(PROJECT_CONFIG), "dev")
    fake_spark = FakeSparkSession()
    store = DeltaPaperStore(config=cfg, spark=fake_spark, volume_root_override=tmp_path)  # type: ignore[arg-type]

    r1 = _fake_arxiv_result(arxiv_id="2401.00001v1", summary="Paper 1 base work.")
    r2 = _fake_arxiv_result(arxiv_id="2402.00002v1", summary="Paper 2 extends 2401.00001.")
    source = ArxivSource(client=FakeArxivClient([r1, r2]))  # type: ignore[arg-type]

    tagger = PaperTagger(
        config=cfg,
        client=FakeOpenAIClient(
            FakeChatCompletions(tags=["llm-agents", "citation-graph"], summary="Short LLM summary.")
        ),
    )
    citation_provider = FakeCitationProvider({"2402.00002": ({"2401.00001"}, set())})

    with patch("autographrs.enrichment.tagger.fetch_arxiv_category_taxonomy", return_value=FAKE_TAXONOMY):
        result = run_ingestion(
            topic="citation graphs",
            num_papers=2,
            config=cfg,
            source=source,
            store=store,
            download_pdfs=False,
            enrich=True,
            tagger=tagger,
            citation_provider=citation_provider,
        )

    assert len(result.enrichments) == 2
    assert result.enrichment_table == "llmops_dev.riccardo_tambone.paper_enrichment"
    assert fake_spark.schemas["llmops_dev.riccardo_tambone.paper_enrichment"] == PAPER_ENRICHMENT_DDL_SCHEMA

    enrich_rows = fake_spark.tables["llmops_dev.riccardo_tambone.paper_enrichment"]
    assert len(enrich_rows) == 2
    row_p1 = next(r for r in enrich_rows if r["paper_id"] == "2401.00001")

    assert row_p1["title"] == "Graph RAG for Scientific Literature"
    assert row_p1["authors"] == ["Alice Smith", "Bob Jones"]
    assert all(not {"referenced_arxiv_ids", "cites_in_set", "cited_by_in_set"} & row.keys() for row in enrich_rows)
    assert all("cited_paper_ids" not in row for row in fake_spark.tables[store.full_table_name])
    assert row_p1["tags"] == ["llm-agents", "citation-graph"]
    assert row_p1["arxiv_category"] == ["computation-and-language", "artificial-intelligence"]
    assert row_p1["citation_status"] == "ok"
    assert row_p1["citation_source"] == "test_provider"
    assert row_p1["citation_error"] is None


def test_failed_citation_lookup_is_persisted_as_unavailable() -> None:
    """An unavailable provider must not appear as a successfully empty bibliography."""
    cfg = ProjectConfig.from_yaml(str(PROJECT_CONFIG), "dev")
    paper = _make_paper("2401.00001", "Paper A", abstract="Mentions 2402.00002.")
    provider = FakeCitationProvider({})
    tagger = PaperTagger(config=cfg, client=FakeOpenAIClient(FakeChatCompletions(["rag"], "Summary.")))
    with (
        patch.object(
            provider,
            "fetch_references_and_citations",
            return_value=CitationLookup(status="rate_limited", error="Semantic Scholar HTTP 429"),
        ),
        patch("autographrs.enrichment.tagger.fetch_arxiv_category_taxonomy", return_value=FAKE_TAXONOMY),
    ):
        _, enrichment = enrich_papers([paper], "test-run", cfg, tagger, provider)
    row = enrichments_to_records(enrichment)[0]
    assert row["citation_status"] == "rate_limited"
    assert row["citation_error"] == "Semantic Scholar HTTP 429"
    assert row["citation_fetched_at"] is not None
    assert row["bibliography"] == []
