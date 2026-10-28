"""Entrypoint functions for the AutoGraphRS ingestion pipeline."""

from datetime import date

from loguru import logger

from autographrs.config import ProjectConfig, load_config
from autographrs.enrichment.citations import CitationProviderProtocol
from autographrs.enrichment.tagger import PaperTagger
from autographrs.models import IngestionResult, RankingCriteria, SearchRequest
from autographrs.sources.arxiv import ArxivSource
from autographrs.sources.base import PaperSource
from autographrs.storage.delta import DeltaPaperStore, ingest_and_store_papers


def run_ingestion(
    topic: str,
    num_papers: int = 10,
    ranking_criteria: str | RankingCriteria = RankingCriteria.RELEVANCE,
    criteria_date: str | date | None = None,
    categories: list[str] | None = None,
    env: str = "dev",
    config_path: str = "project_config.yml",
    config: ProjectConfig | None = None,
    source: PaperSource | None = None,
    store: DeltaPaperStore | None = None,
    download_pdfs: bool = True,
    enrich: bool = True,
    tagger: PaperTagger | None = None,
    citation_provider: CitationProviderProtocol | None = None,
    run_id: str | None = None,
) -> IngestionResult:
    """Search, download, enrich (tags, summary, mutual citations), and store papers.

    Args:
        topic: Topic or search query string
        num_papers: Number of top papers to download and store
        ranking_criteria: Ranking criterion ('relevance', 'submittedDate', or 'lastUpdatedDate')
        criteria_date: Optional date or date-range filter when using a date criterion
        categories: Optional list of category filters (e.g. ['cs.AI', 'cs.CL'])
        env: Target Databricks environment ('dev', 'acc', or 'prd')
        config_path: Path to `project_config.yml`
        config: Optional pre-loaded `ProjectConfig`
        source: Optional custom `PaperSource` provider (defaults to `ArxivSource`)
        store: Optional custom `DeltaPaperStore`
        download_pdfs: Whether to download PDF files into the Unity Catalog Volume
        enrich: Whether to run LLM tool-calling enrichment and mutual citation resolution
        tagger: Optional custom `PaperTagger`
        citation_provider: Optional custom `CitationProviderProtocol`
        run_id: Failed run to resume with unchanged parameters; None creates a new search

    Returns:
        `IngestionResult` containing run metadata, ingested papers, enrichments, and storage locations

    """
    cfg = config or load_config(config_path=config_path, env=env)
    criterion = RankingCriteria(ranking_criteria)
    cleaned_date = criteria_date.strip() if isinstance(criteria_date, str) else criteria_date
    if cleaned_date == "":
        cleaned_date = None

    request = SearchRequest(
        topic=topic,
        num_papers=num_papers,
        ranking_criteria=criterion,
        criteria_date=cleaned_date,
        categories=categories or [],
    )
    paper_source = source or ArxivSource()

    logger.info(
        "Starting ingestion in env='{}' ({}) for topic='{}' (top_n={}, criteria={}, criteria_date={}, enrich={})",
        env,
        cfg.full_schema_name,
        request.topic,
        request.num_papers,
        request.ranking_criteria,
        request.criteria_date,
        enrich,
    )

    return ingest_and_store_papers(
        request=request,
        config=cfg,
        source=paper_source,
        store=store,
        download_pdfs=download_pdfs,
        enrich=enrich,
        tagger=tagger,
        citation_provider=citation_provider,
        run_id=run_id,
    )
