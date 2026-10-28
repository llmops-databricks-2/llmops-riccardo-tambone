"""Enrichment modules: LLM tool-calling tagger and mutual citation resolver."""

from autographrs.enrichment.arxiv_html import ArxivHtmlCitationResolver
from autographrs.enrichment.citation_service import CitationService
from autographrs.enrichment.citations import SemanticScholarCitationResolver, resolve_mutual_citations
from autographrs.enrichment.llm import create_gateway_openai_client
from autographrs.enrichment.openalex import OpenAlexCitationResolver
from autographrs.enrichment.tagger import PaperTagger

__all__ = [
    "PaperTagger",
    "ArxivHtmlCitationResolver",
    "OpenAlexCitationResolver",
    "CitationService",
    "SemanticScholarCitationResolver",
    "create_gateway_openai_client",
    "resolve_mutual_citations",
]
