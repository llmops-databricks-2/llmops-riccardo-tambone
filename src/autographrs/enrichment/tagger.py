"""LLM tool-calling enrichment for paper tags and concise summary."""

import json
import re
import urllib.request
from functools import lru_cache
from typing import Any, Protocol

import backoff
from loguru import logger
from pydantic import BaseModel, Field

from autographrs.config import ProjectConfig
from autographrs.enrichment.llm import create_gateway_openai_client
from autographrs.models import Paper

TOOL_NAME = "record_paper_enrichment"
ARXIV_TAXONOMY_URL = "https://arxiv.org/category_taxonomy"
_TAXONOMY_ENTRY_RE = re.compile(r"<h4>\s*([a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?)\s*<span>\(([^)]+)\)</span>\s*</h4>")


@lru_cache(maxsize=1)
def fetch_arxiv_category_taxonomy(timeout: int = 10) -> dict[str, str]:
    """Fetch and cache the official arXiv category taxonomy from `https://arxiv.org/category_taxonomy`.

    Parses all `<h4>code <span>(Full Name)</span></h4>` entries (~155 official categories)
    and maps each lowercase code (e.g. `'cs.lg'`) to a kebab-case readable label
    (e.g. `'machine-learning'`). Returns an empty dict if offline or unreachable.

    Args:
        timeout: HTTP timeout in seconds

    Returns:
        Dictionary mapping lowercase category code to kebab-case full name

    """
    req = urllib.request.Request(ARXIV_TAXONOMY_URL, headers={"User-Agent": "autographrs/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug("Could not fetch official arXiv category taxonomy ({}); using raw code normalization", exc)
        return {}

    taxonomy: dict[str, str] = {}
    for code, full_name in _TAXONOMY_ENTRY_RE.findall(html):
        slug = re.sub(r"[^a-z0-9]+", "-", full_name.strip().lower()).strip("-")
        if slug:
            taxonomy[code.strip().lower()] = slug
    return taxonomy


SYSTEM_PROMPT = (
    "You are a scientific literature curator. Given a paper's title, abstract, and categories, "
    "call the `record_paper_enrichment` tool with:\n"
    "1. `tags`: 3 to 7 lowercase kebab-case thematic tags (e.g. 'graph-rag', 'knowledge-graph', "
    "'transformer', 'retrieval-augmented-generation').\n"
    "2. `summary`: a concise 2-3 sentence summary of the paper's core problem and method."
)

RECORD_ENRICHMENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Record normalized thematic tags and a concise summary for a research paper.",
        "parameters": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3 to 7 normalized lowercase kebab-case thematic tags.",
                },
                "summary": {
                    "type": "string",
                    "description": "Concise 2-3 sentence summary of the paper.",
                },
            },
            "required": ["tags", "summary"],
            "additionalProperties": False,
        },
    },
}


class TaggingOutput(BaseModel):
    """Validated structured payload returned by the `record_paper_enrichment` tool call."""

    tags: list[str] = Field(default_factory=list, description="Normalized thematic tags")
    summary: str = Field(default="", description="Concise 2-3 sentence summary")


class ChatCompletionsClientProtocol(Protocol):
    """Minimal protocol for an OpenAI-compatible client exposing `chat.completions.create`."""

    @property
    def chat(self) -> Any:  # noqa: ANN401
        """Return chat namespace."""
        ...


def humanize_category(category: str, taxonomy: dict[str, str] | None = None) -> str:
    """Translate an arXiv category code (e.g. 'cs.LG') into its official human-readable name ('machine-learning').

    Args:
        category: Raw category code (e.g. 'cs.LG', 'stat.ML', 'cs.CL')
        taxonomy: Optional pre-loaded taxonomy map; fetched from `arxiv.org/category_taxonomy` if omitted

    Returns:
        Human-readable kebab-case label

    """
    key = category.strip().lower()
    active_taxonomy = taxonomy if taxonomy is not None else fetch_arxiv_category_taxonomy()
    if key in active_taxonomy:
        return active_taxonomy[key]
    return re.sub(r"[^a-z0-9]+", "-", key).strip("-")


def normalize_tag(raw_tag: str, taxonomy: dict[str, str] | None = None) -> str:
    """Normalize a tag string into lowercase kebab-case suitable for Obsidian `#tags` and filenames.

    Automatically translates arXiv category codes like 'cs.LG' into their official names.

    Args:
        raw_tag: Raw category or LLM tag (e.g. 'Graph RAG', 'cs.LG')
        taxonomy: Optional pre-loaded taxonomy map

    Returns:
        Normalized tag (e.g. 'graph-rag', 'machine-learning')

    """
    stripped = raw_tag.strip().lstrip("#")
    lower = stripped.lower()
    active_taxonomy = taxonomy if taxonomy is not None else fetch_arxiv_category_taxonomy()
    if lower in active_taxonomy:
        return active_taxonomy[lower]
    return re.sub(r"[^a-z0-9]+", "-", lower).strip("-")


def humanize_categories(
    categories: list[str],
    taxonomy: dict[str, str] | None = None,
) -> list[str]:
    """Convert a list of raw arXiv category codes into deduplicated humanized labels."""
    active_taxonomy = taxonomy if taxonomy is not None else fetch_arxiv_category_taxonomy()
    result: list[str] = []
    seen: set[str] = set()
    for cat in categories:
        human = humanize_category(cat, taxonomy=active_taxonomy)
        if human and human not in seen:
            seen.add(human)
            result.append(human)
    return result


def normalize_tags(
    raw_tags: list[str],
    taxonomy: dict[str, str] | None = None,
) -> list[str]:
    """Normalize and deduplicate a list of LLM-extracted tags while preserving order."""
    active_taxonomy = taxonomy if taxonomy is not None else fetch_arxiv_category_taxonomy()
    result: list[str] = []
    seen: set[str] = set()
    for candidate in raw_tags:
        norm = normalize_tag(candidate, taxonomy=active_taxonomy)
        if norm and norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


def merge_tags(
    llm_tags: list[str],
    categories: list[str],
    taxonomy: dict[str, str] | None = None,
) -> list[str]:
    """Combine and deduplicate normalized LLM tags and humanized source categories while preserving order."""
    active_taxonomy = taxonomy if taxonomy is not None else fetch_arxiv_category_taxonomy()
    humanized_cats = humanize_categories(categories, taxonomy=active_taxonomy)
    return normalize_tags([*llm_tags, *humanized_cats], taxonomy=active_taxonomy)


class PaperTagger:
    """Extract normalized tags and a short summary per paper using deterministic tool calling."""

    def __init__(
        self,
        config: ProjectConfig,
        client: ChatCompletionsClientProtocol | None = None,
        max_tokens: int = 512,
    ) -> None:
        """Initialize the tagger.

        Args:
            config: Loaded `ProjectConfig` containing `llm_endpoint`
            client: Optional pre-configured OpenAI-compatible client (for testing)
            max_tokens: Maximum tokens for the tool-calling completion

        """
        self.config = config
        self._client = client
        self.max_tokens = max_tokens

    @property
    def client(self) -> ChatCompletionsClientProtocol:
        """Lazily initialize the AI Gateway OpenAI client."""
        if self._client is None:
            self._client = create_gateway_openai_client()
        return self._client

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _call_llm_tool(self, paper: Paper) -> TaggingOutput:
        """Invoke `course_ops.gateway.chat` with tool calling and parse the structured arguments."""
        readable_cats = humanize_categories(paper.categories) if paper.source == "arxiv" else paper.categories
        user_msg = f"Title: {paper.title}\nCategories: {', '.join(readable_cats)}\nAbstract: {paper.abstract}"
        response = self.client.chat.completions.create(
            model=self.config.llm_endpoint,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            tools=[RECORD_ENRICHMENT_TOOL],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
            max_tokens=self.max_tokens,
        )

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            fn = getattr(call, "function", None)
            if fn and getattr(fn, "name", "") == TOOL_NAME:
                payload = json.loads(fn.arguments)
                return TaggingOutput.model_validate(payload)

        # Fallback if the gateway model returned JSON directly in message.content
        content = (getattr(message, "content", None) or "").strip()
        if content.startswith("{"):
            return TaggingOutput.model_validate(json.loads(content))

        raise ValueError(f"Model did not return a `{TOOL_NAME}` tool call for paper {paper.paper_id}")

    def extract_tags_and_summary(self, paper: Paper) -> tuple[list[str], list[str], str]:
        """Extract normalized LLM tags, humanized arXiv categories, and summary for `paper`.

        Args:
            paper: Normalized `Paper` instance

        Returns:
            Tuple of `(llm_tags, arxiv_categories, summary)`

        """
        arxiv_cats = humanize_categories(paper.categories) if paper.source == "arxiv" else []
        try:
            output = self._call_llm_tool(paper)
            llm_tags = normalize_tags(output.tags, taxonomy=None if paper.source == "arxiv" else {})
            summary = " ".join(output.summary.split()).strip() or paper.abstract[:400]
            return llm_tags, arxiv_cats, summary
        except Exception as exc:
            logger.warning(
                "LLM tool-calling enrichment failed for paper {} ({}); using empty LLM tags and abstract fallback",
                paper.paper_id,
                exc,
            )
            return [], arxiv_cats, paper.abstract[:400]
