"""Conservative identity reconciliation using exact identifiers, never titles."""

from collections import defaultdict
from urllib.parse import urlsplit

from autographrs.enrichment.arxiv_html import normalize_doi
from autographrs.models import BibliographicReference, Paper, PaperEnrichment
from autographrs.sources.arxiv import canonical_arxiv_id


def _provider_alias(value: str) -> str | None:
    parsed = urlsplit(value)
    identifier = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    if parsed.hostname in {"www.semanticscholar.org", "semanticscholar.org"} and identifier:
        return f"scholar:{identifier}"
    if parsed.hostname == "openalex.org" and identifier:
        return f"openalex:{identifier}"
    return None


def _doi_alias(value: str) -> str:
    doi = normalize_doi(value)
    if doi.startswith("10.48550/arxiv."):
        return f"arxiv:{canonical_arxiv_id(doi.removeprefix('10.48550/arxiv.'))}"
    return f"doi:{doi}"


def paper_aliases(paper: Paper, enrichment: PaperEnrichment) -> set[str]:
    """Keep source-local identity as well as exact bibliographic/provider aliases."""
    aliases = {f"source-{paper.source}:{paper.paper_id}"}
    if paper.source == "arxiv":
        aliases.add(f"arxiv:{canonical_arxiv_id(paper.paper_id)}")
    if paper.doi:
        aliases.add(_doi_alias(paper.doi))
    for identifier in enrichment.citation_paper_ids or []:
        if alias := _provider_alias(identifier):
            aliases.add(alias)
    return aliases


def reference_aliases(reference: BibliographicReference) -> set[str]:
    """Unresolved targets stay local, regardless of textual similarity."""
    if reference.match_method == "unresolved":
        return set()
    aliases = {f"arxiv:{canonical_arxiv_id(value)}" for value in reference.arxiv_ids}
    aliases.update(_doi_alias(value) for value in reference.dois)
    if reference.provider_id and (alias := _provider_alias(reference.provider_id)):
        aliases.add(alias)
    namespace, _, identifier = reference.target_id.partition(":")
    if identifier and namespace in {"arxiv", "doi", "scholar", "openalex"}:
        if namespace == "doi":
            aliases.add(_doi_alias(identifier))
        elif namespace == "arxiv":
            aliases.add(f"arxiv:{canonical_arxiv_id(identifier)}")
        else:
            aliases.add(reference.target_id)
    return aliases


def identity_groups(aliases: dict[str, set[str]]) -> list[tuple[list[str], bool]]:
    """Quarantine an entire conflicting component instead of accepting an order-dependent merge."""
    parents = {key: key for key in aliases}

    def root(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    by_alias: dict[str, str] = {}
    for key in sorted(aliases):
        for alias in sorted(aliases[key]):
            if alias in by_alias:
                parents[root(key)] = root(by_alias[alias])
            else:
                by_alias[alias] = key
    components: dict[str, list[str]] = defaultdict(list)
    for key in sorted(aliases):
        components[root(key)].append(key)
    result: list[tuple[list[str], bool]] = []
    for members in components.values():
        by_namespace: dict[str, set[str]] = defaultdict(set)
        for key in members:
            for alias in aliases[key]:
                namespace, _, identifier = alias.partition(":")
                by_namespace[namespace].add(identifier)
        ambiguous = sum(key.startswith("paper:") for key in members) > 1 or any(
            len(values) > 1 for values in by_namespace.values()
        )
        if ambiguous:
            result.extend(([key], True) for key in members)
        else:
            result.append((members, False))
    return result
