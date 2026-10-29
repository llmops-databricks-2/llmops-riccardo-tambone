"""Pure citation graph construction, independent of API calls and Spark."""

import hashlib
import json
from collections import defaultdict

from autographrs.graph.identity import identity_groups, paper_aliases, reference_aliases
from autographrs.graph.models import CitationEvidence, CitationGraph, GraphEdge, GraphNode
from autographrs.models import BibliographicReference, Paper, PaperEnrichment


def _digest(*parts: str) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=True).encode()).hexdigest()


def build_citation_graph(run_id: str, dataset: list[tuple[Paper, PaperEnrichment]]) -> CitationGraph:
    """Build an open, one-hop graph from exactly one run's acquired bibliography."""
    selected: dict[str, tuple[Paper, PaperEnrichment]] = {}
    references: dict[str, tuple[str, BibliographicReference]] = {}
    aliases: dict[str, set[str]] = {}
    warnings: set[str] = set()
    for paper, enrichment in sorted(dataset, key=lambda pair: (pair[0].source, pair[0].paper_id)):
        if enrichment.run_id != run_id or (paper.source, paper.paper_id) != (
            enrichment.source,
            enrichment.paper_id,
        ):
            raise ValueError("Graph input must contain matching paper/enrichment pairs from one run")
        key = f"paper:{paper.source}:{paper.paper_id}"
        if key in selected:
            raise ValueError("Graph input contains duplicate selected paper keys")
        selected[key] = (paper, enrichment)
        aliases[key] = paper_aliases(paper, enrichment)
        if enrichment.bibliography is None or enrichment.citation_status != "ok":
            warnings.add(
                f"{key}: bibliography status={enrichment.citation_status}; "
                f"available={enrichment.bibliography is not None}; citation coverage is not guaranteed"
            )
        for index, reference in enumerate(enrichment.bibliography or []):
            ref_key = f"ref:{_digest(key, str(index), reference.reference_id, str(reference.position))}"
            references[ref_key] = (key, reference)
            aliases[ref_key] = reference_aliases(reference)

    nodes: list[GraphNode] = []
    node_for: dict[str, str] = {}
    for members, ambiguous in identity_groups(aliases):
        selected_members = sorted(key for key in members if key in selected)
        identifiers = sorted({alias for key in members for alias in aliases[key]})
        if ambiguous:
            warnings.add(f"Ambiguous identity retained separately: {members[0]} ({', '.join(identifiers)})")
        if selected_members:
            key = selected_members[0]
            paper, enrichment = selected[key]
            node = GraphNode(
                node_id=key,
                kind="selected",
                identity_status="ambiguous" if ambiguous else "identified",
                identifiers=identifiers,
                source=paper.source,
                paper_id=paper.paper_id,
                title=paper.title,
                authors=paper.authors,
                publication_year=paper.published_at.year,
                urls=sorted(
                    {
                        paper.html_url,
                        *[url for member in members if member in references for url in references[member][1].urls],
                    }
                ),
                abstract=paper.abstract,
                summary=enrichment.summary,
                tags=enrichment.tags,
                arxiv_category=enrichment.arxiv_category,
                pdf_volume_path=paper.pdf_volume_path,
            )
        else:
            entries = [references[key][1] for key in sorted(members)]
            representative = next((entry for entry in entries if entry.title), entries[0])
            unresolved = ambiguous or not identifiers
            node = GraphNode(
                node_id=members[0] if unresolved else identifiers[0],
                kind="unresolved" if unresolved else "external",
                identity_status="ambiguous" if ambiguous else "missing" if unresolved else "identified",
                identifiers=identifiers,
                title=representative.title or representative.text,
                authors=representative.authors or [],
                publication_year=representative.publication_year,
                urls=sorted({url for entry in entries for url in entry.urls}),
            )
        nodes.append(node)
        node_for.update(dict.fromkeys(members, node.node_id))

    evidence: dict[tuple[str, str], list[CitationEvidence]] = defaultdict(list)
    for key, (citing_key, reference) in sorted(references.items()):
        src, dst = node_for[citing_key], node_for[key]
        if src == dst:
            warnings.add(f"Self-citation omitted: {citing_key} reference {reference.reference_id}")
            continue
        paper, enrichment = selected[citing_key]
        evidence[(src, dst)].append(
            CitationEvidence(
                source=paper.source,
                paper_id=paper.paper_id,
                reference_id=reference.reference_id,
                position=reference.position,
                text=reference.text,
                provider=enrichment.citation_source,
                evidence_url=reference.evidence_url,
            )
        )
    graph = CitationGraph(
        run_id=run_id,
        nodes=sorted(nodes, key=lambda node: node.node_id),
        edges=[
            GraphEdge(
                edge_id=f"cites:{_digest(src, dst)}",
                source_id=src,
                target_id=dst,
                evidence=sorted(items, key=lambda item: (item.position, item.reference_id, item.evidence_url)),
            )
            for (src, dst), items in sorted(evidence.items())
        ],
        warnings=sorted(warnings),
    )
    graph.validate_integrity()
    return graph
