"""Batch Scholar first, then explicit per-paper OpenAlex fallback without losing diagnostics."""

from datetime import UTC, datetime

from autographrs.enrichment.citations import CitationLookup, SemanticScholarCitationResolver
from autographrs.enrichment.openalex import OpenAlexCitationResolver
from autographrs.models import CitationProviderReport, Paper


class CitationService:
    """Keep useful Scholar bibliographies; fall back only when none could be retrieved."""

    def __init__(
        self,
        scholar: SemanticScholarCitationResolver | None = None,
        openalex: OpenAlexCitationResolver | None = None,
    ) -> None:
        self.scholar = scholar or SemanticScholarCitationResolver()
        self.openalex = openalex or OpenAlexCitationResolver()

    @staticmethod
    def _report(outcome: CitationLookup) -> CitationProviderReport:
        return CitationProviderReport(
            source=outcome.source, status=outcome.status, error=outcome.error, fetched_at=outcome.fetched_at
        )

    def fetch_papers(self, papers: list[Paper]) -> dict[str, CitationLookup]:
        """One bulk Scholar retrieval; failures do not restart its retry loop per paper."""
        primary = self.scholar.fetch_papers(papers)
        results: dict[str, CitationLookup] = {}
        for paper in papers:
            outcome = primary[paper.paper_id]
            reports = [self._report(outcome)]
            identities = list(outcome.provider_identities)
            authorships = outcome.authorships
            if not outcome.bibliography and outcome.status != "disabled":
                secondary = self.openalex.fetch_for_paper(paper)
                if secondary.authorships is not None:
                    authorships = [*(authorships or []), *secondary.authorships]
                reports.append(self._report(secondary))
                if secondary.provider_paper_id:
                    identities.append(secondary.provider_paper_id)
                if secondary.bibliography or (not outcome.references and secondary.status in {"ok", "partial"}):
                    outcome = secondary
            results[paper.paper_id] = outcome.model_copy(
                update={
                    "attempts": reports,
                    "provider_identities": list(dict.fromkeys(identities)),
                    "authorships": authorships,
                },
                deep=True,
            )
        return results

    def fetch_references_and_citations(self, paper_id: str) -> CitationLookup:
        """Compatibility entrypoint for one arXiv identifier; use fetch_papers for other sources."""
        now = datetime.now(UTC)
        paper = Paper(
            paper_id=paper_id,
            title="",
            abstract="",
            published_at=now,
            updated_at=now,
            html_url=f"https://arxiv.org/abs/{paper_id}",
        )
        return self.fetch_papers([paper])[paper_id]
