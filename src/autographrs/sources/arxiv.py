"""ArXiv paper source implementation."""

import re
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path

import arxiv
import backoff
from loguru import logger

from autographrs.models import Paper, RankingCriteria, SearchRequest, SortOrder

_ARXIV_VERSION_SUFFIX = re.compile(r"v\d+$")
_ARXIV_ID_IN_TEXT = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")

_CRITERIA_MAP: dict[RankingCriteria, arxiv.SortCriterion] = {
    RankingCriteria.RELEVANCE: arxiv.SortCriterion.Relevance,
    RankingCriteria.SUBMITTED_DATE: arxiv.SortCriterion.SubmittedDate,
    RankingCriteria.LAST_UPDATED_DATE: arxiv.SortCriterion.LastUpdatedDate,
}

_ORDER_MAP: dict[SortOrder, arxiv.SortOrder] = {
    SortOrder.DESCENDING: arxiv.SortOrder.Descending,
    SortOrder.ASCENDING: arxiv.SortOrder.Ascending,
}


def canonical_arxiv_id(entry_id: str) -> str:
    """Extract canonical versionless arXiv identifier from an entry URL or ID.

    Args:
        entry_id: Full arXiv URL (e.g. 'http://arxiv.org/abs/2301.12345v2') or short ID

    Returns:
        Canonical arXiv ID without version suffix (e.g. '2301.12345')

    """
    short = entry_id.rstrip("/").split("/abs/")[-1]
    return _ARXIV_VERSION_SUFFIX.sub("", short)


def parse_date_bounds(criteria_date: str | date | None) -> tuple[date | None, date | None]:
    """Parse a date filter value into inclusive `(start_date, end_date)` bounds.

    Supported formats:
    - `datetime.date` instance -> `(None, d)` (papers on or before `d`)
    - `'YYYY-MM-DD'` -> single date `(None, d)` (papers on or before `d`)
    - `'YYYY-MM-DD..YYYY-MM-DD'` or `'YYYY-MM-DD:YYYY-MM-DD'` -> `(start, end)`
    - `'>=YYYY-MM-DD'` -> `(start, None)`
    - `'<=YYYY-MM-DD'` -> `(None, end)`
    - `'YYYY'` -> full calendar year `(YYYY-01-01, YYYY-12-31)`

    Args:
        criteria_date: Raw date criterion value from SearchRequest

    Returns:
        Tuple of `(start_date, end_date)`

    Raises:
        ValueError: If the date string cannot be parsed

    """
    if criteria_date is None:
        return None, None
    if isinstance(criteria_date, date) and not isinstance(criteria_date, datetime):
        return None, criteria_date

    raw = str(criteria_date).strip()
    if not raw:
        return None, None

    if ".." in raw or ":" in raw:
        sep = ".." if ".." in raw else ":"
        left, right = [part.strip() for part in raw.split(sep, maxsplit=1)]
        start = date.fromisoformat(left) if left else None
        end = date.fromisoformat(right) if right else None
        if start and end and start > end:
            raise ValueError(f"Invalid date range: start {start} is after end {end}")
        return start, end

    if raw.startswith(">="):
        return date.fromisoformat(raw[2:].strip()), None
    if raw.startswith("<="):
        return None, date.fromisoformat(raw[2:].strip())
    if re.fullmatch(r"\d{4}", raw):
        year = int(raw)
        return date(year, 1, 1), date(year, 12, 31)

    return None, date.fromisoformat(raw)


def format_arxiv_query(request: SearchRequest) -> str:
    """Build the arXiv API query string from a `SearchRequest`.

    Includes optional category filters (`cat:cs.AI`) and `submittedDate` range
    when `criteria_date` provides date bounds.

    Args:
        request: Validated search request

    Returns:
        Formatted arXiv query string

    """
    clauses: list[str] = [f"({request.topic})"]

    if request.categories:
        cat_expr = " OR ".join(f"cat:{cat.strip()}" for cat in request.categories if cat.strip())
        if cat_expr:
            clauses.append(f"({cat_expr})")

    start_date, end_date = parse_date_bounds(request.criteria_date)
    if (start_date or end_date) and request.ranking_criteria in (
        RankingCriteria.SUBMITTED_DATE,
        RankingCriteria.LAST_UPDATED_DATE,
    ):
        lower = start_date.strftime("%Y%m%d0000") if start_date else "199101010000"
        upper = end_date.strftime("%Y%m%d2359") if end_date else "209912312359"
        clauses.append(f"submittedDate:[{lower} TO {upper}]")

    return " AND ".join(clauses)


def _matches_date_bounds(
    paper: Paper,
    ranking_criteria: RankingCriteria,
    start_date: date | None,
    end_date: date | None,
) -> bool:
    """Verify whether a paper's relevant date falls within `[start_date, end_date]`."""
    if start_date is None and end_date is None:
        return True
    target_dt = paper.updated_at if ranking_criteria == RankingCriteria.LAST_UPDATED_DATE else paper.published_at
    target_day = target_dt.date()
    if start_date and target_day < start_date:
        return False
    return not (end_date and target_day > end_date)


class ArxivSource:
    """ArXiv bibliographic provider with rate-limit backoff and PDF downloading."""

    source_name: str = "arxiv"

    def __init__(self, client: arxiv.Client | None = None, download_timeout: int = 60) -> None:
        """Initialize the ArXiv source provider.

        Args:
            client: Optional pre-configured `arxiv.Client` (useful for testing)
            download_timeout: HTTP timeout in seconds when downloading PDFs

        """
        self.client = client or arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
        self.download_timeout = download_timeout

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _fetch_results(self, search: arxiv.Search) -> list[arxiv.Result]:
        """Execute the arXiv search with exponential backoff on transient failures."""
        return list(self.client.results(search))

    def search(self, request: SearchRequest) -> list[Paper]:
        """Search arXiv according to `request` and return up to `num_papers` unique papers.

        Args:
            request: Search parameters

        Returns:
            Deduplicated list of `Paper` objects ordered according to the request

        """
        start_date, end_date = parse_date_bounds(request.criteria_date)
        query_str = format_arxiv_query(request)
        sort_by = _CRITERIA_MAP[request.ranking_criteria]
        sort_order = _ORDER_MAP[request.sort_order]

        # Over-fetch slightly when client-side date filtering or deduplication applies
        max_results = request.num_papers * 2 if (start_date or end_date) else request.num_papers

        logger.info(
            "Searching arXiv: query='{}', sort_by={}, sort_order={}, max_results={}",
            query_str,
            sort_by,
            sort_order,
            max_results,
        )

        arxiv_search = arxiv.Search(
            query=query_str,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        raw_results = self._fetch_results(arxiv_search)

        papers: list[Paper] = []
        seen_ids: set[str] = set()
        for res in raw_results:
            paper = self._to_paper(res)
            if paper.paper_id in seen_ids:
                continue
            if not _matches_date_bounds(paper, request.ranking_criteria, start_date, end_date):
                continue
            seen_ids.add(paper.paper_id)
            papers.append(paper)
            if len(papers) >= request.num_papers:
                break

        logger.info("Selected {} arXiv papers for topic '{}'", len(papers), request.topic)
        return papers

    def download_pdfs(self, papers: list[Paper], destination_dir: str | Path) -> list[Paper]:
        """Download each paper's PDF into `destination_dir` and populate `pdf_volume_path`.

        Args:
            papers: List of papers to download
            destination_dir: Local or mounted Unity Catalog Volume directory

        Returns:
            Updated list of `Paper` instances

        """
        target_dir = Path(destination_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        updated_papers: list[Paper] = []
        for paper in papers:
            if not paper.pdf_url:
                logger.warning("Paper {} has no pdf_url; skipping PDF download", paper.paper_id)
                updated_papers.append(paper)
                continue

            dest_file = target_dir / f"{paper.safe_filename}.pdf"
            try:
                self._download_file(paper.pdf_url, dest_file)
                updated_papers.append(paper.model_copy(update={"pdf_volume_path": dest_file.as_posix()}))
            except Exception as exc:
                logger.warning("Failed to download PDF for {} from {}: {}", paper.paper_id, paper.pdf_url, exc)
                updated_papers.append(paper)

        return updated_papers

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _download_file(self, url: str, dest_file: Path) -> None:
        """Download a single PDF file with retry backoff."""
        req = urllib.request.Request(url, headers={"User-Agent": "autographrs/0.1"})
        with urllib.request.urlopen(req, timeout=self.download_timeout) as response:
            dest_file.write_bytes(response.read())

    @staticmethod
    def _to_paper(result: arxiv.Result) -> Paper:
        """Convert an `arxiv.Result` into our normalized `Paper` model."""
        pid = canonical_arxiv_id(result.entry_id)
        clean_title = " ".join(result.title.split())
        clean_abstract = " ".join(result.summary.split())
        published = result.published
        if published.tzinfo is None:
            published = published.replace(tzinfo=UTC)
        updated = result.updated or published
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=UTC)

        # Extract any explicit arXiv IDs mentioned in abstract or comment as initial citations
        text_for_refs = f"{clean_abstract} {result.comment or ''}"
        cited_ids = sorted({m for m in _ARXIV_ID_IN_TEXT.findall(text_for_refs) if m != pid})

        pdf_url = result.pdf_url or f"https://arxiv.org/pdf/{pid}.pdf"

        return Paper(
            paper_id=pid,
            source="arxiv",
            title=clean_title,
            authors=[author.name for author in result.authors],
            abstract=clean_abstract,
            published_at=published,
            updated_at=updated,
            categories=list(result.categories or []),
            primary_category=result.primary_category,
            doi=result.doi,
            journal_ref=result.journal_ref,
            html_url=result.entry_id,
            pdf_url=pdf_url,
            cited_paper_ids=cited_ids,
        )
