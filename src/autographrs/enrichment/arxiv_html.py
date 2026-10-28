"""Extract bibliography evidence from official arXiv HTML without external citation indexes."""

import hashlib
import re
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Literal
from urllib.parse import unquote, urljoin, urlsplit

from loguru import logger

from autographrs.enrichment.citations import CitationLookup
from autographrs.models import BibliographicReference
from autographrs.sources.arxiv import canonical_arxiv_id

_ARXIV_ID = r"(?:\d{4}\.\d{4,5}|[a-zA-Z][a-zA-Z.-]*(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?"
_ARXIV_TEXT = re.compile(rf"\barxiv\s*:\s*({_ARXIV_ID})(?!\w)", re.IGNORECASE)
_ARXIV_URL_TEXT = re.compile(
    rf"\b(?:https?://)?(?:www\.|export\.)?arxiv\.org/(?:abs|pdf|html)/({_ARXIV_ID})(?!\w)", re.IGNORECASE
)
_DOI = re.compile(r"\b10\.\d{4,9}/[^\s<>\"#]+", re.IGNORECASE)
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


def normalize_doi(value: str) -> str:
    """Normalize DOI identifiers and resolver URLs for exact identity matching."""
    value = unquote(value).strip()
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value, flags=re.IGNORECASE)
    value = value.rstrip(".,;:")
    while value.endswith(")") and value.count(")") > value.count("("):
        value = value[:-1]
    return value.lower()


def _identifiers(text: str, urls: list[str]) -> tuple[list[str], list[str]]:
    arxiv_ids = {canonical_arxiv_id(match) for match in _ARXIV_TEXT.findall(text)}
    arxiv_ids.update(canonical_arxiv_id(match) for match in _ARXIV_URL_TEXT.findall(text))
    dois = {normalize_doi(match.group()) for match in _DOI.finditer(text)}
    for url in urls:
        parsed = urlsplit(url)
        path = unquote(parsed.path)
        if parsed.hostname in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
            identifier = re.sub(r"^/(?:abs|pdf|html)/", "", path).removesuffix(".pdf")
            if re.fullmatch(_ARXIV_ID, identifier):
                arxiv_ids.add(canonical_arxiv_id(identifier))
        elif parsed.hostname in {"doi.org", "dx.doi.org"}:
            identifier = normalize_doi(path.lstrip("/"))
            if _DOI.fullmatch(identifier):
                dois.add(identifier)
    return sorted(arxiv_ids), sorted(dois)


class _BibliographyParser(HTMLParser):
    """Read only LaTeXML bibliography items, not mentions or links in the paper body."""

    def __init__(self, paper_id: str, document_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.paper_id = paper_id
        self.document_url = document_url
        self.entries: list[BibliographicReference] = []
        self.incomplete = False
        self._stack: list[tuple[str, bool]] = []
        self._reference_id = ""
        self._text: list[str] = []
        self._label: list[str] = []
        self._urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track bibliography item boundaries and evidence links."""
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if "ltx_bibitem" in classes:
            if self._stack:
                self._finish(incomplete=True)
            self._reference_id = attributes.get("id") or f"reference-{len(self.entries) + 1}"
            self._text, self._label, self._urls = [], [], []
            self._stack = [(tag, False)]
            return
        if not self._stack:
            return
        if tag == "a" and (href := attributes.get("href")):
            url = urljoin(self.document_url, href)
            if urlsplit(url).scheme in {"http", "https"} and not href.startswith("#"):
                self._urls.append(url)
        if tag not in _VOID_TAGS:
            self._stack.append((tag, "ltx_tag_bibitem" in classes))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle XHTML-style empty elements without leaving the item stack open."""
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        """Finish a reference only when its enclosing element closes."""
        if not self._stack:
            return
        positions = [index for index, (open_tag, _) in enumerate(self._stack) if open_tag == tag]
        if not positions:
            return
        index = positions[-1]
        if index != len(self._stack) - 1:
            self.incomplete = True
        del self._stack[index:]
        if not self._stack:
            self._finish()

    def handle_data(self, data: str) -> None:
        """Collect text and the displayed reference label without executable content."""
        if not self._stack or any(tag in {"script", "style"} for tag, _ in self._stack):
            return
        self._text.append(data)
        if any(is_label for _, is_label in self._stack):
            self._label.append(data)

    def close(self) -> None:
        """Keep truncated entries while marking extraction partial."""
        super().close()
        if self._stack:
            self._finish(incomplete=True)

    def _finish(self, incomplete: bool = False) -> None:
        self.incomplete |= incomplete
        text = " ".join(" ".join(self._text).split())
        urls = list(dict.fromkeys(self._urls))
        if not text:
            self.incomplete = True
            self._stack = []
            return
        arxiv_ids, dois = _identifiers(text, urls)
        position = len(self.entries) + 1
        method: Literal["arxiv_id", "doi", "unresolved"]
        if len(arxiv_ids) == 1 and len(dois) <= 1:
            target_id, method = f"arxiv:{arxiv_ids[0]}", "arxiv_id"
        elif not arxiv_ids and len(dois) == 1:
            target_id, method = f"doi:{dois[0]}", "doi"
        else:
            digest = hashlib.sha256(f"{self.paper_id}\0{position}\0{text}".encode()).hexdigest()
            target_id, method = f"ref:{digest}", "unresolved"
        self.entries.append(
            BibliographicReference(
                reference_id=self._reference_id,
                position=position,
                label=" ".join(" ".join(self._label).split()) or None,
                text=text,
                urls=urls,
                arxiv_ids=arxiv_ids,
                dois=dois,
                target_id=target_id,
                match_method=method,
                evidence_url=f"{self.document_url}#{self._reference_id}",
            )
        )
        self._stack = []


def parse_arxiv_bibliography(html: str, paper_id: str, document_url: str) -> CitationLookup:
    """Retain every extracted entry; successful extraction does not imply resolved identities."""
    parser = _BibliographyParser(paper_id, document_url)
    parser.feed(html)
    parser.close()
    error = None
    status: Literal["ok", "partial", "not_found"]
    if not parser.entries:
        status = "not_found"
        error = "No supported bibliography entries found in arXiv HTML; bibliography coverage is unknown"
    elif parser.incomplete:
        status = "partial"
        error = "Malformed or truncated bibliography HTML; extracted entries may be incomplete"
    else:
        status = "ok"
    if error:
        logger.warning("{}: {}", paper_id, error)
    return CitationLookup(
        references={entry.arxiv_ids[0] for entry in parser.entries if entry.match_method == "arxiv_id"},
        bibliography=parser.entries,
        status=status,
        error=error,
        source="arxiv_html",
    )


class ArxivHtmlCitationResolver:
    """Fetch official HTML with polite pacing, explicit errors and no index/PDF fallback."""

    def __init__(self, timeout: float = 30, min_interval: float = 3.0) -> None:
        if timeout <= 0 or min_interval < 0:
            raise ValueError("Invalid arXiv HTML timeout or pacing interval")
        self.timeout = timeout
        self.min_interval = min_interval
        self._last_request: float | None = None

    def fetch_references_and_citations(self, paper_id: str) -> CitationLookup:
        """Retrieve the bibliography, including references outside arXiv and the current run."""
        if not re.fullmatch(_ARXIV_ID, paper_id):
            raise ValueError(f"Invalid arXiv identifier: {paper_id}")
        if self._last_request is not None:
            time.sleep(max(0.0, self.min_interval - (time.monotonic() - self._last_request)))
        self._last_request = time.monotonic()
        url = f"https://arxiv.org/html/{paper_id}"
        request = urllib.request.Request(url, headers={"User-Agent": "autographrs/0.1", "Accept": "text/html"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                html = response.read().decode("utf-8")
                document_url = response.geturl()
        except (urllib.error.URLError, TimeoutError, UnicodeDecodeError) as exc:
            status: Literal["error", "not_found", "rate_limited"] = "error"
            if isinstance(exc, urllib.error.HTTPError):
                status = "not_found" if exc.code in {404, 410} else "rate_limited" if exc.code == 429 else "error"
                error = f"arXiv HTML HTTP {exc.code}; no citation fallback was used"
            else:
                error = f"arXiv HTML {type(exc).__name__}; no citation fallback was used"
            logger.warning("{}: {}", paper_id, error)
            return CitationLookup(source="arxiv_html", status=status, error=error)
        return parse_arxiv_bibliography(html, paper_id, document_url)
