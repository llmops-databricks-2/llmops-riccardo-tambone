"""Abstract interface for scientific paper sources."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from autographrs.models import Paper, SearchRequest


@runtime_checkable
class PaperSource(Protocol):
    """Protocol that every bibliographic source provider must implement."""

    source_name: str

    def search(self, request: SearchRequest) -> list[Paper]:
        """Search the index and return up to `request.num_papers` normalized papers.

        Args:
            request: Search parameters including topic, criterion, and optional value

        Returns:
            Ordered list of normalized Paper instances

        """
        ...

    def download_pdfs(self, papers: list[Paper], destination_dir: str | Path) -> list[Paper]:
        """Download PDF files for the given papers into `destination_dir`.

        Args:
            papers: Papers whose PDFs should be downloaded
            destination_dir: Target directory path (local or Unity Catalog Volume path)

        Returns:
            Updated Paper instances with `pdf_volume_path` populated when downloaded

        """
        ...
