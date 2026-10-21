"""Paper source provider interfaces and implementations."""

from autographrs.sources.arxiv import ArxivSource
from autographrs.sources.base import PaperSource

__all__ = ["ArxivSource", "PaperSource"]
