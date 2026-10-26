"""Common contract for source extractors (PDF now; web and code in later stages)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol

from app.core.models import Section, SourceInfo, SourceType


class Extractor(Protocol):
    source_type: SourceType

    def describe(self, path: Path) -> SourceInfo:
        """Cheap inspection: title and page count. Raises InvalidDocument if unreadable."""
        ...

    def extract(self, path: Path) -> Iterator[Section]:
        """Yield text sections with provenance, one at a time (streaming)."""
        ...
