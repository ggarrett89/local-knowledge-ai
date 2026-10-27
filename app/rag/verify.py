"""Checks on the model's reply. The model is never trusted to produce references.

- citation numbers must point at passages that were actually given;
- page numbers and URLs the model wrote itself are removed unless they can be verified;
- an explicit refusal token is recognised, and an "Inference:" section is split off.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Set, Tuple

from app.rag.prompt import REFUSAL_TOKEN

_CITATION = re.compile(r"(\s*)\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")
_INFERENCE = re.compile(
    r"(?im)^[ \t>*_#-]*(?:inference|inferência|inferencia)\s*(?:\*\*|__)?\s*[:：]\s*(?:\*\*|__)?"
)
_URL = re.compile(r"https?://[^\s)\]>\"']+")
_PAGE = re.compile(r"(?i)\b(?:pp?\.|pages?|págs?\.|páginas?)\s*(\d+)(?:\s*[-–]\s*(\d+))?")
_REMOVED = "[reference removed]"
_REFUSAL = re.compile(re.escape(REFUSAL_TOKEN), re.IGNORECASE)


def is_refusal(reply: str) -> bool:
    """True when the model declined: the reply starts with the token, or is essentially only the token."""
    text = reply.strip().strip("`*_ \"'.")
    if text.upper().startswith(REFUSAL_TOKEN):
        return True
    return bool(_REFUSAL.search(text)) and len(_REFUSAL.sub("", text).strip(" .:*_-\n")) < 40


def split_inference(reply: str) -> Tuple[str, Optional[str]]:
    """Separate the sourced part from a trailing "Inference:" section."""
    match = _INFERENCE.search(reply)
    if not match:
        return reply.strip(), None
    inference = reply[match.end():].strip()
    return reply[: match.start()].strip(), (inference or None)


def validate_citations(text: str, passage_count: int) -> Tuple[str, List[int], int]:
    """Drop citation numbers that do not exist. Returns (clean text, valid numbers in order, dropped count)."""
    valid: List[int] = []
    dropped = 0

    def fix(match: "re.Match[str]") -> str:
        nonlocal dropped
        kept: List[str] = []
        for raw in re.split(r"[,;]", match.group(2)):
            number = int(raw)
            if 1 <= number <= passage_count:
                kept.append(str(number))
                if number not in valid:
                    valid.append(number)
            else:
                dropped += 1
        return f"{match.group(1)}[{', '.join(kept)}]" if kept else ""

    cleaned = _CITATION.sub(fix, text)
    return cleaned.strip(), valid, dropped


def remove_unverified_references(
    text: str, passages_text: str, allowed_pages: Set[int]
) -> Tuple[str, List[str]]:
    """Remove URLs and page numbers written by the model that cannot be verified.

    A URL is kept only if it appears verbatim in a retrieved passage. A page number is kept only if
    it belongs to a page range of a cited passage (that is real metadata, not the model's invention).
    """
    removed: List[str] = []

    def fix_url(match: "re.Match[str]") -> str:
        url = match.group(0).rstrip(".,;:")
        if url in passages_text:
            return match.group(0)
        removed.append(url)
        return _REMOVED

    def fix_page(match: "re.Match[str]") -> str:
        numbers = {int(g) for g in match.groups() if g}
        if match.group(2):
            numbers = set(range(int(match.group(1)), int(match.group(2)) + 1))
        if numbers and numbers <= allowed_pages:
            return match.group(0)
        removed.append(match.group(0))
        return _REMOVED

    text = _URL.sub(fix_url, text)
    text = _PAGE.sub(fix_page, text)
    return text, removed


def strip_refusal_token(text: str) -> str:
    return _REFUSAL.sub("", text).strip()
