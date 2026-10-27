"""Context assembly: choose the passages that fit the budget and render them for the prompt."""

from __future__ import annotations

import re
from typing import List, Sequence

from app.core.models import SearchHit

# Passage delimiters must not be forgeable from inside a passage.
_DELIMITER = re.compile(r"<(/?)\s*passage", re.IGNORECASE)


def sanitize(text: str) -> str:
    """Neutralize anything that looks like our passage tags (prompt-injection hygiene)."""
    return _DELIMITER.sub(lambda m: f"<\u200b{m.group(1)}passage", text)


def select_within_budget(hits: Sequence[SearchHit], budget_chars: int) -> List[SearchHit]:
    """Keep hits best-first while their text fits the budget.

    The best hit is always kept, even if it alone exceeds the budget: one passage is the minimum
    useful context, and chunks are bounded by the chunk size anyway.
    """
    selected: List[SearchHit] = []
    used = 0
    for hit in hits:
        size = len(hit.chunk.text)
        if selected and used + size > budget_chars:
            break
        selected.append(hit)
        used += size
    return selected


def render_passages(hits: Sequence[SearchHit]) -> str:
    blocks = [f'<passage id="{i}">\n{sanitize(h.chunk.text)}\n</passage>' for i, h in enumerate(hits, 1)]
    return "\n\n".join(blocks)
