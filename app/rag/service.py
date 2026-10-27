"""Grounded answers: retrieve, build a bounded context, ask the model, then verify its reply.

The model only writes prose. Which passages support the answer, and every reference shown to the
user, come from the stored metadata of the passages that were actually retrieved.
"""

from __future__ import annotations

import logging
import re
import time
from typing import List, Optional, Sequence, Set

from app.core.interfaces import ChatClient
from app.core.models import SearchFilters, SearchHit
from app.core.references import format_reference
from app.rag import verify
from app.rag.context import render_passages, select_within_budget
from app.rag.models import RagAnswer, Source
from app.rag.prompt import build_messages
from app.retrieval.service import Retriever

logger = logging.getLogger(__name__)

INSUFFICIENT_MESSAGE = "The indexed sources do not contain enough information to answer this question."
_EXCERPT_CHARS = 300


class RagService:
    def __init__(self, retriever: Retriever, chat: ChatClient, *, context_chars: int = 6000) -> None:
        self._retriever = retriever
        self._chat = chat
        self._context_chars = context_chars

    def answer(
        self,
        question: str,
        *,
        k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
        max_distance: Optional[float] = None,
    ) -> RagAnswer:
        started = time.perf_counter()
        hits = self._retriever.search(question, k=k, filters=filters, max_distance=max_distance)
        if not hits:
            # Nothing close enough: do not ask the model to answer from nothing.
            logger.info("rag status=insufficient_evidence reason=no_passages")
            return RagAnswer(
                status="insufficient_evidence",
                answer=INSUFFICIENT_MESSAGE,
                notice="No passage was close enough to the question (or the filters matched no document).",
                model=self._chat.model,
            )

        selected = select_within_budget(hits, self._context_chars)
        retrieved = [self._source(i, h) for i, h in enumerate(selected, 1)]
        question = question.strip()
        reply = self._chat.complete(build_messages(question, render_passages(selected)))
        result = self._interpret(reply, selected, retrieved)
        logger.info(
            "rag status=%s passages=%d cited=%d removed_refs=%d seconds=%.2f",
            result.status, len(retrieved), len(result.sources), len(result.removed_references),
            time.perf_counter() - started,
        )
        return result

    # -- internals -----------------------------------------------------------------

    def _interpret(self, reply: str, selected: Sequence[SearchHit], retrieved: List[Source]) -> RagAnswer:
        model = self._chat.model
        if verify.is_refusal(reply):
            return RagAnswer(
                status="insufficient_evidence", answer=INSUFFICIENT_MESSAGE, retrieved=retrieved,
                notice="The retrieved passages were judged insufficient to answer the question.", model=model,
            )

        body, inference = verify.split_inference(verify.strip_refusal_token(reply))
        body, cited, dropped = verify.validate_citations(body, len(selected))
        grounding = verify.ground_citations(body, [h.chunk.text for h in selected])
        body, cited = grounding.text, grounding.cited
        passages_text = "\n".join(h.chunk.text for h in selected)
        allowed_pages = self._pages_of([selected[n - 1] for n in cited])
        body, removed = verify.remove_unverified_references(body, passages_text, allowed_pages)
        if inference:
            inference, _, _ = verify.validate_citations(inference, len(selected))
            inference, removed_in_inference = verify.remove_unverified_references(inference, passages_text, set())
            removed += removed_in_inference

        unverified = verify.find_unverified_identifiers(f"{body}\n{inference or ''}", passages_text)

        notices: List[str] = []
        if dropped:
            notices.append(f"{dropped} citation number(s) pointed at passages that do not exist and were removed.")
        if grounding.repaired:
            notices.append(
                f"{grounding.repaired} citation(s) pointed at a passage that does not contain the statement "
                "and were moved to the passage that does."
            )
        if grounding.unsupported:
            notices.append(
                f"{grounding.unsupported} statement(s) could not be found in any retrieved passage; "
                "their citation was removed. Treat them as unverified."
            )
        if removed:
            notices.append("Page numbers or URLs written by the model that could not be verified were removed.")
        if unverified:
            listed = ", ".join(f"`{t}`" for t in unverified[:8])
            notices.append(f"These commands, options or file names appear in none of the retrieved passages "
                           f"and may be invented by the model: {listed}.")

        if not body.strip():
            notices.append("The model gave no statement based on the passages.")
            return RagAnswer(
                status="insufficient_evidence", answer=INSUFFICIENT_MESSAGE, inference=None, retrieved=retrieved,
                notice=" ".join(notices), removed_references=removed, model=model,
            )
        if not cited:
            notices.append(
                "The answer cites none of the retrieved passages, so it could not be checked against the sources."
            )
            status = "unverified"
        else:
            status = "answered"
        return RagAnswer(
            status=status, answer=body, inference=inference,
            sources=[retrieved[n - 1] for n in cited], retrieved=retrieved,
            notice=" ".join(notices) or None, removed_references=removed,
            unverified_identifiers=unverified, model=model,
        )

    @staticmethod
    def _pages_of(hits: Sequence[SearchHit]) -> Set[int]:
        pages: Set[int] = set()
        for h in hits:
            p = h.chunk.provenance
            if p.page_start is not None:
                pages.update(range(p.page_start, (p.page_end or p.page_start) + 1))
        return pages

    @staticmethod
    def _source(number: int, hit: SearchHit) -> Source:
        p = hit.chunk.provenance
        excerpt = re.sub(r"\s+", " ", hit.chunk.text).strip()
        return Source(
            number=number,
            reference=format_reference(p),
            document_id=p.document_id,
            chunk_id=hit.chunk.chunk_id,
            title=p.title,
            page_start=p.page_start,
            page_end=p.page_end,
            distance=round(hit.distance, 4),
            excerpt=excerpt if len(excerpt) <= _EXCERPT_CHARS else excerpt[:_EXCERPT_CHARS].rstrip() + "...",
        )
