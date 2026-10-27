"""Result types for grounded answers. Sources always come from stored metadata."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

AnswerStatus = Literal["answered", "insufficient_evidence", "unverified"]


class Source(BaseModel):
    """A retrieved passage. ``number`` is the [n] the model was told to cite."""

    number: int
    reference: str
    document_id: str
    chunk_id: str
    title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    distance: float
    excerpt: str


class RagAnswer(BaseModel):
    """``answered``: the answer cites at least one valid passage.
    ``insufficient_evidence``: no passage was close enough, or the model said the passages do not answer.
    ``unverified``: the model answered without any valid citation, so nothing in it is checked.
    """

    status: AnswerStatus
    answer: str
    inference: Optional[str] = None
    sources: List[Source] = Field(default_factory=list)  # passages actually cited, from metadata
    retrieved: List[Source] = Field(default_factory=list)  # every passage the model was shown
    notice: Optional[str] = None
    removed_references: List[str] = Field(default_factory=list)  # model-written pages/URLs we could not verify
    model: Optional[str] = None
