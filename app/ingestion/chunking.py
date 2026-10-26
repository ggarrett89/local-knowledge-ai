"""Streaming text chunking with overlap and page provenance.

Sizes are in characters. Sections (pages) are consumed lazily, so memory stays bounded
by roughly one chunk plus one page regardless of document size.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator, List, NamedTuple, Optional, Tuple

from app.core.models import Section


class TextChunk(NamedTuple):
    index: int
    text: str
    page_start: Optional[int]
    page_end: Optional[int]


_SEPARATORS = ("\n\n", ". ", ".\n", "\n", " ")
_Marks = List[Tuple[int, Optional[int]]]  # (start offset in buffer, page number)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_sections(sections: Iterable[Section], *, chunk_size: int, overlap: int) -> Iterator[TextChunk]:
    if overlap < 0 or chunk_size < 2 or overlap >= chunk_size // 2:
        raise ValueError("overlap must be >= 0 and smaller than half of chunk_size")
    min_cut = max(chunk_size // 2, overlap + 1)

    buf = ""
    marks: _Marks = []
    new_start = 0  # offset in buf where text not yet emitted begins
    index = 0

    for section in sections:
        text = normalize_text(section.text)
        if not text:
            continue
        if buf:
            buf += "\n\n"
        marks.append((len(buf), section.page_number))
        buf += text

        while len(buf) > chunk_size:
            cut = _find_cut(buf, min_cut, chunk_size)
            chunk = _emit(buf, 0, cut, marks, index)
            if chunk is not None:
                yield chunk
                index += 1
            start = _overlap_start(buf, cut, overlap)
            buf, marks = buf[start:], _shift_marks(marks, start)
            new_start = cut - start

    if buf[new_start:].strip():
        chunk = _emit(buf, 0, len(buf), marks, index)
        if chunk is not None:
            yield chunk


def _find_cut(buf: str, lo: int, hi: int) -> int:
    window = buf[:hi]
    for sep in _SEPARATORS:
        pos = window.rfind(sep, lo)
        if pos != -1:
            return pos + len(sep)
    return hi


def _overlap_start(buf: str, cut: int, overlap: int) -> int:
    """Start of the next chunk: ``overlap`` chars before the cut, moved to a word start."""
    start = cut - overlap
    while 0 < start < cut and not buf[start - 1].isspace():
        start += 1
    return max(start, 1) if overlap else cut


def _emit(buf: str, a: int, b: int, marks: _Marks, index: int) -> Optional[TextChunk]:
    raw = buf[a:b]
    text = raw.strip()
    if not text:
        return None
    lead = len(raw) - len(raw.lstrip())
    first = a + lead
    last = first + len(text) - 1
    return TextChunk(index, text, _page_at(marks, first), _page_at(marks, last))


def _page_at(marks: _Marks, offset: int) -> Optional[int]:
    page: Optional[int] = marks[0][1] if marks else None
    for start, number in marks:
        if start <= offset:
            page = number
        else:
            break
    return page


def _shift_marks(marks: _Marks, start: int) -> _Marks:
    shifted: _Marks = []
    carry: Optional[Tuple[int, Optional[int]]] = None
    for offset, number in marks:
        if offset <= start:
            carry = (0, number)
        else:
            if carry is not None:
                shifted.append(carry)
                carry = None
            shifted.append((offset - start, number))
    if carry is not None:
        shifted.append(carry)
    return shifted
