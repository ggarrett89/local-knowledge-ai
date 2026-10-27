"""Checks on the model's reply. The model is never trusted to produce references.

- citation numbers must point at passages that were actually given;
- page numbers and URLs the model wrote itself are removed unless they can be verified;
- an explicit refusal token is recognised, and an "Inference:" section is split off.
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

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


# --- grounding: does the cited passage actually contain what the statement says? ---------------------------

_GROUP = re.compile(r"((?:\s*\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\])+)")
_WORD = re.compile(r"[A-Za-z\u00C0-\u00FF0-9_]+(?:[-./][A-Za-z\u00C0-\u00FF0-9_]+)*")
_STOP = frozenset("""
the and for are but not you your with from that this these those they them their there here have has had was were
been being will would should could can may might must shall into onto over under than then when where which while
what who whom whose how why also such only just like each any all some more most other another about after before
between through during without within using used use uses does done doing make makes made get gets got per via
para com uma uns umas dos das nos nas que por mais como mas seu sua seus suas pelo pela pelos pelas entre sobre
quando onde qual quais ser são foi eram está estão pode podem deve devem também apenas cada outro outra
""".split())
_MIN_WEIGHT = 3      # statements with fewer distinctive words than this are not judged
SUPPORT_THRESHOLD = 0.5   # the passage clearly contains the statement
WEAK_THRESHOLD = 0.25     # paraphrases score lower; below this a citation is essentially unsupported
BETTER_MARGIN = 0.25      # another passage must beat the cited one by this much to take the citation over


class Grounding(NamedTuple):
    text: str
    cited: List[int]       # passage numbers still cited, in order of first citation
    repaired: int          # citations re-attributed to the passage that really contains the statement
    unsupported: int       # statements found in none of the passages (their citation was removed)


def _is_code_like(token: str) -> bool:
    return bool(re.search(r"[-_./0-9]", token))


def claim_tokens(claim: str) -> Dict[str, int]:
    """Distinctive words of a statement. Identifiers and commands (ansible-vault, group_vars/all, file.yml)
    weigh double: they are what a wrong attribution gets wrong."""
    tokens: Dict[str, int] = {}
    claim = _PAGE.sub(" ", _URL.sub(" ", claim))  # references are judged separately, not as content
    for match in _WORD.finditer(claim):
        token = match.group(0).lower().strip("-./_")
        if len(token) < 3 or token in _STOP or token.isdigit():
            continue
        tokens[token] = 2 if _is_code_like(token) else 1
    return tokens


class PassageIndex:
    def __init__(self, text: str) -> None:
        self.text = text.lower()
        words = {m.group(0).lower().strip("-./_") for m in _WORD.finditer(text)}
        self.words = words
        self.stems = {w[:6] for w in words if len(w) >= 6}

    def has(self, token: str) -> bool:
        if token in self.words:
            return True
        if _is_code_like(token):
            return token in self.text
        return len(token) >= 6 and token[:6] in self.stems


def support(tokens: Dict[str, int], passage: PassageIndex) -> float:
    total = sum(tokens.values())
    return sum(w for t, w in tokens.items() if passage.has(t)) / total if total else 0.0


def ground_citations(text: str, passages: Sequence[str]) -> Grounding:
    """Check each cited statement against the passages and fix attribution.

    A statement is the text since the previous citation. Per statement, in this order:
    1. cited passages that clearly contain it (support >= 0.5) are kept, unless another passage beats the
       best of them by BETTER_MARGIN, in which case the citation moves there (counted as repaired);
    2. else, if another passage clearly contains it, the citation moves there (counted as repaired);
    3. else, cited passages that partly match (support >= 0.25, typical of paraphrase) are kept;
    4. else the citation is removed, the text stays uncited and is counted as unsupported.
    This is a lexical check: it can miss paraphrases and cannot prove a claim is correct.
    """
    index = [PassageIndex(p) for p in passages]
    pieces: List[str] = []
    cited: List[int] = []
    repaired = unsupported = 0
    position = 0
    for group in _GROUP.finditer(text):
        claim = text[position:group.start()]
        position = group.end()
        numbers = [int(n) for n in re.findall(r"\d+", group.group(1))]
        tokens = claim_tokens(claim)
        if sum(tokens.values()) >= _MIN_WEIGHT:
            scores = [support(tokens, idx) for idx in index]
            unique = [n for n in dict.fromkeys(numbers) if 1 <= n <= len(index)]
            strong = [n for n in unique if scores[n - 1] >= SUPPORT_THRESHOLD]
            best = max(range(len(index)), key=lambda i: scores[i]) if index else -1
            if strong:
                top_cited = max(scores[n - 1] for n in strong)
                if best >= 0 and (best + 1) not in strong and scores[best] >= top_cited + BETTER_MARGIN:
                    numbers, repaired = [best + 1], repaired + 1
                else:
                    repaired += len(strong) != len(unique)
                    numbers = strong
            elif best >= 0 and scores[best] >= SUPPORT_THRESHOLD:
                numbers, repaired = [best + 1], repaired + 1
            else:
                weak = [n for n in unique if scores[n - 1] >= WEAK_THRESHOLD]
                if weak:
                    numbers = weak
                else:
                    numbers, unsupported = [], unsupported + 1
        pieces.append(claim)
        if numbers:
            pieces.append(" [" + ", ".join(str(n) for n in dict.fromkeys(numbers)) + "]")
            cited.extend(n for n in numbers if n not in cited)
    pieces.append(text[position:])
    return Grounding("".join(pieces), cited, repaired, unsupported)


# --- identifiers: commands, options and file names must come from the passages --------------------------

_FENCE = re.compile(r"```[\w+-]*\n?(.*?)```", re.DOTALL)
_SPAN = re.compile(r"`([^`\n]+)`")
_TRIM = " \t$()[]{}<>,;:!?'\"*"


def code_tokens(text: str) -> List[str]:
    """Identifier-like tokens (``ansible-vault``, ``--flag``, ``group_vars/all``, ``file.yml``) that the
    model wrote in code formatting (inline backticks or fenced blocks)."""
    chunks = _FENCE.findall(text)
    chunks += _SPAN.findall(_FENCE.sub(" ", text))
    found: List[str] = []
    for chunk in chunks:
        for raw in chunk.split():
            token = raw.strip(_TRIM).rstrip("/.")
            if len(token) >= 3 and _is_code_like(token) and not token.replace(".", "").isdigit():
                if token not in found:
                    found.append(token)
    return found


def find_unverified_identifiers(text: str, passages_text: str) -> List[str]:
    """Code-like tokens in the model's text that appear in none of the passages (possibly invented)."""
    haystack = passages_text.lower()
    return [t for t in code_tokens(text) if t.lower() not in haystack]
