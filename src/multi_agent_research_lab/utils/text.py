"""Text helpers shared by the critic and the evaluation package."""

import re

REF_PATTERN = re.compile(r"\[S\d+\]")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9_\-']+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vs",
    "what",
    "why",
    "with",
}


def extract_refs(text: str | None) -> set[str]:
    """All `[S#]` citation markers present in a text."""

    return set(REF_PATTERN.findall(text or ""))


def citation_coverage(text: str | None, available_refs: list[str]) -> float:
    """Share of retrieved sources actually cited in the text (0..1)."""

    if not available_refs:
        return 0.0
    cited = extract_refs(text)
    return round(len({ref for ref in available_refs if ref in cited}) / len(available_refs), 4)


def unsupported_refs(text: str | None, available_refs: list[str]) -> list[str]:
    """Citation markers that point at a source which was never retrieved."""

    return sorted(extract_refs(text) - set(available_refs))


def word_count(text: str | None) -> int:
    return len(_WORD_PATTERN.findall(text or ""))


def keyword_overlap(query: str, text: str | None) -> float:
    """Share of meaningful query keywords that appear in the answer (0..1)."""

    keywords = {
        word.lower()
        for word in _WORD_PATTERN.findall(query)
        if len(word) > 2 and word.lower() not in _STOPWORDS
    }
    if not keywords:
        return 1.0
    lowered = (text or "").lower()
    return round(sum(1 for word in keywords if word in lowered) / len(keywords), 4)
