"""Quality scoring.

Two scorers with the same output shape:

- `heuristic_quality`: deterministic rubric (free, reproducible, used in CI).
- `LLMJudge`: optional LLM-as-judge, falls back to the heuristic when unavailable.

The rubric mirrors `docs/peer_review_rubric.md` so automated and human scores are
comparable on the same 0-10 scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from multi_agent_research_lab.core.schemas import RunStatus
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.utils.text import (
    citation_coverage,
    keyword_overlap,
    unsupported_refs,
    word_count,
)

TARGET_MIN_WORDS = 120
TARGET_MAX_WORDS = 900

WEIGHTS = {
    "completeness": 2.0,  # an answer exists and is not a failure stub
    "grounding": 3.0,  # cites retrieved sources, invents none
    "structure": 2.0,  # required sections + bullets
    "relevance": 2.0,  # covers the query keywords
    "reliability": 1.0,  # no errors recorded during the run
}


@dataclass
class QualityScore:
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict[str, float]:
        return {"score": self.score, **self.breakdown}


def heuristic_quality(state: ResearchState) -> QualityScore:
    """Score an answer 0-10 from measurable properties only."""

    answer = state.final_answer or ""
    refs = state.source_refs()
    breakdown: dict[str, float] = {}
    notes: list[str] = []

    words = word_count(answer)
    if not answer.strip():
        completeness = 0.0
    elif words < TARGET_MIN_WORDS:
        completeness = 0.6
        notes.append(f"short answer ({words} words)")
    elif words > TARGET_MAX_WORDS:
        completeness = 0.8
        notes.append(f"verbose answer ({words} words)")
    else:
        completeness = 1.0
    breakdown["completeness"] = round(completeness * WEIGHTS["completeness"], 3)

    coverage = citation_coverage(answer, refs)
    invented = unsupported_refs(answer, refs)
    grounding = coverage if refs else 0.0
    if invented:
        grounding = max(0.0, grounding - 0.5)
        notes.append(f"unsupported citations: {', '.join(invented)}")
    if not refs:
        notes.append("no retrieved sources to cite")
    breakdown["grounding"] = round(grounding * WEIGHTS["grounding"], 3)

    sections = sum(marker in answer for marker in ("## Answer", "## Key points", "## Limitations"))
    structure = sections / 3
    if "- " in answer:
        structure = min(1.0, structure + 0.1)
    breakdown["structure"] = round(structure * WEIGHTS["structure"], 3)

    relevance = keyword_overlap(state.request.query, answer)
    breakdown["relevance"] = round(relevance * WEIGHTS["relevance"], 3)

    reliability = 1.0
    if state.status is RunStatus.FAILED:
        reliability = 0.0
    elif state.errors:
        reliability = 0.5
        notes.append(f"{len(state.errors)} error(s) during the run")
    breakdown["reliability"] = round(reliability * WEIGHTS["reliability"], 3)

    total = round(min(10.0, sum(breakdown.values())), 2)
    return QualityScore(score=total, breakdown=breakdown, notes="; ".join(notes))


JUDGE_SYSTEM = """ROLE: judge
Score a research answer from 0 to 10 on: correctness of grounding, structure, relevance,
and usefulness. Reply with the number on the first line, then one short justification."""


class LLMJudge:
    """Optional LLM-as-judge; degrades to the heuristic score on any failure."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def score(self, state: ResearchState) -> QualityScore:
        heuristic = heuristic_quality(state)
        try:
            response = self.llm.complete(
                JUDGE_SYSTEM,
                f"Question: {state.request.query}\n\nAnswer:\n{state.final_answer or '(none)'}",
                temperature=0.0,
            )
            value = float(response.content.strip().splitlines()[0].split()[0])
        except Exception:  # noqa: BLE001 - the judge must never break a run
            return heuristic
        value = max(0.0, min(10.0, value))
        blended = round((value + heuristic.score) / 2, 2)
        return QualityScore(
            score=blended,
            breakdown={**heuristic.breakdown, "llm_judge": value},
            notes=f"blend(heuristic={heuristic.score}, judge={value})",
        )
