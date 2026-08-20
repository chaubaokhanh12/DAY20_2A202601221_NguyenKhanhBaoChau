"""Analyst agent: turns raw notes into structured, decision-ready insight."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState

SYSTEM = """You analyse research notes for a technical decision.
Produce exactly these sections:
Key claims: 2-4 numbered claims, each ending with its source marker(s).
Tensions: where sources disagree (or 'none observed').
Evidence gaps: what the sources do NOT cover.
Confidence: low | medium | high + one clause of justification.
Never invent a source marker that is not in the input."""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"
    agent_name = AgentName.ANALYST
    temperature = 0.1

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        if not state.research_notes and not state.sources:
            state.add_error("analyst: no research notes to analyse")
            state.add_trace_event("analyst.skipped", {"reason": "empty research context"})
            return state

        analysis = self.think(
            state,
            self.system_prompt(SYSTEM),
            (
                f"Question: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Research notes:\n{state.research_notes or '(none)'}\n\n"
                f"Sources:\n{state.sources_block()}\n\n"
                "Write the analysis now."
            ),
            fallback=self._structured_fallback(state),
        )
        state.analysis_notes = analysis
        self.record(
            state,
            analysis,
            {"claims": analysis.count("\n"), "refs_used": self._refs_used(state, analysis)},
        )
        return state

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _refs_used(state: ResearchState, text: str) -> list[str]:
        return [ref for ref in state.source_refs() if ref in text]

    @staticmethod
    def _structured_fallback(state: ResearchState) -> str:
        claims = [
            f"{index}. {doc.title}: {doc.snippet.strip()} {doc.ref}"
            for index, doc in enumerate(state.sources[:4], start=1)
        ]
        return (
            "Key claims:\n"
            + ("\n".join(claims) or "1. No source available.")
            + "\nTensions: not assessed (offline fallback).\n"
            "Evidence gaps: no quantitative comparison retrieved.\n"
            "Confidence: low - produced without an LLM call."
        )
