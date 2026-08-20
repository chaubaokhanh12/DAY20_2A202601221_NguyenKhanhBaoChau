"""Critic agent: deterministic validation first, LLM review second.

Cheap mechanical checks (hallucinated citations, coverage, structure, length) run without
an LLM call so they are reliable and free; the model is only asked for a qualitative
verdict on top. This is the guardrail that turns "looks fine" into a measurable gate.
"""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, CriticVerdict
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.utils.text import citation_coverage, unsupported_refs, word_count

SYSTEM = """You review a draft answer against its sources.
Reply with exactly two parts:
line 1: 'VERDICT: approve' or 'VERDICT: revise'
line 2+: one bullet per concrete problem (unsupported claim, missing citation, off-topic).
Be strict about claims that no source supports, but do not invent new requirements."""

MIN_WORDS = 60
MIN_COVERAGE = 0.5


class CriticAgent(BaseAgent):
    """Fact-checking and quality-gate agent."""

    name = "critic"
    agent_name = AgentName.CRITIC
    temperature = 0.0

    def run(self, state: ResearchState) -> ResearchState:
        """Validate the final answer and append findings to the state."""

        answer = state.final_answer
        if not answer:
            state.add_error("critic: nothing to review")
            return state

        refs = state.source_refs()
        coverage = citation_coverage(answer, refs)
        issues: list[str] = []

        hallucinated = unsupported_refs(answer, refs)
        if hallucinated:
            issues.append(f"citations pointing to unknown sources: {', '.join(hallucinated)}")
        if refs and coverage < MIN_COVERAGE:
            issues.append(f"only {coverage:.0%} of retrieved sources are cited (min 50%)")
        if word_count(answer) < MIN_WORDS:
            issues.append(f"answer is too short ({word_count(answer)} words)")
        if "## Answer" not in answer:
            issues.append("missing the required '## Answer' section")

        review = self.think(
            state,
            self.system_prompt(SYSTEM),
            (
                f"Question: {state.request.query}\n\n"
                f"Draft:\n{answer}\n\n"
                f"Sources:\n{state.sources_block() or '(none)'}\n\n"
                "Review the draft now."
            ),
            fallback="VERDICT: approve\n(no LLM available; mechanical checks only)",
        )
        llm_wants_revision = review.lower().startswith("verdict: revise")
        if llm_wants_revision:
            issues.extend(
                line.lstrip("-* ").strip()
                for line in review.splitlines()[1:]
                if line.strip().startswith(("-", "*"))
            )

        verdict = CriticVerdict(
            approved=not issues,
            citation_coverage=coverage,
            issues=issues,
            suggestions="Cite every retrieved source at least once and keep claims grounded."
            if issues
            else "",
        )
        state.critic_verdict = verdict
        state.critic_notes = review
        self.record(
            state,
            review,
            {
                "approved": verdict.approved,
                "citation_coverage": coverage,
                "issues": issues,
            },
        )
        return state
