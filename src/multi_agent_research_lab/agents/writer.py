"""Writer agent: final answer synthesis, citation rendering, critic-driven revision."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState

SYSTEM = """You write the final answer for the requested audience.
Structure: '## Answer' (2-4 sentences), '## Key points' (3-5 bullets), '## Limitations' (1-2 lines).
Every factual sentence carries the source marker it comes from, e.g. [S1].
Only use markers that appear in the provided context. Do not add a sources list;
it is appended automatically."""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"
    agent_name = AgentName.WRITER
    temperature = 0.4

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        context = state.analysis_notes or state.research_notes
        if not context and not state.sources:
            state.add_error("writer: no context available; answering from the question alone")

        revision = state.critic_verdict is not None and not state.critic_verdict.approved
        feedback = ""
        if revision and state.critic_verdict is not None:
            feedback = (
                "\nRevision requested. Fix these issues:\n"
                + "\n".join(f"- {issue}" for issue in state.critic_verdict.issues)
                + f"\n{state.critic_verdict.suggestions}\n"
            )

        answer = self.think(
            state,
            self.system_prompt(SYSTEM),
            (
                f"Question: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Analysis:\n{context or '(none)'}\n\n"
                f"Sources:\n{state.sources_block() or '(none)'}\n"
                f"{feedback}\n"
                "Write the final answer now."
            ),
            fallback=self._fallback_answer(state, context),
        )

        state.final_answer = self._with_sources(state, answer)
        if revision:
            state.revisions += 1
            state.critic_verdict = None  # force a fresh review of the rewritten draft
        self.record(
            state,
            state.final_answer,
            {
                "words": len(state.final_answer.split()),
                "revision": state.revisions,
                "cited_refs": [ref for ref in state.source_refs() if ref in state.final_answer],
            },
        )
        return state

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _with_sources(state: ResearchState, answer: str) -> str:
        """Append a deterministic source list so citations are always resolvable."""

        if not state.sources:
            return answer
        body = answer.split("## Sources")[0].rstrip()
        listed = "\n".join(
            f"{doc.ref} {doc.title}" + (f" - {doc.url}" if doc.url else "") for doc in state.sources
        )
        return f"{body}\n\n## Sources\n{listed}\n"

    @staticmethod
    def _fallback_answer(state: ResearchState, context: str | None) -> str:
        refs = " ".join(state.source_refs()[:3])
        return (
            f"## Answer\n{state.request.query} - synthesised without an LLM provider, from "
            f"retrieved evidence only. {refs}\n\n"
            "## Key points\n"
            + "\n".join(f"- {doc.snippet.strip()} {doc.ref}" for doc in state.sources[:5])
            + "\n\n## Limitations\nOffline fallback answer: extractive, not generated.\n"
            + ("" if context else "No analysis context was available.\n")
        )
