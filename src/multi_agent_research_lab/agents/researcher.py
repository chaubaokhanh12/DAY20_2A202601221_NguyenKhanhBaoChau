"""Researcher agent: retrieval + note taking with citation capture."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import SearchError
from multi_agent_research_lab.core.schemas import AgentName, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import Tracer
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import (
    SearchClient,
    assign_refs,
    get_search_client,
)

SYSTEM = """You collect evidence for a research question.
Rules:
- Use ONLY the numbered sources given to you.
- Every bullet must end with its source marker, e.g. [S2].
- Max 8 bullets, one fact per bullet, no filler.
- Finish with one line starting with 'Open question:'."""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"
    agent_name = AgentName.RESEARCHER
    temperature = 0.2

    def __init__(
        self,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        tracer: Tracer | None = None,
        search: SearchClient | None = None,
    ) -> None:
        super().__init__(llm, settings, tracer)
        self.search = search if search is not None else get_search_client(self.settings)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        try:
            documents = self.search.search(state.request.query, state.request.max_sources)
        except SearchError as exc:
            state.add_error(f"researcher: search failed ({exc})")
            documents = []

        documents = self._dedupe(documents)[: state.request.max_sources]
        if not documents:
            state.add_trace_event("researcher.empty", {"query": state.request.query})
            return state

        state.sources = assign_refs(documents)
        notes = self.think(
            state,
            self.system_prompt(SYSTEM),
            (
                f"Question: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Sources:\n{state.sources_block()}\n\n"
                "Write the research notes now."
            ),
            fallback=self._extractive_notes(state),
        )
        state.research_notes = notes
        self.record(
            state,
            notes,
            {
                "num_sources": len(state.sources),
                "provider": self.search.name,
                "refs": state.source_refs(),
            },
        )
        return state

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _dedupe(documents: list[SourceDocument]) -> list[SourceDocument]:
        """Drop duplicate URLs/titles - duplicated evidence inflates citation coverage."""

        seen: set[str] = set()
        unique: list[SourceDocument] = []
        for doc in documents:
            key = (doc.url or doc.title).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(doc)
        return unique

    @staticmethod
    def _extractive_notes(state: ResearchState) -> str:
        """Deterministic fallback used when the LLM provider is unavailable."""

        bullets = [f"- {doc.snippet.strip()} {doc.ref}" for doc in state.sources]
        return (
            f"Research notes (extractive fallback) on: {state.request.query}\n"
            + "\n".join(bullets)
            + "\nOpen question: which claims are still valid at production scale?"
        )
