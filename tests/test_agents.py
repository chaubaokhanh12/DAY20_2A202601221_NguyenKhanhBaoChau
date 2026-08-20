from multi_agent_research_lab.agents import AnalystAgent, CriticAgent, ResearcherAgent, WriterAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import LLMError
from multi_agent_research_lab.core.schemas import CriticVerdict, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse, MockLLMClient
from multi_agent_research_lab.services.search_client import MockSearchClient


class BrokenLLM(LLMClient):
    """Simulates a provider outage."""

    model = "broken"

    def complete(self, system_prompt: str, user_prompt: str, **kwargs: object) -> LLMResponse:
        raise LLMError("provider down")


def _researcher(settings: Settings, search: MockSearchClient) -> ResearcherAgent:
    return ResearcherAgent(llm=MockLLMClient(), settings=settings, search=search)


# ---------------------------------------------------------------------- researcher
def test_researcher_collects_sources_and_notes(
    settings: Settings, search: MockSearchClient, state: ResearchState
) -> None:
    _researcher(settings, search).run(state)

    assert len(state.sources) == state.request.max_sources
    assert state.source_refs() == ["[S1]", "[S2]", "[S3]"]
    assert state.research_notes
    assert state.usage.llm_calls == 1
    assert state.agent_results[0].metadata["num_sources"] == 3


def test_researcher_deduplicates_sources() -> None:
    duplicated = [
        SourceDocument(title="A", url="https://x.test", snippet="one"),
        SourceDocument(title="A copy", url="https://x.test", snippet="one again"),
        SourceDocument(title="B", url="https://y.test", snippet="two"),
    ]
    assert [d.title for d in ResearcherAgent._dedupe(duplicated)] == ["A", "B"]


def test_researcher_falls_back_when_llm_is_down(
    settings: Settings, search: MockSearchClient, state: ResearchState
) -> None:
    ResearcherAgent(llm=BrokenLLM(), settings=settings, search=search).run(state)

    assert state.sources, "retrieval must still happen when the LLM is down"
    assert "extractive fallback" in (state.research_notes or "")
    assert any("LLM call failed" in error for error in state.errors)


# ------------------------------------------------------------------------- analyst
def test_analyst_skips_without_research_context(settings: Settings, state: ResearchState) -> None:
    AnalystAgent(llm=MockLLMClient(), settings=settings).run(state)

    assert state.analysis_notes is None
    assert state.errors and "no research notes" in state.errors[0]


def test_analyst_produces_structured_notes(
    settings: Settings, search: MockSearchClient, state: ResearchState
) -> None:
    _researcher(settings, search).run(state)
    AnalystAgent(llm=MockLLMClient(), settings=settings).run(state)

    assert "Key claims" in (state.analysis_notes or "")
    assert state.agent_results[-1].metadata["refs_used"]


# -------------------------------------------------------------------------- writer
def test_writer_appends_resolvable_sources(
    settings: Settings, search: MockSearchClient, state: ResearchState
) -> None:
    _researcher(settings, search).run(state)
    AnalystAgent(llm=MockLLMClient(), settings=settings).run(state)
    WriterAgent(llm=MockLLMClient(), settings=settings).run(state)

    answer = state.final_answer or ""
    assert "## Answer" in answer
    assert answer.count("## Sources") == 1
    for ref in state.source_refs():
        assert ref in answer


def test_writer_consumes_critic_feedback(
    settings: Settings, search: MockSearchClient, state: ResearchState
) -> None:
    _researcher(settings, search).run(state)
    state.final_answer = "draft"
    state.critic_verdict = CriticVerdict(approved=False, issues=["cite [S3]"])

    WriterAgent(llm=MockLLMClient(), settings=settings).run(state)

    assert state.revisions == 1
    assert state.critic_verdict is None, "a rewritten draft must be reviewed again"


# -------------------------------------------------------------------------- critic
def test_critic_approves_a_well_grounded_answer(
    settings: Settings, search: MockSearchClient, state: ResearchState
) -> None:
    _researcher(settings, search).run(state)
    AnalystAgent(llm=MockLLMClient(), settings=settings).run(state)
    WriterAgent(llm=MockLLMClient(), settings=settings).run(state)
    CriticAgent(llm=MockLLMClient(), settings=settings).run(state)

    verdict = state.critic_verdict
    assert verdict is not None
    assert verdict.approved
    assert verdict.citation_coverage == 1.0


def test_critic_flags_hallucinated_citations_and_low_coverage(
    settings: Settings, search: MockSearchClient, state: ResearchState
) -> None:
    _researcher(settings, search).run(state)
    state.final_answer = "## Answer\nEverything is fine [S9]. " + "word " * 80

    CriticAgent(llm=MockLLMClient(), settings=settings).run(state)

    verdict = state.critic_verdict
    assert verdict is not None
    assert not verdict.approved
    assert any("[S9]" in issue for issue in verdict.issues)
    assert any("cited" in issue for issue in verdict.issues)
