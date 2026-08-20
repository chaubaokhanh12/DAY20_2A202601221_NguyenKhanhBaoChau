import pytest

from multi_agent_research_lab.core.config import Engine, Settings
from multi_agent_research_lab.core.errors import LLMError, SearchError
from multi_agent_research_lab.core.schemas import ResearchQuery, RunStatus, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.baseline import SingleAgentBaseline
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow, langgraph_available
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse, MockLLMClient
from multi_agent_research_lab.services.search_client import MockSearchClient, SearchClient
from tests.conftest import offline_settings

HAPPY_PATH = ["researcher", "analyst", "writer", "critic", "done"]


class BrokenSearch(SearchClient):
    name = "broken"

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        raise SearchError("search provider down")


class BrokenLLM(LLMClient):
    model = "broken"

    def complete(self, system_prompt: str, user_prompt: str, **kwargs: object) -> LLMResponse:
        raise LLMError("provider down")


def _workflow(settings: Settings, engine: Engine, **kwargs: object) -> MultiAgentWorkflow:
    return MultiAgentWorkflow(
        settings=settings,
        llm=kwargs.get("llm", MockLLMClient()),  # type: ignore[arg-type]
        search=kwargs.get("search", MockSearchClient()),  # type: ignore[arg-type]
        engine=engine,
    )


def test_sequential_engine_end_to_end(settings: Settings, state: ResearchState) -> None:
    result = _workflow(settings, Engine.SEQUENTIAL).run(state)

    assert result.route_history == HAPPY_PATH
    assert result.status is RunStatus.COMPLETED
    assert result.research_notes and result.analysis_notes and result.final_answer
    assert result.usage.llm_calls == 4
    assert not result.errors


@pytest.mark.skipif(not langgraph_available(), reason="langgraph is not installed")
def test_langgraph_engine_matches_sequential(settings: Settings) -> None:
    query = ResearchQuery(query="Summarize production guardrails for LLM agents", max_sources=3)
    sequential = _workflow(settings, Engine.SEQUENTIAL).run(ResearchState(request=query))
    graph = _workflow(settings, Engine.LANGGRAPH).run(ResearchState(request=query))

    assert graph.route_history == sequential.route_history == HAPPY_PATH
    assert graph.final_answer == sequential.final_answer
    assert graph.usage.llm_calls == sequential.usage.llm_calls


def test_llm_outage_degrades_instead_of_crashing(settings: Settings, state: ResearchState) -> None:
    result = _workflow(settings, Engine.SEQUENTIAL, llm=BrokenLLM()).run(state)

    assert result.final_answer, "the run must still return something usable"
    assert result.status is RunStatus.DEGRADED
    assert result.sources, "retrieval is independent of the LLM"
    assert len(result.errors) >= 3


def test_search_outage_still_produces_an_answer(settings: Settings, state: ResearchState) -> None:
    result = _workflow(settings, Engine.SEQUENTIAL, search=BrokenSearch()).run(state)

    assert result.final_answer
    assert result.status is RunStatus.DEGRADED
    assert "researcher" in result.route_history
    assert result.route_history[-1] == "done"
    assert result.iteration <= settings.max_iterations


def test_timeout_guard_stops_the_loop(state: ResearchState) -> None:
    impatient = offline_settings(TIMEOUT_SECONDS=5)
    state.started_at -= 999  # pretend the run began long ago
    result = _workflow(impatient, Engine.SEQUENTIAL).run(state)

    assert any("timeout" in error for error in result.errors)
    assert result.status is RunStatus.FAILED
    assert result.final_answer, "a failed run still returns an explanation"


def test_iteration_budget_is_never_exceeded(state: ResearchState) -> None:
    tight = offline_settings(MAX_ITERATIONS=2)
    result = _workflow(tight, Engine.SEQUENTIAL).run(state)

    assert result.iteration <= 2
    assert result.final_answer


def test_baseline_makes_exactly_one_call(state: ResearchState, settings: Settings) -> None:
    result = SingleAgentBaseline(llm=MockLLMClient(), settings=settings).run(state)

    assert result.usage.llm_calls == 1
    assert result.route_history == ["baseline"]
    assert not result.sources, "the baseline has no retrieval step - that is the point"
    assert result.status is RunStatus.COMPLETED
