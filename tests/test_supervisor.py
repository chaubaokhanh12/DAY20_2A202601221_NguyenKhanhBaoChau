"""Routing policy is the heart of the lab, so it gets the densest unit tests."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import CriticVerdict, Route, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import MockLLMClient


def _supervisor(settings: Settings, critic: bool = True) -> SupervisorAgent:
    return SupervisorAgent(llm=MockLLMClient(), settings=settings, enable_critic=critic)


def test_routes_to_researcher_when_no_sources(settings: Settings, state: ResearchState) -> None:
    assert _supervisor(settings).decide(state).route is Route.RESEARCHER


def test_routes_through_the_expected_happy_path(settings: Settings, state: ResearchState) -> None:
    supervisor = _supervisor(settings)

    state.sources = [SourceDocument(title="t", snippet="s", ref="[S1]")]
    state.research_notes = "notes [S1]"
    assert supervisor.decide(state).route is Route.ANALYST

    state.analysis_notes = "analysis [S1]"
    assert supervisor.decide(state).route is Route.WRITER

    state.final_answer = "## Answer [S1]"
    assert supervisor.decide(state).route is Route.CRITIC

    state.critic_verdict = CriticVerdict(approved=True, citation_coverage=1.0)
    assert supervisor.decide(state).route is Route.DONE


def test_critic_can_be_disabled(settings: Settings, state: ResearchState) -> None:
    state.sources = [SourceDocument(title="t", snippet="s", ref="[S1]")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    assert _supervisor(settings, critic=False).decide(state).route is Route.DONE


def test_revision_loop_is_bounded(settings: Settings, state: ResearchState) -> None:
    supervisor = _supervisor(settings)
    state.sources = [SourceDocument(title="t", snippet="s", ref="[S1]")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    state.critic_verdict = CriticVerdict(approved=False, issues=["missing citations"])

    assert supervisor.decide(state).route is Route.WRITER

    state.revisions = settings.max_revisions
    assert supervisor.decide(state).route is Route.DONE


def test_iteration_budget_stops_the_run(settings: Settings, state: ResearchState) -> None:
    state.iteration = settings.max_iterations
    decision = _supervisor(settings).decide(state)
    assert decision.route is Route.DONE
    assert "iteration budget" in decision.reason


def test_cost_budget_stops_the_run(settings: Settings, state: ResearchState) -> None:
    state.usage.add(input_tokens=1, output_tokens=1, cost_usd=settings.max_cost_usd + 1)
    decision = _supervisor(settings).decide(state)
    assert decision.route is Route.DONE
    assert "cost budget" in decision.reason


def test_repeated_researcher_failure_degrades_instead_of_looping(
    settings: Settings, state: ResearchState
) -> None:
    state.route_history = ["researcher", "researcher"]
    state.iteration = 2
    decision = _supervisor(settings).decide(state)
    assert decision.route is Route.WRITER
    assert "degrading" in decision.reason


def test_run_records_decision_and_iteration(settings: Settings, state: ResearchState) -> None:
    _supervisor(settings).run(state)
    assert state.route_history == [Route.RESEARCHER.value]
    assert state.iteration == 1
    assert state.routing_decisions[0].reason
    assert state.trace[0]["name"] == "supervisor.route"
