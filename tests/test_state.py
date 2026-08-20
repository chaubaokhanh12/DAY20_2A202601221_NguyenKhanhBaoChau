from multi_agent_research_lab.core.schemas import (
    AgentName,
    ResearchQuery,
    Route,
    RoutingDecision,
    RunStatus,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState


def test_state_records_route_and_trace() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.record_route("researcher")
    state.add_trace_event("route", {"next": "researcher"})
    assert state.iteration == 1
    assert state.route_history == ["researcher"]
    assert state.trace[0]["name"] == "route"


def test_record_decision_keeps_reason_for_the_trace() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.record_decision(RoutingDecision(route=Route.ANALYST, reason="sources ready"))

    assert state.route_history == ["analyst"]
    assert state.routing_decisions[0].iteration == 0
    assert state.trace[0]["payload"]["reason"] == "sources ready"


def test_missing_fields_drive_routing() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    assert state.missing_fields() == ["research_notes", "analysis_notes", "final_answer"]

    state.sources = [SourceDocument(title="T", snippet="S", ref="[S1]")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    assert state.missing_fields() == []


def test_sources_block_and_refs() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.sources = [
        SourceDocument(title="T1", url="https://a.test", snippet="S1", ref="[S1]"),
        SourceDocument(title="T2", snippet="S2", ref="[S2]"),
    ]
    block = state.sources_block()

    assert state.source_refs() == ["[S1]", "[S2]"]
    assert "[S1] T1 (https://a.test)" in block
    assert "[S2] T2" in block


def test_usage_accumulates_and_finish_sets_status() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.usage.add(input_tokens=100, output_tokens=50, cost_usd=0.001)
    state.usage.add(input_tokens=100, output_tokens=50, cost_usd=0.001)
    state.add_agent_result(AgentName.WRITER, "answer", {"words": 1})
    state.finish(RunStatus.COMPLETED)

    assert state.usage.llm_calls == 2
    assert state.usage.cost_usd == 0.002
    assert state.status is RunStatus.COMPLETED
    assert state.finished_at is not None
    assert state.trace[-1]["name"] == "workflow.finish"
