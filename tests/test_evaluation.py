from multi_agent_research_lab.core.config import Engine, Settings
from multi_agent_research_lab.core.schemas import ResearchQuery, RunStatus
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    aggregate,
    compare_runners,
    run_benchmark,
)
from multi_agent_research_lab.evaluation.quality import heuristic_quality
from multi_agent_research_lab.graph.baseline import SingleAgentBaseline
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import MockLLMClient
from multi_agent_research_lab.services.search_client import MockSearchClient
from multi_agent_research_lab.utils.text import (
    citation_coverage,
    keyword_overlap,
    unsupported_refs,
    word_count,
)

QUERY = "Summarize production guardrails for LLM agents"


def _multi(settings: Settings) -> ResearchState:
    workflow = MultiAgentWorkflow(
        settings=settings,
        llm=MockLLMClient(),
        search=MockSearchClient(),
        engine=Engine.SEQUENTIAL,
    )
    return workflow.run(ResearchState(request=ResearchQuery(query=QUERY, max_sources=3)))


def _single(settings: Settings) -> ResearchState:
    return SingleAgentBaseline(llm=MockLLMClient(), settings=settings).run(
        ResearchState(request=ResearchQuery(query=QUERY))
    )


# ------------------------------------------------------------------- text helpers
def test_text_helpers() -> None:
    text = "claim one [S1] and claim two [S2] plus a fake [S9]"
    assert citation_coverage(text, ["[S1]", "[S2]", "[S3]"]) == round(2 / 3, 4)
    assert unsupported_refs(text, ["[S1]", "[S2]"]) == ["[S9]"]
    assert citation_coverage(text, []) == 0.0
    assert word_count("one two three") == 3
    assert keyword_overlap("production guardrails", "guardrails in production") == 1.0
    assert keyword_overlap("graphrag latency", "nothing relevant") == 0.0


# ----------------------------------------------------------------------- scoring
def test_multi_agent_scores_above_baseline(settings: Settings) -> None:
    multi = heuristic_quality(_multi(settings))
    single = heuristic_quality(_single(settings))

    assert multi.score > single.score
    assert multi.breakdown["grounding"] > single.breakdown["grounding"]
    assert 0 <= multi.score <= 10


def test_failed_run_scores_low(settings: Settings) -> None:
    state = ResearchState(request=ResearchQuery(query=QUERY))
    state.status = RunStatus.FAILED
    score = heuristic_quality(state)

    assert score.score < 3
    assert score.breakdown["reliability"] == 0.0


# --------------------------------------------------------------------- benchmark
def test_run_benchmark_collects_all_metrics(settings: Settings) -> None:
    state, metrics = run_benchmark("multi_agent", QUERY, lambda _q: _multi(settings))

    assert metrics.latency_seconds > 0
    assert metrics.failure_rate == 0.0
    assert metrics.citation_coverage == 1.0
    assert metrics.llm_calls == state.usage.llm_calls == 4
    assert metrics.route_history[-1] == "done"


def test_run_benchmark_records_a_crash_instead_of_raising() -> None:
    def boom(_query: str) -> ResearchState:
        raise RuntimeError("kaboom")

    _state, metrics = run_benchmark("broken", QUERY, boom)

    assert metrics.failure_rate == 1.0
    assert metrics.quality_score == 0.0


def test_aggregate_averages_rows(settings: Settings) -> None:
    rows = [run_benchmark(f"multi[{i}]", QUERY, lambda _q: _multi(settings))[1] for i in range(2)]
    summary = aggregate("multi_agent", rows)

    assert summary.run_name == "multi_agent"
    assert summary.notes == "mean over 2 queries"
    assert summary.quality_score == rows[0].quality_score


def test_compare_runners_produces_one_row_per_run(settings: Settings) -> None:
    summary, per_run, samples = compare_runners(
        {"single_agent": lambda _q: _single(settings), "multi_agent": lambda _q: _multi(settings)},
        [QUERY, "Research GraphRAG state-of-the-art"],
    )

    assert [row.run_name for row in summary] == ["single_agent", "multi_agent"]
    assert all(len(rows) == 2 for rows in per_run.values())
    assert samples["multi_agent"].sources
    assert (summary[1].citation_coverage or 0) > (summary[0].citation_coverage or 0)
