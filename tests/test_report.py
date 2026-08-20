from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.report import render_full_report, render_markdown_report


def _metrics(name: str, quality: float, coverage: float) -> BenchmarkMetrics:
    return BenchmarkMetrics(
        run_name=name,
        latency_seconds=1.23,
        estimated_cost_usd=0.0012,
        quality_score=quality,
        citation_coverage=coverage,
        failure_rate=0.0,
        llm_calls=1 if name == "single_agent" else 4,
        total_tokens=200 if name == "single_agent" else 2000,
        notes="mean over 3 queries",
    )


def test_report_renders_markdown() -> None:
    report = render_markdown_report([BenchmarkMetrics(run_name="baseline", latency_seconds=1.23)])
    assert "Benchmark Report" in report
    assert "baseline" in report


def test_full_report_contains_comparison_and_samples() -> None:
    summary = [_metrics("single_agent", 4.2, 0.0), _metrics("multi_agent", 9.1, 1.0)]
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.final_answer = "## Answer\nGrounded answer [S1]"
    state.record_route("researcher")

    report = render_full_report(
        summary,
        per_run={"multi_agent": summary},
        queries=["Explain multi-agent systems"],
        samples={"multi_agent": state},
        context={"LLM provider": "mock"},
    )

    assert "Single-agent vs multi-agent" in report
    assert "| Quality (0-10) | 4.2 | 9.1 | +4.9 |" in report
    assert "10.00x" in report  # tokens 200 -> 2000
    assert "Sample answer - multi_agent" in report
    assert "researcher" in report
