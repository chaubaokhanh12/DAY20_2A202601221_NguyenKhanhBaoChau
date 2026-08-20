"""Benchmark: single-agent vs multi-agent on the same queries and the same metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from statistics import mean
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, RunStatus
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.quality import QualityScore, heuristic_quality
from multi_agent_research_lab.utils.text import citation_coverage, word_count

Runner = Callable[[str], ResearchState]
Scorer = Callable[[ResearchState], QualityScore]


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    scorer: Scorer | None = None,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run one query and collect latency, cost, quality, grounding, and failure signal."""

    scorer = scorer or heuristic_quality
    started = perf_counter()
    failed = False
    try:
        state = runner(query)
    except Exception as exc:  # noqa: BLE001 - a crash is a benchmark data point
        from multi_agent_research_lab.core.schemas import ResearchQuery

        state = ResearchState(request=ResearchQuery(query=query))
        state.add_error(f"runner crashed: {exc!r}")
        state.status = RunStatus.FAILED
        failed = True
    latency = perf_counter() - started

    quality = scorer(state)
    failed = failed or state.status is RunStatus.FAILED
    notes = quality.notes or ("degraded run" if state.errors else "ok")

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=state.usage.cost_usd,
        quality_score=quality.score,
        citation_coverage=citation_coverage(state.final_answer, state.source_refs()),
        failure_rate=1.0 if failed else 0.0,
        notes=notes,
        llm_calls=state.usage.llm_calls,
        total_tokens=state.usage.input_tokens + state.usage.output_tokens,
        answer_words=word_count(state.final_answer),
        num_sources=len(state.sources),
        route_history=list(state.route_history),
        quality_breakdown=quality.breakdown,
    )
    return state, metrics


def aggregate(run_name: str, metrics: Sequence[BenchmarkMetrics]) -> BenchmarkMetrics:
    """Average per-query metrics into one row per run (what the report compares)."""

    if not metrics:
        raise ValueError("aggregate() needs at least one metric")

    def avg(values: list[float | None]) -> float:
        usable = [value for value in values if value is not None]
        return round(mean(usable), 4) if usable else 0.0

    return BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(mean(m.latency_seconds for m in metrics), 4),
        estimated_cost_usd=avg([m.estimated_cost_usd for m in metrics]),
        quality_score=avg([m.quality_score for m in metrics]),
        citation_coverage=avg([m.citation_coverage for m in metrics]),
        failure_rate=avg([m.failure_rate for m in metrics]),
        notes=f"mean over {len(metrics)} queries",
        llm_calls=int(avg([float(m.llm_calls or 0) for m in metrics])),
        total_tokens=int(avg([float(m.total_tokens or 0) for m in metrics])),
        answer_words=int(avg([float(m.answer_words or 0) for m in metrics])),
        num_sources=int(avg([float(m.num_sources or 0) for m in metrics])),
    )


def compare_runners(
    runners: dict[str, Runner],
    queries: Sequence[str],
    scorer: Scorer | None = None,
) -> tuple[list[BenchmarkMetrics], dict[str, list[BenchmarkMetrics]], dict[str, ResearchState]]:
    """Run every runner over every query.

    Returns (aggregated rows, per-query rows by run, one sample state per run).
    """

    per_run: dict[str, list[BenchmarkMetrics]] = {}
    samples: dict[str, ResearchState] = {}
    for name, runner in runners.items():
        rows: list[BenchmarkMetrics] = []
        for index, query in enumerate(queries):
            state, metrics = run_benchmark(f"{name}[{index}]", query, runner, scorer)
            rows.append(metrics)
            samples.setdefault(name, state)
        per_run[name] = rows
    summary = [aggregate(name, rows) for name, rows in per_run.items()]
    return summary, per_run, samples
