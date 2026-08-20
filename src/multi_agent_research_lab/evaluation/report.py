"""Benchmark report rendering."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

_HEADER = "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |"
_DIVIDER = "|---|---:|---:|---:|---:|---:|---|"


def _row(item: BenchmarkMetrics) -> str:
    cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
    quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
    citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
    failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
    return (
        f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
        f"| {citation} | {failure} | {item.notes} |"
    )


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a markdown table."""

    lines = ["# Benchmark Report", "", _HEADER, _DIVIDER]
    lines.extend(_row(item) for item in metrics)
    return "\n".join(lines) + "\n"


def _delta_section(summary: Sequence[BenchmarkMetrics]) -> list[str]:
    by_name = {m.run_name: m for m in summary}
    baseline = by_name.get("single_agent")
    multi = by_name.get("multi_agent")
    if baseline is None or multi is None:
        return []

    def ratio(new: float | None, old: float | None) -> str:
        if not old:
            return "n/a"
        return f"{(new or 0) / old:.2f}x"

    return [
        "## Single-agent vs multi-agent",
        "",
        "| Dimension | single_agent | multi_agent | Delta |",
        "|---|---:|---:|---:|",
        f"| Latency (s) | {baseline.latency_seconds:.2f} | {multi.latency_seconds:.2f} "
        f"| {ratio(multi.latency_seconds, baseline.latency_seconds)} |",
        f"| Cost (USD) | {baseline.estimated_cost_usd:.4f} | {multi.estimated_cost_usd:.4f} "
        f"| {ratio(multi.estimated_cost_usd, baseline.estimated_cost_usd)} |",
        f"| Tokens | {baseline.total_tokens} | {multi.total_tokens} "
        f"| {ratio(float(multi.total_tokens or 0), float(baseline.total_tokens or 0))} |",
        f"| LLM calls | {baseline.llm_calls} | {multi.llm_calls} "
        f"| {ratio(float(multi.llm_calls or 0), float(baseline.llm_calls or 0))} |",
        f"| Quality (0-10) | {baseline.quality_score:.1f} | {multi.quality_score:.1f} "
        f"| {(multi.quality_score or 0) - (baseline.quality_score or 0):+.1f} |",
        f"| Citation coverage | {baseline.citation_coverage:.0%} "
        f"| {multi.citation_coverage:.0%} "
        f"| {(multi.citation_coverage or 0) - (baseline.citation_coverage or 0):+.0%} |",
        f"| Failure rate | {baseline.failure_rate:.0%} | {multi.failure_rate:.0%} "
        f"| {(multi.failure_rate or 0) - (baseline.failure_rate or 0):+.0%} |",
        "",
    ]


def render_full_report(
    summary: Sequence[BenchmarkMetrics],
    per_run: dict[str, list[BenchmarkMetrics]] | None = None,
    queries: Sequence[str] | None = None,
    samples: dict[str, ResearchState] | None = None,
    context: dict[str, str] | None = None,
) -> str:
    """Full report: setup, aggregated comparison, per-query detail, trace, sample answer."""

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Benchmark Report - single-agent vs multi-agent",
        "",
        f"_Generated {generated}_",
        "",
        "## Setup",
        "",
    ]
    for key, value in (context or {}).items():
        lines.append(f"- **{key}**: {value}")
    if queries:
        lines.append("- **Queries**:")
        lines.extend(f"  {index + 1}. {query}" for index, query in enumerate(queries))
    lines += ["", "## Summary (mean per query)", "", _HEADER, _DIVIDER]
    lines.extend(_row(item) for item in summary)
    lines.append("")
    lines.extend(_delta_section(summary))

    if per_run:
        lines += ["## Per-query results", "", _HEADER, _DIVIDER]
        for rows in per_run.values():
            lines.extend(_row(item) for item in rows)
        lines.append("")

    if samples:
        lines += ["## Route history", ""]
        for name, state in samples.items():
            lines.append(f"- `{name}`: {' -> '.join(state.route_history) or '(none)'}")
        lines.append("")
        for name, state in samples.items():
            answer = (state.final_answer or "(no answer)").strip()
            lines += [
                f"<details><summary>Sample answer - {name}</summary>",
                "",
                answer,
                "",
                "</details>",
                "",
            ]
    return "\n".join(lines) + "\n"
