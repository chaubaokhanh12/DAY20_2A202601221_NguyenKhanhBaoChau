"""Command-line entrypoint.

Commands:
    doctor       show which providers/engine will actually be used
    baseline     single-agent control run
    multi-agent  supervisor + researcher + analyst + writer (+ critic)
    benchmark    run both over the config queries and write reports/benchmark_report.md
"""

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import (
    DEFAULT_CONFIG_PATH,
    Engine,
    LLMProvider,
    get_settings,
    load_lab_config,
)
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import compare_runners
from multi_agent_research_lab.evaluation.quality import heuristic_quality
from multi_agent_research_lab.evaluation.report import render_full_report
from multi_agent_research_lab.graph.baseline import SingleAgentBaseline
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow, langgraph_available
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import Tracer
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI", no_args_is_help=True)
console = Console()

QueryOption = Annotated[str, typer.Option("--query", "-q", help="Research query")]


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(
    query: str, max_sources: int = 5, audience: str = "technical learners"
) -> ResearchQuery:
    try:
        return ResearchQuery(query=query, max_sources=max_sources, audience=audience)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _print_state(state: ResearchState, title: str, show_trace: bool) -> None:
    console.print(Panel(Markdown(state.final_answer or "(no answer)"), title=title))

    table = Table(title="Run summary", show_header=True, header_style="bold cyan")
    for column in ("status", "iterations", "routes", "sources", "LLM calls", "tokens", "cost USD"):
        table.add_column(column)
    table.add_row(
        state.status.value,
        str(state.iteration),
        " -> ".join(state.route_history),
        str(len(state.sources)),
        str(state.usage.llm_calls),
        str(state.usage.input_tokens + state.usage.output_tokens),
        f"{state.usage.cost_usd:.5f}",
    )
    console.print(table)

    quality = heuristic_quality(state)
    console.print(
        f"[bold]Quality (heuristic 0-10):[/bold] {quality.score} [dim]{quality.breakdown}[/dim]"
    )
    if state.errors:
        console.print(Panel("\n".join(f"- {e}" for e in state.errors), title="Errors", style="red"))
    if show_trace:
        trace = Table(title="Trace", show_header=True, header_style="bold magenta")
        trace.add_column("#")
        trace.add_column("event")
        trace.add_column("payload", overflow="fold")
        for index, event in enumerate(state.trace, start=1):
            trace.add_row(str(index), str(event["name"]), str(event["payload"]))
        console.print(trace)


def _save_artifacts(state: ResearchState, tracer: Tracer, prefix: str) -> None:
    store = LocalArtifactStore()
    trace_path = store.write_text(
        f"traces/{prefix}_trace.json", json.dumps(tracer.to_dict(), indent=2, ensure_ascii=False)
    )
    answer_path = store.write_text(f"answers/{prefix}_answer.md", state.final_answer or "")
    state_path = store.write_text(f"states/{prefix}_state.json", state.model_dump_json(indent=2))
    console.print(f"[dim]Saved {trace_path}, {answer_path}, {state_path}[/dim]")


@app.command()
def doctor() -> None:
    """Show the effective configuration (which provider/engine a run will use)."""

    _init()
    settings = get_settings()
    table = Table(title="Effective configuration", header_style="bold cyan")
    table.add_column("Setting")
    table.add_column("Value")
    rows = {
        "LLM provider": settings.resolved_llm_provider().value,
        "LLM model": settings.openai_model if settings.has_openai else "mock",
        "Search provider": settings.resolved_search_provider().value,
        "Engine": (
            settings.engine.value
            if settings.engine is not Engine.AUTO
            else ("langgraph" if langgraph_available() else "sequential")
        ),
        "langgraph installed": str(langgraph_available()),
        "Critic enabled": str(settings.enable_critic),
        "max_iterations": str(settings.max_iterations),
        "timeout_seconds": str(settings.timeout_seconds),
        "retry_attempts": str(settings.retry_attempts),
        "max_cost_usd": str(settings.max_cost_usd),
    }
    for key, value in rows.items():
        table.add_row(key, value)
    console.print(table)
    if not settings.has_openai:
        console.print(
            "[yellow]No OPENAI_API_KEY: running fully offline with deterministic mocks.[/yellow]"
        )


@app.command()
def baseline(
    query: QueryOption,
    trace: Annotated[bool, typer.Option("--trace", help="Print the trace table")] = False,
    save: Annotated[bool, typer.Option("--save", help="Write artifacts to reports/")] = False,
) -> None:
    """Run the single-agent baseline."""

    _init()
    tracer = Tracer(run_name="baseline")
    state = ResearchState(request=_parse_query(query))
    state = SingleAgentBaseline(tracer=tracer).run(state)
    _print_state(state, "Single-Agent Baseline", trace)
    if save:
        _save_artifacts(state, tracer, "baseline")


@app.command("multi-agent")
def multi_agent(
    query: QueryOption,
    max_sources: Annotated[int, typer.Option("--max-sources", min=1, max=20)] = 5,
    audience: Annotated[str, typer.Option("--audience")] = "technical learners",
    critic: Annotated[bool, typer.Option("--critic/--no-critic")] = True,
    engine: Annotated[
        Engine, typer.Option("--engine", help="langgraph | sequential | auto")
    ] = Engine.AUTO,
    trace: Annotated[bool, typer.Option("--trace", help="Print the trace table")] = False,
    save: Annotated[bool, typer.Option("--save", help="Write artifacts to reports/")] = False,
) -> None:
    """Run the multi-agent workflow."""

    _init()
    tracer = Tracer(run_name="multi_agent")
    state = ResearchState(request=_parse_query(query, max_sources, audience))
    workflow = MultiAgentWorkflow(tracer=tracer, enable_critic=critic, engine=engine)
    state = workflow.run(state)
    _print_state(state, f"Multi-Agent ({workflow.engine.value})", trace)
    if save:
        _save_artifacts(state, tracer, "multi_agent")


@app.command()
def benchmark(
    config: Annotated[Path, typer.Option("--config", help="YAML lab config")] = DEFAULT_CONFIG_PATH,
    query: Annotated[
        list[str] | None, typer.Option("--query", "-q", help="Override config queries")
    ] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("reports/benchmark_report.md"),
    engine: Annotated[Engine, typer.Option("--engine")] = Engine.AUTO,
    critic: Annotated[bool, typer.Option("--critic/--no-critic")] = True,
) -> None:
    """Benchmark single-agent vs multi-agent and write a markdown report."""

    _init()
    settings = get_settings()
    lab_config = load_lab_config(config)
    queries = list(query) if query else lab_config.benchmark.queries
    if not queries:
        console.print(Panel.fit("No benchmark queries configured.", style="red"))
        raise typer.Exit(code=1)

    def run_single(text: str) -> ResearchState:
        return SingleAgentBaseline().run(ResearchState(request=ResearchQuery(query=text)))

    def run_multi(text: str) -> ResearchState:
        workflow = MultiAgentWorkflow(enable_critic=critic, engine=engine)
        return workflow.run(ResearchState(request=ResearchQuery(query=text)))

    with console.status(f"Running {len(queries)} queries x 2 runners..."):
        summary, per_run, samples = compare_runners(
            {"single_agent": run_single, "multi_agent": run_multi}, queries
        )

    context = {
        "LLM provider": settings.resolved_llm_provider().value,
        "Search provider": settings.resolved_search_provider().value,
        "Engine": MultiAgentWorkflow(enable_critic=critic, engine=engine).engine.value,
        "Critic": str(critic),
        "Quality scorer": "heuristic rubric (deterministic)",
    }
    if settings.resolved_llm_provider() is LLMProvider.MOCK:
        context["Caveat"] = (
            "offline mock model: latency and cost are not representative, and the templated "
            "writer always satisfies the structure/grounding rubric - the quality gap here "
            "measures the pipeline shape (retrieval + citation), not model skill"
        )

    report = render_full_report(
        summary, per_run=per_run, queries=queries, samples=samples, context=context
    )
    path = LocalArtifactStore(output.parent).write_text(output.name, report)
    console.print(Markdown(report))
    console.print(f"[green]Report written to {path}[/green]")


if __name__ == "__main__":
    app()
