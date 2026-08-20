"""Shared state for the multi-agent workflow.

The state is the *only* channel agents use to hand off work. Every field below exists
because some downstream agent, guardrail, or metric needs it:

- `sources` / `research_notes`  -> Analyst input, citation validation, coverage metric.
- `analysis_notes`             -> Writer input.
- `final_answer`               -> Critic input, quality scoring, CLI output.
- `route_history` / `routing_decisions` -> loop guard + trace explanation.
- `usage`                      -> cost benchmark.
- `errors` / `status`          -> failure-rate benchmark and fallback logic.
"""

from time import time
from typing import Any

from pydantic import BaseModel, Field

from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    CriticVerdict,
    ResearchQuery,
    Route,
    RoutingDecision,
    RunStatus,
    SourceDocument,
    UsageStats,
)


class ResearchState(BaseModel):
    """Single source of truth passed through the workflow."""

    request: ResearchQuery
    iteration: int = 0
    route_history: list[str] = Field(default_factory=list)
    routing_decisions: list[RoutingDecision] = Field(default_factory=list)

    sources: list[SourceDocument] = Field(default_factory=list)
    research_notes: str | None = None
    analysis_notes: str | None = None
    final_answer: str | None = None
    critic_notes: str | None = None
    critic_verdict: CriticVerdict | None = None
    revisions: int = 0

    agent_results: list[AgentResult] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    usage: UsageStats = Field(default_factory=UsageStats)
    status: RunStatus = RunStatus.PENDING
    started_at: float = Field(default_factory=time)
    finished_at: float | None = None

    # ----------------------------------------------------------------- routing
    def record_route(self, route: str) -> None:
        """Append a route and count it as one workflow iteration."""

        self.route_history.append(route)
        self.iteration += 1

    def record_decision(self, decision: RoutingDecision) -> None:
        """Record a supervisor decision together with its justification.

        `iteration` counts *worker dispatches*, so the terminal `done` decision is logged
        in the route history but does not consume a step of the iteration budget.
        """

        decision.iteration = self.iteration
        self.routing_decisions.append(decision)
        if decision.route is Route.DONE:
            self.route_history.append(decision.route.value)
        else:
            self.record_route(decision.route.value)
        self.add_trace_event(
            "supervisor.route",
            {"route": decision.route.value, "reason": decision.reason},
        )

    # ------------------------------------------------------------------ traces
    def add_trace_event(self, name: str, payload: dict[str, Any]) -> None:
        self.trace.append({"name": name, "payload": payload})

    def add_agent_result(
        self, agent: AgentName, content: str, metadata: dict[str, Any] | None = None
    ) -> AgentResult:
        result = AgentResult(agent=agent, content=content, metadata=metadata or {})
        self.agent_results.append(result)
        return result

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.add_trace_event("error", {"message": message})

    # ------------------------------------------------------------- inspection
    def missing_fields(self) -> list[str]:
        """Fields the supervisor still needs before the run can finish."""

        missing = []
        if not self.sources and not self.research_notes:
            missing.append("research_notes")
        if not self.analysis_notes:
            missing.append("analysis_notes")
        if not self.final_answer:
            missing.append("final_answer")
        return missing

    def source_refs(self) -> list[str]:
        return [source.ref for source in self.sources if source.ref]

    def elapsed_seconds(self) -> float:
        return (self.finished_at or time()) - self.started_at

    def sources_block(self) -> str:
        """Sources rendered as prompt context with stable citation markers."""

        lines = []
        for source in self.sources:
            url = f" ({source.url})" if source.url else ""
            lines.append(f"{source.ref} {source.title}{url}\n    {source.snippet}")
        return "\n".join(lines)

    def finish(self, status: RunStatus) -> None:
        self.status = status
        self.finished_at = time()
        self.add_trace_event(
            "workflow.finish",
            {
                "status": status.value,
                "iterations": self.iteration,
                "elapsed_seconds": round(self.elapsed_seconds(), 3),
                "cost_usd": self.usage.cost_usd,
            },
        )
