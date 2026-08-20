"""Supervisor / router.

Routing policy is deliberately deterministic and state-driven: the supervisor looks at
which fields are still missing and picks the single agent that can fill the next gap.
A rule-based router is cheaper, reproducible, and trivially unit-testable - the LLM is
only worth spending on the work itself, not on `if analysis_notes is None`.

Stop conditions (all enforced here, not inside workers):
- iteration budget (`max_iterations`)
- cost budget (`max_cost_usd`)
- revision budget (`max_revisions`) after critic feedback
- repeated failure of the same route (anti-oscillation)
"""

from collections import Counter

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import AgentName, Route, RoutingDecision
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import Tracer
from multi_agent_research_lab.services.llm_client import LLMClient

MAX_SAME_ROUTE = 2


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"
    agent_name = AgentName.SUPERVISOR
    temperature = 0.0

    def __init__(
        self,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        tracer: Tracer | None = None,
        enable_critic: bool | None = None,
    ) -> None:
        super().__init__(llm, settings, tracer)
        self.enable_critic = self.settings.enable_critic if enable_critic is None else enable_critic

    def decide(self, state: ResearchState) -> RoutingDecision:
        """Return the next route plus the reason (recorded for trace explanation)."""

        if state.iteration >= self.settings.max_iterations:
            return RoutingDecision(
                route=Route.DONE,
                reason=f"iteration budget reached ({self.settings.max_iterations})",
            )

        if state.usage.cost_usd > self.settings.max_cost_usd:
            return RoutingDecision(
                route=Route.DONE,
                reason=f"cost budget exceeded (${state.usage.cost_usd:.4f})",
            )

        counts = Counter(state.route_history)
        if not state.sources and not state.research_notes:
            if counts[Route.RESEARCHER.value] >= MAX_SAME_ROUTE:
                return RoutingDecision(
                    route=Route.WRITER if not state.final_answer else Route.DONE,
                    reason="researcher failed repeatedly; degrading to direct answer",
                )
            return RoutingDecision(route=Route.RESEARCHER, reason="no sources collected yet")

        if not state.analysis_notes:
            if counts[Route.ANALYST.value] >= MAX_SAME_ROUTE:
                return RoutingDecision(
                    route=Route.WRITER, reason="analyst failed repeatedly; writing from raw notes"
                )
            return RoutingDecision(route=Route.ANALYST, reason="sources ready, analysis missing")

        if not state.final_answer:
            return RoutingDecision(route=Route.WRITER, reason="analysis ready, answer missing")

        if self.enable_critic and state.critic_verdict is None:
            return RoutingDecision(route=Route.CRITIC, reason="draft ready, needs review")

        verdict = state.critic_verdict
        if (
            verdict is not None
            and not verdict.approved
            and state.revisions < self.settings.max_revisions
        ):
            return RoutingDecision(
                route=Route.WRITER,
                reason=f"critic requested revision ({'; '.join(verdict.issues) or 'quality'})",
            )

        return RoutingDecision(route=Route.DONE, reason="all required fields present")

    def run(self, state: ResearchState) -> ResearchState:
        """Record the next route in `state.route_history` / `state.routing_decisions`."""

        state.record_decision(self.decide(state))
        return state
