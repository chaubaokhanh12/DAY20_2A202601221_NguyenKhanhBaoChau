"""Multi-agent orchestration.

Two interchangeable engines share one routing policy and one set of agents:

- `langgraph`  : real `StateGraph` with a supervisor node and conditional edges.
- `sequential` : dependency-free supervisor loop (used as fallback and in CI).

Both enforce the same guardrails: iteration budget, wall-clock timeout, per-agent retry,
error capture with degraded output instead of a crash.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from multi_agent_research_lab.agents import (
    AnalystAgent,
    BaseAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.config import Engine, Settings, get_settings
from multi_agent_research_lab.core.errors import BudgetExceededError, LabError
from multi_agent_research_lab.core.schemas import Route, RunStatus
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import Tracer
from multi_agent_research_lab.services.llm_client import LLMClient, get_llm_client
from multi_agent_research_lab.services.search_client import SearchClient, get_search_client

logger = logging.getLogger(__name__)


def langgraph_available() -> bool:
    try:  # pragma: no cover - import probe
        import langgraph.graph  # noqa: F401
    except ImportError:
        return False
    return True


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
        tracer: Tracer | None = None,
        enable_critic: bool | None = None,
        engine: Engine | str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.tracer = tracer or Tracer(run_name="multi_agent")
        self.enable_critic = self.settings.enable_critic if enable_critic is None else enable_critic
        requested = Engine(engine) if engine is not None else self.settings.engine
        if requested is Engine.AUTO:
            requested = Engine.LANGGRAPH if langgraph_available() else Engine.SEQUENTIAL
        if requested is Engine.LANGGRAPH and not langgraph_available():
            logger.warning("langgraph not installed - falling back to the sequential engine.")
            requested = Engine.SEQUENTIAL
        self.engine = requested
        self._compiled: Any = None

        # Resolve providers once and inject them: one client per run, one warning, one
        # place to swap OpenAI/Tavily for the offline mocks.
        llm = llm if llm is not None else get_llm_client(self.settings)
        search = search if search is not None else get_search_client(self.settings)

        self.supervisor = SupervisorAgent(
            llm=llm, settings=self.settings, tracer=self.tracer, enable_critic=self.enable_critic
        )
        self.workers: dict[Route, BaseAgent] = {
            Route.RESEARCHER: ResearcherAgent(
                llm=llm, settings=self.settings, tracer=self.tracer, search=search
            ),
            Route.ANALYST: AnalystAgent(llm=llm, settings=self.settings, tracer=self.tracer),
            Route.WRITER: WriterAgent(llm=llm, settings=self.settings, tracer=self.tracer),
            Route.CRITIC: CriticAgent(llm=llm, settings=self.settings, tracer=self.tracer),
        }

    # ------------------------------------------------------------------- graph
    def build(self) -> Any:
        """Create and compile the LangGraph graph (compiled once per workflow instance)."""

        if self._compiled is not None:
            return self._compiled

        from langgraph.graph import END, StateGraph

        graph: Any = StateGraph(ResearchState)
        graph.add_node(Route.DONE.value, self._noop_node)
        graph.add_node("supervisor", self._supervisor_node)
        for route in self.workers:
            graph.add_node(route.value, self._worker_node(route))

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._next_route,
            {route.value: route.value for route in [*self.workers, Route.DONE]},
        )
        for route in self.workers:
            graph.add_edge(route.value, "supervisor")
        graph.add_edge(Route.DONE.value, END)
        self._compiled = graph.compile()
        return self._compiled

    def _supervisor_node(self, state: ResearchState) -> dict[str, Any]:
        with self.tracer.span("supervisor", {"iteration": state.iteration}):
            self.supervisor.run(state)
        return state.model_dump()

    def _worker_node(self, route: Route) -> Callable[[ResearchState], dict[str, Any]]:
        def node(state: ResearchState) -> dict[str, Any]:
            self._execute(route, state)
            return state.model_dump()

        return node

    @staticmethod
    def _noop_node(state: ResearchState) -> dict[str, Any]:
        return {}

    @staticmethod
    def _next_route(state: ResearchState) -> str:
        return state.route_history[-1] if state.route_history else Route.DONE.value

    # ----------------------------------------------------------------- running
    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow and return the final state."""

        state.status = RunStatus.RUNNING
        state.add_trace_event(
            "workflow.start",
            {
                "engine": self.engine.value,
                "critic": self.enable_critic,
                "max_iterations": self.settings.max_iterations,
                "timeout_seconds": self.settings.timeout_seconds,
            },
        )
        with self.tracer.span("workflow", {"engine": self.engine.value}):
            if self.engine is Engine.LANGGRAPH:
                state = self._run_langgraph(state)
            else:
                state = self._run_sequential(state)
        return self._finalize(state)

    def _run_sequential(self, state: ResearchState) -> ResearchState:
        while True:
            with self.tracer.span("supervisor", {"iteration": state.iteration}):
                self.supervisor.run(state)
            route = Route(state.route_history[-1])
            if route is Route.DONE:
                break
            try:
                self._execute(route, state)
            except BudgetExceededError as exc:
                state.add_error(str(exc))
                break
        return state

    def _run_langgraph(self, state: ResearchState) -> ResearchState:
        graph = self.build()
        try:
            raw = graph.invoke(
                state,
                config={"recursion_limit": self.settings.max_iterations * 2 + 6},
            )
        except BudgetExceededError as exc:
            state.add_error(str(exc))
            return state
        except Exception as exc:  # noqa: BLE001 - engine failure must degrade, not crash
            state.add_error(f"langgraph engine failed ({exc!r}); returning partial state")
            return state
        if isinstance(raw, ResearchState):
            return raw
        return ResearchState.model_validate(raw)

    def _execute(self, route: Route, state: ResearchState) -> ResearchState:
        """Run one worker with timeout guard, retry, and error capture."""

        elapsed = state.elapsed_seconds()
        if elapsed > self.settings.timeout_seconds:
            raise BudgetExceededError(
                f"timeout after {elapsed:.1f}s (limit {self.settings.timeout_seconds}s)"
            )

        agent = self.workers[route]
        attempts = max(1, self.settings.retry_attempts - 1)
        for attempt in range(1, attempts + 1):
            try:
                with self.tracer.span(
                    agent.name, {"iteration": state.iteration, "attempt": attempt}
                ):
                    return agent.run(state)
            except LabError as exc:
                state.add_error(f"{agent.name}: attempt {attempt} failed ({exc})")
                if attempt == attempts:
                    state.add_trace_event(f"{agent.name}.gave_up", {"attempts": attempts})
        return state

    def _finalize(self, state: ResearchState) -> ResearchState:
        """Guarantee a usable answer and a truthful status."""

        if not state.final_answer:
            state.final_answer = (
                "## Answer\nThe workflow could not produce a grounded answer within its "
                "budget.\n\n## Limitations\n"
                + ("\n".join(f"- {error}" for error in state.errors) or "- unknown failure")
                + "\n"
            )
            state.finish(RunStatus.FAILED)
        elif state.errors:
            state.finish(RunStatus.DEGRADED)
        else:
            state.finish(RunStatus.COMPLETED)
        return state
