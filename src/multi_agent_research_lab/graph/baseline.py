"""Single-agent baseline.

One prompt, one call, no retrieval - the control group the multi-agent workflow is
measured against. It shares `ResearchState` so both runs feed the same benchmark code.
"""

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import LLMError
from multi_agent_research_lab.core.schemas import AgentName, RunStatus
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import Tracer
from multi_agent_research_lab.services.llm_client import LLMClient, get_llm_client

SYSTEM = """ROLE: baseline
You are a single assistant answering a research question end to end.
Structure: '## Answer', '## Key points', '## Limitations'.
You have no retrieval tool, so never fabricate citations or URLs."""


class SingleAgentBaseline:
    """Single-agent control run."""

    name = "baseline"

    def __init__(
        self,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm if llm is not None else get_llm_client(self.settings)
        self.tracer = tracer or Tracer(run_name="baseline")

    def run(self, state: ResearchState) -> ResearchState:
        state.status = RunStatus.RUNNING
        state.add_trace_event("baseline.start", {"query": state.request.query})
        with self.tracer.span("baseline", {"model": self.llm.model}):
            try:
                response = self.llm.complete(
                    SYSTEM,
                    (
                        f"Question: {state.request.query}\n"
                        f"Audience: {state.request.audience}\n\n"
                        "Answer now."
                    ),
                    temperature=0.3,
                )
            except LLMError as exc:
                state.add_error(f"baseline: LLM call failed ({exc})")
                state.final_answer = (
                    "## Answer\nThe single-agent baseline could not reach the LLM provider.\n"
                )
                state.finish(RunStatus.FAILED)
                return state

        state.usage.add(
            input_tokens=response.input_tokens or 0,
            output_tokens=response.output_tokens or 0,
            cost_usd=response.cost_usd or 0.0,
        )
        state.final_answer = response.content
        state.record_route(AgentName.BASELINE.value)
        state.add_agent_result(
            AgentName.BASELINE,
            response.content,
            {"model": response.model, "tokens": response.total_tokens},
        )
        state.finish(RunStatus.COMPLETED)
        return state
