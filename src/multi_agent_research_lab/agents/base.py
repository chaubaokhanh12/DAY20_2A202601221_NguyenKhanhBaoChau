"""Base agent contract.

Every agent is a pure `state -> state` function with one responsibility. Shared plumbing
(prompting, usage accounting, tracing, deterministic fallback) lives here so each concrete
agent only contains its own domain logic.
"""

from abc import ABC, abstractmethod
from typing import Any

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import LLMError
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import Tracer
from multi_agent_research_lab.services.llm_client import LLMClient, get_llm_client


class BaseAgent(ABC):
    """Minimal interface every agent must implement."""

    name: str
    agent_name: AgentName
    temperature: float = 0.2

    def __init__(
        self,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm if llm is not None else get_llm_client(self.settings)
        self.tracer = tracer

    @abstractmethod
    def run(self, state: ResearchState) -> ResearchState:
        """Read and update shared state, then return it."""

    # ----------------------------------------------------------------- helpers
    def think(
        self,
        state: ResearchState,
        system_prompt: str,
        user_prompt: str,
        *,
        fallback: str,
        temperature: float | None = None,
    ) -> str:
        """One LLM call with usage accounting, tracing, and a deterministic fallback.

        Guardrail: a provider failure degrades the run (recorded in `state.errors`)
        instead of aborting it - the workflow still produces a usable answer.
        """

        try:
            response = self.llm.complete(
                system_prompt,
                user_prompt,
                temperature=self.temperature if temperature is None else temperature,
            )
        except LLMError as exc:
            state.add_error(f"{self.name}: LLM call failed ({exc}); used offline fallback.")
            state.add_trace_event(f"{self.name}.fallback", {"reason": str(exc)})
            return fallback

        state.usage.add(
            input_tokens=response.input_tokens or 0,
            output_tokens=response.output_tokens or 0,
            cost_usd=response.cost_usd or 0.0,
        )
        state.add_trace_event(
            f"{self.name}.llm",
            {
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "latency_seconds": round(response.latency_seconds, 4),
            },
        )
        return response.content.strip() or fallback

    def record(
        self, state: ResearchState, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        state.add_agent_result(self.agent_name, content, metadata)
        state.add_trace_event(f"{self.name}.done", metadata or {})

    def system_prompt(self, description: str) -> str:
        """System prompts always carry a `ROLE:` marker (used by the offline mock LLM)."""

        return f"ROLE: {self.name}\n{description}"
