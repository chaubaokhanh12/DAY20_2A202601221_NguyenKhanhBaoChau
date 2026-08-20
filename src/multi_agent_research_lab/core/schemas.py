"""Public schemas exchanged between CLI, agents, and evaluators."""

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field


class AgentName(StrEnum):
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"
    BASELINE = "baseline"


class Route(StrEnum):
    """Possible supervisor decisions."""

    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"
    DONE = "done"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"


class ResearchQuery(BaseModel):
    query: str = Field(..., min_length=5)
    max_sources: int = Field(default=5, ge=1, le=20)
    audience: str = "technical learners"


class AgentResult(BaseModel):
    agent: AgentName
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceDocument(BaseModel):
    """A retrieved document. `ref` is the citation marker used in generated text."""

    title: str
    url: str | None = None
    snippet: str
    ref: str = ""
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    """Explicit, auditable output of the supervisor."""

    route: Route
    reason: str
    iteration: int = 0


class UsageStats(BaseModel):
    """Token / cost accounting aggregated across every LLM call of a run."""

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, *, input_tokens: int, output_tokens: int, cost_usd: float) -> Self:
        self.llm_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd = round(self.cost_usd + cost_usd, 8)
        return self


class CriticVerdict(BaseModel):
    """Structured output of the optional critic agent."""

    approved: bool
    citation_coverage: float = Field(default=0.0, ge=0, le=1)
    issues: list[str] = Field(default_factory=list)
    suggestions: str = ""


class BenchmarkMetrics(BaseModel):
    run_name: str
    latency_seconds: float
    estimated_cost_usd: float | None = None
    quality_score: float | None = Field(default=None, ge=0, le=10)
    citation_coverage: float | None = Field(default=None, ge=0, le=1)
    failure_rate: float | None = Field(default=None, ge=0, le=1)
    notes: str = ""

    llm_calls: int | None = None
    total_tokens: int | None = None
    answer_words: int | None = None
    num_sources: int | None = None
    route_history: list[str] = Field(default_factory=list)
    quality_breakdown: dict[str, float] = Field(default_factory=dict)
