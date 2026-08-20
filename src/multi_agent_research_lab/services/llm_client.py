"""LLM client abstraction.

Production note: agents depend on this interface instead of importing an SDK directly.
Retry, timeout, token accounting, and cost estimation live here - never inside agents.

Two implementations ship with the lab:

- `OpenAIClient`: real provider, used automatically when `OPENAI_API_KEY` is set.
- `MockLLMClient`: deterministic offline stand-in so the whole workflow, the tests, and
  the benchmark run with zero credentials (and with reproducible output).
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import LLMProvider, Settings, get_settings
from multi_agent_research_lab.core.errors import LLMError

logger = logging.getLogger(__name__)

ROLE_MARKER = "ROLE:"
_REF_PATTERN = re.compile(r"\[S\d+\]")

# USD per 1M tokens, (input, output). Source: provider pricing pages, keep in one place.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "mock": (0.0, 0.0),
}
_DEFAULT_PRICE = (0.15, 0.60)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for one call. Unknown models fall back to gpt-4o-mini pricing."""

    price_in, price_out = PRICING_PER_MTOK.get(model, _DEFAULT_PRICE)
    return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 8)


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) used when a provider reports no usage."""

    return max(1, len(text) // 4)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model: str = "unknown"
    latency_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return (self.input_tokens or 0) + (self.output_tokens or 0)


class LLMClient(ABC):
    """Provider-agnostic LLM client."""

    model: str = "unknown"

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return a model completion."""


class OpenAIClient(LLMClient):
    """OpenAI-backed client with retry, timeout, and usage accounting."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set; use LLM_PROVIDER=mock for offline runs.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise LLMError("openai package missing. Install with: pip install -e '.[llm]'") from exc

        self.model = self.settings.openai_model
        self._client: Any = OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=float(self.settings.request_timeout_seconds),
            max_retries=0,  # retry policy is owned by this class, not the SDK
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        started = perf_counter()

        @retry(
            stop=stop_after_attempt(self.settings.retry_attempts),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            reraise=False,
        )
        def _call() -> Any:
            return self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

        try:
            completion = _call()
        except RetryError as exc:
            raise LLMError(
                f"OpenAI call failed after {self.settings.retry_attempts} attempts: "
                f"{exc.last_attempt.exception()!r}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
            raise LLMError(f"OpenAI call failed: {exc!r}") from exc

        content = (completion.choices[0].message.content or "").strip()
        usage = getattr(completion, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) or approx_tokens(
            system_prompt + user_prompt
        )
        output_tokens = getattr(usage, "completion_tokens", None) or approx_tokens(content)
        latency = perf_counter() - started
        logger.debug(
            "llm.complete model=%s in=%s out=%s latency=%.2fs",
            self.model,
            input_tokens,
            output_tokens,
            latency,
        )
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost(self.model, input_tokens, output_tokens),
            model=self.model,
            latency_seconds=latency,
        )


class MockLLMClient(LLMClient):
    """Deterministic offline LLM.

    It reads the `ROLE: <agent>` marker that every agent puts in its system prompt and
    produces a role-appropriate, citation-preserving answer built from the prompt itself.
    Deterministic on purpose: tests and benchmarks must be reproducible.
    """

    model = "mock"

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        started = perf_counter()
        role = self._role(system_prompt)
        content = self._render(role, user_prompt)
        input_tokens = approx_tokens(system_prompt + user_prompt)
        output_tokens = approx_tokens(content)
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,
            model=self.model,
            latency_seconds=perf_counter() - started,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _role(system_prompt: str) -> str:
        for line in system_prompt.splitlines():
            if ROLE_MARKER in line:
                return line.split(ROLE_MARKER, 1)[1].strip().lower()
        return "generic"

    @staticmethod
    def _question(user_prompt: str) -> str:
        for line in user_prompt.splitlines():
            if line.lower().startswith(("question:", "query:")):
                return line.split(":", 1)[1].strip()
        return user_prompt.strip().splitlines()[0] if user_prompt.strip() else "the question"

    @staticmethod
    def _refs(user_prompt: str) -> list[str]:
        seen: list[str] = []
        for ref in _REF_PATTERN.findall(user_prompt):
            if ref not in seen:
                seen.append(ref)
        return seen

    @staticmethod
    def _source_lines(user_prompt: str) -> list[tuple[str, str]]:
        """Extract `[S1] Title (url)` lines from the prompt as (ref, title) pairs."""

        pairs: list[tuple[str, str]] = []
        for raw in user_prompt.splitlines():
            line = raw.strip()
            match = _REF_PATTERN.match(line)
            if not match:
                continue
            title = line[match.end() :].strip()
            title = re.sub(r"\s*\(https?://[^)]+\)$", "", title).strip(" -—")
            if title:
                pairs.append((match.group(0), title))
        return pairs

    def _render(self, role: str, user_prompt: str) -> str:
        question = self._question(user_prompt)
        sources = self._source_lines(user_prompt)
        refs = self._refs(user_prompt) or [ref for ref, _ in sources]

        if role == "researcher":
            bullets = [f"- {title} {ref}" for ref, title in sources] or [
                "- No external source retrieved; answer relies on parametric knowledge."
            ]
            return (
                f"Research notes on: {question}\n"
                + "\n".join(bullets)
                + "\n- Open question: which of these findings still hold at production scale?"
            )

        if role == "analyst":
            joined = " ".join(refs[:3]) or "(no sources)"
            return (
                f"Key claims about {question}:\n"
                f"1. The sources converge on a staged approach, not a single tool {joined}.\n"
                f"2. Measured trade-offs (latency, cost, accuracy) differ per workload {joined}.\n"
                "Tensions: sources disagree on how much orchestration overhead is acceptable.\n"
                "Evidence gaps: little quantitative data on failure recovery.\n"
                "Confidence: medium - based on retrieved snippets only."
            )

        if role == "writer":
            cite = " ".join(refs) if refs else ""
            body = (
                f"## Answer\n{question} is best addressed by combining retrieval with an "
                f"explicit division of labour between specialised steps. {cite}\n\n"
                "## Key points\n"
                f"- Ground every claim in retrieved evidence. {refs[0] if refs else ''}\n"
                f"- Keep an explicit controller so each step has one job. "
                f"{refs[1] if len(refs) > 1 else ''}\n"
                f"- Measure latency, cost, and quality before adding agents. "
                f"{refs[2] if len(refs) > 2 else ''}\n\n"
                "## Limitations\nOffline deterministic model: wording is templated, "
                "citations are preserved from the retrieved sources.\n"
            )
            if sources:
                body += "\n## Sources\n" + "\n".join(f"{ref} {title}" for ref, title in sources)
            return body

        if role == "critic":
            return "VERDICT: approve\nNo unsupported citation detected in the draft."

        if role == "baseline":
            return (
                f"{question}\n\n"
                "Single-agent answer: the topic involves several trade-offs and the right "
                "choice depends on data freshness, budget, and latency targets. Without a "
                "retrieval step this answer cannot cite sources, which is exactly the gap "
                "the multi-agent workflow closes."
            )

        return f"[mock:{role}] {question}"


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Factory: real provider when credentials exist, deterministic mock otherwise."""

    settings = settings or get_settings()
    provider = settings.resolved_llm_provider()
    if provider is LLMProvider.OPENAI:
        return OpenAIClient(settings)
    if settings.llm_provider is LLMProvider.AUTO:
        logger.warning("No OPENAI_API_KEY found - falling back to deterministic MockLLMClient.")
    return MockLLMClient()
