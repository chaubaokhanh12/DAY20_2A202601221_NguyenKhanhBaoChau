"""Shared fixtures.

Every test runs against the deterministic offline providers, so the suite never needs a
network connection, never spends money, and always produces the same numbers - even on a
machine where the developer exported a real `OPENAI_API_KEY`.
"""

from typing import Any

import pytest

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import MockLLMClient
from multi_agent_research_lab.services.search_client import MockSearchClient

QUERY = "Compare single-agent and multi-agent workflows for customer support"


def offline_settings(**overrides: Any) -> Settings:
    """Settings pinned to the mock providers; `overrides` use the env-var alias names."""

    values: dict[str, Any] = {
        "LLM_PROVIDER": "mock",
        "SEARCH_PROVIDER": "mock",
        "ENGINE": "sequential",
        "OPENAI_API_KEY": None,
        "TAVILY_API_KEY": None,
        "MAX_ITERATIONS": 8,
        "TIMEOUT_SECONDS": 30,
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings() -> Settings:
    return offline_settings()


@pytest.fixture
def llm() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def search() -> MockSearchClient:
    return MockSearchClient()


@pytest.fixture
def state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query=QUERY, max_sources=3))
