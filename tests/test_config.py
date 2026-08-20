from pathlib import Path

from multi_agent_research_lab.core.config import (
    LLMProvider,
    SearchProvider,
    Settings,
    load_lab_config,
)
from tests.conftest import offline_settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.openai_model
    assert settings.max_iterations >= 1


def test_provider_resolution_prefers_real_provider_when_a_key_exists() -> None:
    offline = offline_settings(LLM_PROVIDER="auto", SEARCH_PROVIDER="auto")
    online = offline_settings(
        LLM_PROVIDER="auto",
        SEARCH_PROVIDER="auto",
        OPENAI_API_KEY="sk-test",
        TAVILY_API_KEY="tvly-test",
    )

    assert offline.resolved_llm_provider() is LLMProvider.MOCK
    assert offline.resolved_search_provider() is SearchProvider.MOCK
    assert online.resolved_llm_provider() is LLMProvider.OPENAI
    assert online.resolved_search_provider() is SearchProvider.TAVILY


def test_explicit_provider_overrides_key_detection() -> None:
    forced = offline_settings(OPENAI_API_KEY="sk-test", LLM_PROVIDER="mock")
    assert forced.resolved_llm_provider() is LLMProvider.MOCK


def test_lab_config_loads_yaml() -> None:
    config = load_lab_config(Path("configs/lab_default.yaml"))

    assert config.lab.name == "multi-agent-research-lab"
    assert config.lab.max_iterations >= 1
    assert len(config.benchmark.queries) >= 3
    assert config.agent("writer").temperature > config.agent("supervisor").temperature


def test_lab_config_falls_back_when_file_is_missing() -> None:
    config = load_lab_config(Path("configs/does_not_exist.yaml"))
    assert config.benchmark.queries == []
    assert config.agent("unknown").model
