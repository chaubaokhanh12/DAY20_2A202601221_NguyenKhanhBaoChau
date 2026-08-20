"""Application configuration.

Keep config small and explicit. Do not read environment variables directly in agents.
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path("configs/lab_default.yaml")


class LLMProvider(StrEnum):
    AUTO = "auto"
    OPENAI = "openai"
    MOCK = "mock"


class SearchProvider(StrEnum):
    AUTO = "auto"
    TAVILY = "tavily"
    MOCK = "mock"


class Engine(StrEnum):
    """Orchestration backend for the multi-agent workflow."""

    AUTO = "auto"
    LANGGRAPH = "langgraph"
    SEQUENTIAL = "sequential"


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="local", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")

    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="multi-agent-research-lab", validation_alias="LANGSMITH_PROJECT"
    )

    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST"
    )

    tavily_api_key: str | None = Field(default=None, validation_alias="TAVILY_API_KEY")

    # Provider selection. `auto` = use the real provider when a key exists, else the
    # deterministic offline mock so the lab always runs.
    llm_provider: LLMProvider = Field(default=LLMProvider.AUTO, validation_alias="LLM_PROVIDER")
    search_provider: SearchProvider = Field(
        default=SearchProvider.AUTO, validation_alias="SEARCH_PROVIDER"
    )
    engine: Engine = Field(default=Engine.AUTO, validation_alias="ENGINE")

    # Guardrails
    max_iterations: int = Field(default=8, ge=1, le=20, validation_alias="MAX_ITERATIONS")
    timeout_seconds: int = Field(default=60, ge=5, le=600, validation_alias="TIMEOUT_SECONDS")
    request_timeout_seconds: int = Field(
        default=30, ge=1, le=300, validation_alias="REQUEST_TIMEOUT_SECONDS"
    )
    retry_attempts: int = Field(default=3, ge=1, le=6, validation_alias="RETRY_ATTEMPTS")
    max_revisions: int = Field(default=1, ge=0, le=3, validation_alias="MAX_REVISIONS")
    max_cost_usd: float = Field(default=1.0, ge=0.0, validation_alias="MAX_COST_USD")
    enable_critic: bool = Field(default=True, validation_alias="ENABLE_CRITIC")

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key)

    def resolved_llm_provider(self) -> LLMProvider:
        if self.llm_provider is not LLMProvider.AUTO:
            return self.llm_provider
        return LLMProvider.OPENAI if self.has_openai else LLMProvider.MOCK

    def resolved_search_provider(self) -> SearchProvider:
        if self.search_provider is not SearchProvider.AUTO:
            return self.search_provider
        return SearchProvider.TAVILY if self.has_tavily else SearchProvider.MOCK


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()


class AgentConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class LabMeta(BaseModel):
    name: str = "multi-agent-research-lab"
    max_iterations: int = 8
    timeout_seconds: int = 60


class BenchmarkConfig(BaseModel):
    queries: list[str] = Field(default_factory=list)


class LabConfig(BaseModel):
    """Typed view of `configs/*.yaml`."""

    lab: LabMeta = Field(default_factory=LabMeta)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    def agent(self, name: str) -> AgentConfig:
        return self.agents.get(name, AgentConfig())


def load_lab_config(path: Path | str = DEFAULT_CONFIG_PATH) -> LabConfig:
    """Load the YAML lab config, falling back to defaults when the file is absent."""

    config_path = Path(path)
    if not config_path.exists():
        return LabConfig()
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return LabConfig.model_validate(raw)
