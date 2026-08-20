"""CLI smoke tests - they also prove the lab runs with zero credentials."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from multi_agent_research_lab.cli import app
from multi_agent_research_lab.core.config import get_settings

runner = CliRunner()


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Force the offline providers regardless of the developer's real environment."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    monkeypatch.setenv("ENGINE", "sequential")
    monkeypatch.setenv("COLUMNS", "200")  # keep rich from wrapping assertions apart
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_doctor_reports_offline_mode() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "mock" in result.stdout


def test_baseline_command() -> None:
    result = runner.invoke(app, ["baseline", "-q", "Summarize guardrails for LLM agents"])
    assert result.exit_code == 0
    assert "Single-Agent Baseline" in result.stdout
    assert "Quality" in result.stdout


def test_multi_agent_command_prints_routes() -> None:
    result = runner.invoke(
        app, ["multi-agent", "-q", "Summarize guardrails for LLM agents", "--no-critic"]
    )
    assert result.exit_code == 0
    assert "researcher" in result.stdout


def test_rejects_too_short_query() -> None:
    result = runner.invoke(app, ["multi-agent", "-q", "hi"])
    assert result.exit_code == 1
    assert "Invalid query" in result.stdout


def test_benchmark_writes_a_report(tmp_path: Path) -> None:
    output = tmp_path / "benchmark_report.md"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "-q",
            "Summarize production guardrails for LLM agents",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0
    report = output.read_text(encoding="utf-8")
    assert "single_agent" in report
    assert "multi_agent" in report
    assert "Citation coverage" in report
