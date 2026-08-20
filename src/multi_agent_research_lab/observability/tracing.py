"""Tracing hooks.

Provider-neutral by design: a run produces a list of spans that can be printed, saved as
JSON, or forwarded to LangSmith when `LANGSMITH_API_KEY` is set. Agents only ever call
`tracer.span(...)`, so swapping backends never touches agent code.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal standalone span (kept for scripts and notebooks)."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started


class Tracer:
    """Collects spans for a single workflow run."""

    def __init__(self, run_name: str = "run", forward_to_langsmith: bool = False) -> None:
        self.run_name = run_name
        self.spans: list[dict[str, Any]] = []
        self.forward_to_langsmith = forward_to_langsmith
        self._started_at = datetime.now(UTC)

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        started = perf_counter()
        span: dict[str, Any] = {
            "name": name,
            "attributes": dict(attributes or {}),
            "status": "ok",
            "duration_seconds": None,
            "started_at": datetime.now(UTC).isoformat(),
        }
        self.spans.append(span)
        try:
            yield span
        except Exception as exc:
            span["status"] = "error"
            span["error"] = repr(exc)
            raise
        finally:
            span["duration_seconds"] = round(perf_counter() - started, 4)
            logger.debug("span %s (%.3fs) %s", name, span["duration_seconds"], span["status"])
            if self.forward_to_langsmith:
                self._forward(span)

    # ---------------------------------------------------------------- exporting
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "started_at": self._started_at.isoformat(),
            # spans nest, so the longest one is the wall time - summing would double count
            "wall_seconds": round(
                max((s["duration_seconds"] or 0.0 for s in self.spans), default=0.0), 4
            ),
            "span_count": len(self.spans),
            "spans": self.spans,
        }

    def export_json(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), "utf-8")
        return target

    def to_markdown(self) -> str:
        lines = [
            f"### Trace `{self.run_name}`",
            "",
            "| # | Span | Duration (s) | Status | Attrs |",
            "|--:|---|--:|---|---|",
        ]
        for index, span in enumerate(self.spans, start=1):
            attrs = ", ".join(f"{k}={v}" for k, v in span["attributes"].items())
            lines.append(
                f"| {index} | {span['name']} | {span['duration_seconds']:.3f} "
                f"| {span['status']} | {attrs} |"
            )
        return "\n".join(lines) + "\n"

    def _forward(self, span: dict[str, Any]) -> None:
        """Best-effort LangSmith export; never breaks a run if the provider is down."""

        try:  # pragma: no cover - requires network + credentials
            from langsmith import Client

            Client().create_run(
                name=span["name"],
                run_type="chain",
                inputs=span["attributes"],
                outputs={"status": span["status"], "duration": span["duration_seconds"]},
                project_name=self.run_name,
            )
        except Exception as exc:  # noqa: BLE001 - observability must not break the run
            logger.debug("LangSmith export skipped: %r", exc)
            self.forward_to_langsmith = False
