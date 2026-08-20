"""Small local web UI for demonstrating the lab workflow."""

from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from multi_agent_research_lab.core.config import Engine, get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.quality import heuristic_quality
from multi_agent_research_lab.graph.baseline import SingleAgentBaseline
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.tracing import Tracer
from multi_agent_research_lab.utils.text import citation_coverage

ROOT = Path(__file__).parent
WEB_ROOT = ROOT / "demo"


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/compare":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            query = str(payload.get("query", "")).strip()
            if not query:
                raise ValueError("Please enter a research question.")
            if bool(payload.get("offline")):
                os.environ["LLM_PROVIDER"] = "mock"
                os.environ["SEARCH_PROVIDER"] = "mock"
                get_settings.cache_clear()
            else:
                os.environ.pop("LLM_PROVIDER", None)
                os.environ.pop("SEARCH_PROVIDER", None)
                get_settings.cache_clear()
            request = ResearchQuery(
                query=query,
                max_sources=int(payload.get("max_sources", 5)),
                audience="engineering team",
            )
            baseline_tracer = Tracer(run_name="browser_baseline")
            baseline = SingleAgentBaseline(tracer=baseline_tracer).run(ResearchState(request=request))
            tracer = Tracer(run_name="browser_multi_agent")
            workflow = MultiAgentWorkflow(
                tracer=tracer,
                enable_critic=bool(payload.get("critic", True)),
                engine=Engine(str(payload.get("engine", "auto"))),
            )
            state = workflow.run(ResearchState(request=request))
            self._json(
                {
                    "baseline": self._result(baseline, "No retrieval: answer comes only from the configured LLM."),
                    "multi_agent": self._result(
                        state,
                        "Retrieved sources are listed below. Their provider is recorded per source.",
                    ),
                    "engine": workflow.engine.value,
                    "llm_provider": workflow.settings.resolved_llm_provider().value,
                    "search_provider": workflow.settings.resolved_search_provider().value,
                }
            )
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": f"Workflow failed: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _json(self, data: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _result(state: ResearchState, provenance: str) -> dict[str, object]:
        quality = heuristic_quality(state)
        return {
            "state": state.model_dump(mode="json"),
            "quality": {"score": quality.score, "breakdown": quality.breakdown, "notes": quality.notes},
            "latency_seconds": round(state.elapsed_seconds(), 3),
            "citation_coverage": citation_coverage(state.final_answer or "", state.source_refs()),
            "provenance": provenance,
        }


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DemoHandler)
    print("Lab demo: http://127.0.0.1:8000")
    server.serve_forever()
