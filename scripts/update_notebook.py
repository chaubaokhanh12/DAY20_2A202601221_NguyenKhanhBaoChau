"""One-off helper: fill the notebook's TODO cells with the finished implementations.

Kept in the repo so the notebook can be regenerated deterministically after the source
code evolves (run: `python scripts/update_notebook.py`).
"""

import json
from pathlib import Path

NOTEBOOK = Path("notebooks/demo_multi_agent_walkthrough.ipynb")

CELL_MOCK_SERVICES = '''from dataclasses import dataclass


class MockSearchClient:
    """Trả về nguồn giả lập — cùng interface với services.search_client.SearchClient."""

    _FAKE_DOCS = [
        SourceDocument(
            title="RAG vs Fine-tuning: A Practical Guide",
            url="https://example.com/rag-vs-ft",
            snippet="RAG phù hợp khi dữ liệu thay đổi thường xuyên; fine-tuning tốt cho style/format.",
            ref="[S1]",
        ),
        SourceDocument(
            title="Retrieval-Augmented Generation Survey",
            url="https://example.com/rag-survey",
            snippet="RAG giảm hallucination bằng cách grounding vào tài liệu ngoài.",
            ref="[S2]",
        ),
        SourceDocument(
            title="When to Fine-tune LLMs",
            url="https://example.com/when-finetune",
            snippet="Fine-tuning hiệu quả khi cần hành vi nhất quán và latency thấp.",
            ref="[S3]",
        ),
    ]

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        return self._FAKE_DOCS[:max_results]


@dataclass(frozen=True)
class MockLLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class MockLLMClient:
    """Giả lập LLM — cùng interface với services.llm_client.LLMClient.

    Phân nhánh theo vai trò trong `ROLE:` của system prompt và giữ nguyên các marker
    `[S#]` có trong user prompt, nhờ vậy citation vẫn kiểm chứng được khi chạy offline.
    """

    def complete(self, system_prompt: str, user_prompt: str) -> MockLLMResponse:
        role = "generic"
        for line in system_prompt.splitlines():
            if "ROLE:" in line:
                role = line.split("ROLE:", 1)[1].strip().lower()
                break

        refs = re.findall(r"\\[S\\d+\\]", user_prompt)
        cite = " ".join(dict.fromkeys(refs))

        if role == "analyst":
            content = (
                "Key claims:\\n"
                f"1. RAG thắng khi dữ liệu đổi nhanh và cần trích dẫn {cite}.\\n"
                f"2. Fine-tuning thắng khi cần style ổn định, latency thấp {cite}.\\n"
                "Tensions: hai nguồn không thống nhất về chi phí vận hành.\\n"
                "Evidence gaps: thiếu số liệu định lượng.\\n"
                "Confidence: medium."
            )
        elif role == "writer":
            content = (
                f"## Answer\\nChọn RAG hay fine-tuning phụ thuộc tốc độ thay đổi của dữ liệu {cite}.\\n\\n"
                "## Key points\\n"
                f"- Dữ liệu đổi liên tục + cần citation → RAG {refs[0] if refs else ''}\\n"
                f"- Cần style/format nhất quán → fine-tuning {refs[-1] if refs else ''}\\n\\n"
                "## Limitations\\nMock LLM: nội dung theo template, chỉ để prototype."
            )
        else:
            content = f"[mock:{role}] {user_prompt.strip().splitlines()[0]}"

        return MockLLMResponse(
            content=content,
            input_tokens=len(system_prompt + user_prompt) // 4,
            output_tokens=len(content) // 4,
        )


# Smoke test
search_client = MockSearchClient()
docs = search_client.search(query.query, max_results=query.max_sources)
for d in docs:
    print(f"- {d.ref} {d.title}: {d.snippet[:60]}...")

print("\\n", MockLLMClient().complete("ROLE: analyst", "Sources:\\n[S1] a\\n[S2] b").content[:120])
'''

CELL_AGENTS = '''class DemoResearcherAgent:
    """MẪU: thu thập nguồn và ghi chú nghiên cứu."""

    name = "researcher"

    def __init__(self, search_client: MockSearchClient) -> None:
        self.search_client = search_client

    def run(self, state: ResearchState) -> ResearchState:
        docs = self.search_client.search(
            state.request.query, max_results=state.request.max_sources
        )
        state.sources = docs
        state.research_notes = "\\n".join(f"- {d.title}: {d.snippet} {d.ref}" for d in docs)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes,
                metadata={"num_sources": len(docs)},
            )
        )
        state.add_trace_event("researcher.done", {"num_sources": len(docs)})
        return state


class DemoAnalystAgent:
    """Phân tích sources thành analysis_notes."""

    name = "analyst"

    def __init__(self, llm_client: MockLLMClient) -> None:
        self.llm_client = llm_client

    def run(self, state: ResearchState) -> ResearchState:
        # 1. Guard: không có ngữ cảnh thì dừng sớm thay vì bịa ra phân tích.
        if not state.sources:
            state.errors.append("analyst: no sources to analyse")
            state.add_trace_event("analyst.skipped", {"reason": "empty sources"})
            return state

        # 2. Gọi LLM với ngữ cảnh đã đánh dấu citation.
        sources_block = "\\n".join(f"{d.ref} {d.title}: {d.snippet}" for d in state.sources)
        response = self.llm_client.complete(
            system_prompt="ROLE: analyst\\nYou analyse research notes into key claims.",
            user_prompt=(
                f"Question: {state.request.query}\\n\\n"
                f"Notes:\\n{state.research_notes}\\n\\nSources:\\n{sources_block}"
            ),
        )

        # 3. Ghi state + AgentResult, 4. trace.
        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={"tokens": (response.input_tokens or 0) + (response.output_tokens or 0)},
            )
        )
        state.add_trace_event("analyst.done", {"chars": len(response.content)})
        return state


class DemoWriterAgent:
    """Viết final_answer có trích dẫn nguồn."""

    name = "writer"

    def __init__(self, llm_client: MockLLMClient) -> None:
        self.llm_client = llm_client

    def run(self, state: ResearchState) -> ResearchState:
        context = state.analysis_notes or state.research_notes or ""
        sources_block = "\\n".join(f"{d.ref} {d.title}: {d.snippet}" for d in state.sources)
        response = self.llm_client.complete(
            system_prompt="ROLE: writer\\nYou write the final answer with citations.",
            user_prompt=(
                f"Question: {state.request.query}\\n"
                f"Audience: {state.request.audience}\\n\\n"
                f"Analysis:\\n{context}\\n\\nSources:\\n{sources_block}"
            ),
        )

        citations = "\\n".join(f"{d.ref} {d.title} ({d.url})" for d in state.sources)
        state.final_answer = f"{response.content}\\n\\n## Sources\\n{citations}"
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer,
                metadata={"words": len(state.final_answer.split())},
            )
        )
        state.add_trace_event("writer.done", {"words": len(state.final_answer.split())})
        return state


# Smoke test agent mẫu
state = ResearchState(request=query)
state = DemoResearcherAgent(search_client).run(state)
print(state.research_notes)
'''

CELL_ROUTER = '''MAX_ITERATIONS = 6


def demo_supervisor_route(state: ResearchState) -> str:
    """Trả về một trong: 'researcher' | 'analyst' | 'writer' | 'done'."""
    # Guard chống vòng lặp vô hạn — GIỮ NGUYÊN dòng này
    if state.iteration >= MAX_ITERATIONS:
        return "done"

    # Routing = "field nào còn thiếu thì gọi agent lấp field đó".
    if not state.sources:
        # Fallback: nếu researcher đã fail 2 lần thì đừng lặp nữa, viết bằng thứ đang có.
        if state.route_history.count("researcher") >= 2:
            return "writer" if not state.final_answer else "done"
        return "researcher"
    if not state.analysis_notes:
        if state.route_history.count("analyst") >= 2:
            return "writer"
        return "analyst"
    if not state.final_answer:
        return "writer"
    return "done"


# Kiểm tra nhanh policy bằng state giả lập
probe = ResearchState(request=query)
print("empty state        ->", demo_supervisor_route(probe))
probe.sources = search_client.search(query.query, 2)
print("có sources         ->", demo_supervisor_route(probe))
probe.analysis_notes = "..."
print("có analysis        ->", demo_supervisor_route(probe))
probe.final_answer = "..."
print("có final answer    ->", demo_supervisor_route(probe))
'''

CELL_BENCHMARK = '''from multi_agent_research_lab.evaluation.benchmark import run_benchmark


def run_single_agent(query_text: str) -> ResearchState:
    """Baseline: một lần gọi LLM duy nhất, không search, không phân tích."""
    q = ResearchQuery(query=query_text, max_sources=3)
    state = ResearchState(request=q)

    response = MockLLMClient().complete(
        system_prompt="ROLE: baseline\\nAnswer the question end to end. You have no search tool.",
        user_prompt=f"Question: {query_text}",
    )
    state.final_answer = response.content
    state.record_route("baseline")
    state.agent_results.append(
        AgentResult(agent=AgentName.WRITER, content=response.content, metadata={"calls": 1})
    )
    return state


def compute_citation_coverage(state: ResearchState) -> float:
    """Tỷ lệ nguồn trong state.sources được nhắc đến trong final_answer."""
    if not state.sources or not state.final_answer:
        return 0.0
    answer = state.final_answer
    hits = sum(
        1
        for d in state.sources
        if (d.ref and d.ref in answer) or d.title in answer or (d.url and d.url in answer)
    )
    return round(hits / len(state.sources), 4)


demo_query = "So sánh RAG và fine-tuning cho domain adaptation"

results: list[BenchmarkMetrics] = []
for run_name, runner in [
    ("single_agent", run_single_agent),
    ("multi_agent", run_demo_workflow),
]:
    st, metrics = run_benchmark(run_name, demo_query, runner)
    metrics.citation_coverage = compute_citation_coverage(st)
    results.append(metrics)

print(f"{'run':<15}{'latency (s)':<15}{'quality':<10}{'citation cov.':<15}")
for m in results:
    print(f"{m.run_name:<15}{m.latency_seconds:<15.4f}{m.quality_score!s:<10}{m.citation_coverage:<15}")

print("\\n→ Multi-agent tốn nhiều call hơn nhưng là bản duy nhất có citation kiểm chứng được.")
'''

CELL_SETUP_IMPORTS = '''import re
import sys
from pathlib import Path

# Cho phép import package khi chạy notebook từ thư mục notebooks/
repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(repo_root / "src"))

from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    BenchmarkMetrics,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState

print("✅ Import OK — package sẵn sàng")
'''

CELL_WORKFLOW = '''def run_demo_workflow(query_text: str) -> ResearchState:
    q = ResearchQuery(query=query_text, max_sources=3)
    state = ResearchState(request=q)

    llm = MockLLMClient()
    agents = {
        "researcher": DemoResearcherAgent(MockSearchClient()),
        "analyst": DemoAnalystAgent(llm),
        "writer": DemoWriterAgent(llm),
    }

    while True:
        route = demo_supervisor_route(state)
        state.record_route(route)
        if route == "done":
            break
        state = agents[route].run(state)

    return state


final_state = run_demo_workflow("So sánh RAG và fine-tuning cho domain adaptation")
print("Route history:", final_state.route_history)
print("\\n=== FINAL ANSWER ===\\n")
print(final_state.final_answer)
'''

MD_INTRO = """# Lab 20 — Multi-Agent Research Demo Notebook

Notebook này **prototype** các khối logic của bài lab; bản production đầy đủ (supervisor
có budget guard, critic, LangGraph engine, benchmark) nằm trong `src/multi_agent_research_lab/`.

**Luồng làm việc:**
1. Khám phá schemas & shared state
2. Mock services (LLM + Search) để chạy không cần API key
3. Agent demo (Researcher → Analyst → Writer)
4. Supervisor routing + vòng lặp workflow mini
5. Benchmark single-agent vs multi-agent

> ✅ Toàn bộ ô `TODO(student)` trong notebook này **đã được hoàn thành**. Đối chiếu với
> bản chính thức trong `src/` theo bảng mapping ở cuối notebook.
"""

MD_MOCK = """## 2. Mock Services

Để demo không cần API key, ta dùng mock. Trong bản chính thức (`src/services/`), cùng
interface đó được cài đặt cho provider thật (OpenAI / Tavily) và tự fallback về mock khi
thiếu key.

- `MockSearchClient`: nguồn giả lập, đã gắn sẵn citation marker `[S#]`.
- `MockLLMClient`: phân nhánh theo `ROLE:` trong system prompt, giữ nguyên marker `[S#]`.
"""

MD_AGENTS = """## 3. Demo Agents

Mỗi agent tuân theo contract `BaseAgent.run(state) -> state`: đọc state, ghi đúng field
mình phụ trách, ghi `AgentResult` + trace event, rồi trả state về cho supervisor.
"""

MD_NEXT = """## 7. Từ notebook sang `src/`

| Notebook | Bản chính thức trong `src/multi_agent_research_lab/` |
|---|---|
| `MockLLMClient` | `services/llm_client.py` (`MockLLMClient` + `OpenAIClient` + cost/retry) |
| `MockSearchClient` | `services/search_client.py` (+ `TavilySearchClient`, dedupe, scoring) |
| `DemoResearcherAgent` / `DemoAnalystAgent` / `DemoWriterAgent` | `agents/researcher.py`, `agents/analyst.py`, `agents/writer.py` (+ `agents/critic.py`) |
| `demo_supervisor_route` | `agents/supervisor.py` (+ budget iteration/cost, revision loop) |
| `run_demo_workflow` | `graph/workflow.py` (LangGraph `StateGraph` + sequential fallback) |
| `run_single_agent` | `graph/baseline.py` |
| `compute_citation_coverage` | `utils/text.py` + `evaluation/quality.py` |

Verify bản chính thức:

```bash
python -m multi_agent_research_lab.cli doctor
python -m multi_agent_research_lab.cli multi-agent -q "..." --trace
python -m multi_agent_research_lab.cli benchmark
make lint && make test
bash scripts/check_todos.sh   # không còn TODO(student) trong src/
```
"""


def as_source(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    replacements = {
        0: MD_INTRO,
        2: CELL_SETUP_IMPORTS,
        5: MD_MOCK,
        6: CELL_MOCK_SERVICES,
        7: MD_AGENTS,
        8: CELL_AGENTS,
        10: CELL_ROUTER,
        12: CELL_WORKFLOW,
        14: CELL_BENCHMARK,
        15: MD_NEXT,
    }
    for index, text in replacements.items():
        cells[index]["source"] = as_source(text)
        if cells[index]["cell_type"] == "code":
            cells[index]["outputs"] = []
            cells[index]["execution_count"] = None
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"updated {NOTEBOOK} ({len(replacements)} cells)")


if __name__ == "__main__":
    main()
