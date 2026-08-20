# Lab 20: Multi-Agent Research System

Hệ thống nghiên cứu **Supervisor + Researcher + Analyst + Writer + Critic**, có benchmark
đối chiếu với single-agent baseline. Bài nộp Day 20 — Nguyễn Khánh Bảo Châu (2A202601221).

> **Chạy được ngay, không cần API key.** Khi thiếu `OPENAI_API_KEY` / `TAVILY_API_KEY`, hệ
> thống tự chuyển sang provider mock **deterministic** (cùng interface, cùng code path), nên
> demo, test và benchmark luôn tái lập được. Có key thì tự dùng OpenAI + Tavily.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,llm]"
```

```bash
python -m multi_agent_research_lab.cli doctor
```

```bash
python -m multi_agent_research_lab.cli multi-agent -q "Research GraphRAG state-of-the-art and write a 500-word summary" --trace
```

```bash
python -m multi_agent_research_lab.cli benchmark
```

| Lệnh | Việc nó làm |
|---|---|
| `doctor` | In cấu hình hiệu lực: provider nào, engine nào, guardrail nào |
| `baseline -q "..."` | Single-agent control run (1 LLM call, không retrieval) |
| `multi-agent -q "..."` | Chạy workflow đầy đủ; `--trace` in trace, `--save` ghi artifact, `--engine`, `--no-critic` |
| `benchmark` | Chạy cả hai runner trên query trong `configs/lab_default.yaml` → `reports/benchmark_report.md` |

## Kết quả benchmark (mock provider, 3 query, deterministic)

| Run | Latency | Cost | Quality (0-10) | Citation coverage | LLM calls | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| single_agent | ~0.00s | $0 | 4.2 | 0% | 1 | 156 |
| multi_agent | ~0.4s | $0 | 10.0 | 100% | 4 | 2 395 |

Đọc đúng bảng này: multi-agent **mua** chất lượng và khả năng kiểm chứng bằng **4x số call
và ~15x token**. Chi tiết + per-query + sample answer: `reports/benchmark_report.md`.

## Kiến trúc

```text
                 ┌──────────────┐
   query ───────▶│  Supervisor  │◀───────────── mỗi worker trả state về đây
                 └──────┬───────┘
        route theo field còn thiếu + budget
     ┌──────────┬───────┴───────┬────────────┐
     ▼          ▼               ▼            ▼
 Researcher   Analyst        Writer       Critic ──▶ (revise 1 lần nếu không đạt)
 sources      analysis       final        verdict
 [S1..Sn]     notes          answer       + coverage
     └──────────┴───────┬───────┴────────────┘
                        ▼
        ResearchState (usage, trace, errors, status)
                        ▼
             CLI · trace JSON · benchmark report
```

Hai engine dùng chung một routing policy và một bộ agent:

- **`langgraph`** — `StateGraph` thật với node supervisor + conditional edges (mặc định).
- **`sequential`** — vòng lặp supervisor thuần Python, không phụ thuộc, dùng làm fallback và CI.

`tests/test_workflow.py::test_langgraph_engine_matches_sequential` khoá hành vi của hai
engine phải giống hệt nhau.

## Guardrails

| Guardrail | Giá trị mặc định | Cài ở đâu |
|---|---|---|
| Max iterations (worker dispatch) | 8 | `SupervisorAgent.decide` |
| Wall-clock timeout | 60s | `MultiAgentWorkflow._execute` |
| Per-request timeout | 30s | `OpenAIClient` |
| Retry + backoff | 3 lần | `OpenAIClient` + retry cấp agent trong workflow |
| Cost ceiling | $1.00 | `SupervisorAgent.decide` |
| Revision budget | 1 | Supervisor + Writer |
| Anti-oscillation | cùng route ≤ 2 lần | `SupervisorAgent.decide` |
| Validation | Pydantic mọi handoff + Critic chặn citation ảo | `core/`, `agents/critic.py` |

Không guardrail nào ném exception ra người dùng: run luôn kết thúc với `final_answer` và
status `completed` / `degraded` / `failed`.

## Cấu trúc repo

```text
src/multi_agent_research_lab/
├── agents/        supervisor, researcher, analyst, writer, critic (+ base: prompt/usage/fallback)
├── core/          config (env + YAML), state, schemas, errors
├── graph/         workflow.py (LangGraph + sequential), baseline.py (single-agent)
├── services/      llm_client.py (OpenAI + mock + pricing), search_client.py (Tavily + mock), storage
├── evaluation/    benchmark.py, quality.py (rubric 0-10 + LLM judge), report.py
├── observability/ logging.py, tracing.py (spans, JSON export, LangSmith best-effort)
├── utils/         text.py (citation coverage, keyword overlap), timer.py
└── cli.py         doctor | baseline | multi-agent | benchmark
```

## Chất lượng

```bash
make lint && make typecheck && make test
```

- `ruff check` + `ruff format --check`: sạch.
- `mypy --strict` trên `src` và `tests`: sạch.
- 55 test, chạy offline, deterministic — phủ routing (8 nhánh), fallback khi provider chết,
  timeout, budget, citation ảo, benchmark, CLI.
- CI (`.github/workflows/ci.yml`) chạy lint + format + typecheck + test.

## Deliverables

| # | Deliverable | Ở đâu |
|---|---|---|
| 1 | Thiết kế hệ thống | `docs/design_template.md` |
| 2 | Trace | `reports/traces/sample_multi_agent_trace.json`, hoặc `cli multi-agent --trace` |
| 3 | Benchmark report | `reports/benchmark_report.md` |
| 4 | Failure mode + cách fix | `docs/failure_modes.md` |
| 5 | Exit ticket | `docs/exit_ticket.md` |
| 6 | Notebook walkthrough | `notebooks/demo_multi_agent_walkthrough.ipynb` |

## Dùng provider thật

```bash
cp .env.example .env   # điền OPENAI_API_KEY (và TAVILY_API_KEY nếu muốn search thật)
python -m multi_agent_research_lab.cli doctor   # xác nhận đã chuyển sang openai/tavily
```

Cost được ước lượng từ bảng giá trong `services/llm_client.py`; cập nhật bảng đó khi đổi model.
macOS gặp lỗi SSL certificate: xem `docs/lab_guide.md`.

## References

- Anthropic: Building effective agents — https://www.anthropic.com/engineering/building-effective-agents
- OpenAI Agents SDK orchestration/handoffs — https://developers.openai.com/api/docs/guides/agents/orchestration
- LangGraph concepts — https://langchain-ai.github.io/langgraph/concepts/
- LangSmith tracing — https://docs.smith.langchain.com/
- Langfuse tracing — https://langfuse.com/docs
