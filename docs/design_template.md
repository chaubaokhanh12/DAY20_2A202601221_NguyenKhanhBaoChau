# Design — Multi-Agent Research System

> Bản điền hoàn chỉnh của design template (deliverable #1 của lab).
> Người thực hiện: Nguyễn Khánh Bảo Châu — 2A202601221.

## Problem

Nhận một câu hỏi nghiên cứu dạng mở ("Research GraphRAG state-of-the-art and write a
500-word summary") và trả về câu trả lời có cấu trúc, **có trích dẫn kiểm chứng được**,
trong ngân sách thời gian/chi phí xác định, đồng thời ghi lại trace đủ để giải thích
"agent nào đã làm gì, tốn bao nhiêu".

## Why multi-agent?

Single-agent làm được việc này ở mức "nghe hợp lý" nhưng không kiểm chứng được:

| Vấn đề của single-agent | Multi-agent xử lý thế nào |
|---|---|
| Không có bước retrieval tách biệt → không có citation | Researcher chịu trách nhiệm riêng cho nguồn + gán marker `[S#]` |
| Một prompt phải vừa tìm, vừa phân tích, vừa viết → loãng context | Mỗi agent một prompt ngắn, một nhiệm vụ |
| Không có điểm nào để chèn kiểm tra chất lượng | Critic là một node độc lập, chặn draft không đạt |
| Sai ở đâu cũng khó biết | Mỗi bước là một span trace + một `AgentResult` |

Đo thực tế (mock provider, 3 query — xem `reports/benchmark_report.md`): quality
4.2 → 10.0, citation coverage 0% → 100%, đổi lại 4x số LLM call và ~15x token.
**Kết luận: multi-agent đáng dùng khi yêu cầu là "trả lời có bằng chứng", không đáng
khi chỉ cần một câu trả lời nhanh.**

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode & xử lý |
|---|---|---|---|---|
| Supervisor | Chọn agent kế tiếp, giữ ngân sách, quyết định dừng | Toàn bộ `ResearchState` | `RoutingDecision(route, reason)` | Oscillation → đếm `route_history`, cùng route ≥ 2 lần thì degrade sang Writer |
| Researcher | Tìm nguồn, dedupe, gán `[S#]`, viết research notes | `request.query`, `max_sources` | `sources`, `research_notes` | Search provider chết → ghi `errors`, Supervisor degrade; LLM chết → extractive notes từ snippet |
| Analyst | Rút key claims, mâu thuẫn, evidence gap, confidence | `research_notes`, `sources` | `analysis_notes` | Không có nguồn → skip có ghi lý do, không bịa |
| Writer | Viết câu trả lời cuối theo audience + gắn citation | `analysis_notes`/`research_notes`, `sources` | `final_answer` | LLM chết → fallback extractive; danh sách `## Sources` luôn được append deterministic |
| Critic (bonus) | Kiểm tra citation ảo, coverage, cấu trúc, độ dài | `final_answer`, `sources` | `critic_verdict`, `critic_notes` | Chỉ được yêu cầu sửa `max_revisions` lần |

## Shared state

`src/multi_agent_research_lab/core/state.py` — mỗi field tồn tại vì có consumer cụ thể:

| Field | Ai ghi | Ai đọc | Vì sao cần |
|---|---|---|---|
| `request` | CLI | tất cả | query, audience, max_sources |
| `sources` | Researcher | Analyst, Writer, Critic, metric | citation coverage chỉ tính được khi biết tập nguồn |
| `research_notes` / `analysis_notes` | Researcher / Analyst | Analyst / Writer | handoff, tránh nhồi lại toàn bộ context |
| `final_answer` | Writer | Critic, CLI, benchmark | deliverable |
| `critic_verdict` | Critic | Supervisor | điều kiện vòng revision |
| `route_history`, `routing_decisions` | Supervisor | guard + trace | giải thích "vì sao đi nhánh này" |
| `usage` | mọi LLM call | benchmark, cost guard | so sánh chi phí |
| `errors`, `status` | mọi agent | benchmark, CLI | failure rate + degrade thay vì crash |
| `trace` | mọi agent | CLI `--trace`, export JSON | debug sau sự cố |

## Routing policy

```text
supervisor ──> researcher ──┐
     ^                      │
     ├──────── analyst <────┘
     ├──────── writer  <──── (revision khi critic từ chối)
     ├──────── critic
     └──────── done
```

Quy tắc (deterministic, không tốn LLM call):

1. `iteration >= max_iterations` → done (lý do được ghi lại).
2. `usage.cost_usd > max_cost_usd` → done.
3. chưa có `sources` → researcher (đã fail 2 lần → writer).
4. chưa có `analysis_notes` → analyst (đã fail 2 lần → writer).
5. chưa có `final_answer` → writer.
6. critic bật & chưa có verdict → critic.
7. critic từ chối & `revisions < max_revisions` → writer.
8. còn lại → done.

Router bằng rule chứ không bằng LLM vì: rẻ hơn, deterministic, unit-test được
(`tests/test_supervisor.py` phủ toàn bộ 8 nhánh). LLM chỉ dùng cho phần nội dung.

## Guardrails

- **Max iterations**: 8 lượt dispatch worker (`MAX_ITERATIONS`), decision `done` không tính.
- **Timeout**: `TIMEOUT_SECONDS=60` wall-clock, kiểm tra trước mỗi worker → `BudgetExceededError`.
- **Request timeout**: `REQUEST_TIMEOUT_SECONDS=30` cho từng call provider.
- **Retry**: tenacity `stop_after_attempt(3)` + exponential backoff trong `OpenAIClient`;
  thêm một vòng retry cấp agent trong `MultiAgentWorkflow._execute`.
- **Fallback**: mọi agent có nhánh offline deterministic; workflow luôn trả `final_answer`
  (status `degraded`/`failed`) thay vì ném exception ra CLI.
- **Cost ceiling**: `MAX_COST_USD=1.0`, kiểm tra trong Supervisor.
- **Validation**: toàn bộ handoff qua Pydantic; Critic chặn citation trỏ tới nguồn không tồn tại.
- **Recursion limit** của LangGraph = `max_iterations * 2 + 6` (chốt chặn cuối).

## Benchmark plan

| Query | Vì sao chọn |
|---|---|
| Research GraphRAG state-of-the-art and write a 500-word summary | cần tổng hợp nhiều nguồn |
| Compare single-agent and multi-agent workflows for customer support | cần đối chiếu quan điểm trái chiều |
| Summarize production guardrails for LLM agents | cần liệt kê đầy đủ, dễ lộ thiếu sót |

Metric: latency (wall-clock), cost (token × pricing table), quality (rubric 0-10 trong
`evaluation/quality.py`, cùng thang với peer-review rubric), citation coverage, failure rate.

Expected outcome: multi-agent thắng rõ ở citation coverage và quality, thua ở latency và
cost — và đó chính là trade-off cần nói ra trong report chứ không giấu đi.
