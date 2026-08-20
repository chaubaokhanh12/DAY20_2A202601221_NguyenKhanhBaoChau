# Failure modes & fixes

Deliverable #4 của lab: những kiểu hỏng đã gặp/đã lường trước khi xây hệ multi-agent này,
cách hệ thống hiện tại chặn chúng, và test nào chứng minh điều đó.

## 1. Routing oscillation (agent bị gọi lặp vô hạn)

**Triệu chứng.** Researcher trả về rỗng → Supervisor thấy `sources` rỗng → gọi lại
Researcher → lặp mãi cho tới khi hết quota.

**Vì sao xảy ra.** Routing chỉ nhìn "field còn thiếu" mà không nhìn "đã thử bao nhiêu lần".

**Fix.** `SupervisorAgent.decide()` đếm `route_history`: cùng một route ≥ 2 lần thì
degrade sang bước tiếp theo thay vì thử lại, cộng thêm hai chốt chặn cứng là
`max_iterations` và `max_cost_usd`. LangGraph còn có `recursion_limit` là lưới cuối.

**Test.** `tests/test_supervisor.py::test_repeated_researcher_failure_degrades_instead_of_looping`,
`tests/test_workflow.py::test_iteration_budget_is_never_exceeded`.

## 2. Citation ảo (hallucinated `[S#]`)

**Triệu chứng.** Writer trích `[S9]` trong khi chỉ có 5 nguồn được retrieve — câu trả lời
"trông có bằng chứng" nhưng nguồn không tồn tại. Đây là failure mode nguy hiểm nhất vì nó
làm tăng độ tin cậy giả.

**Fix.** Hai lớp: (a) Writer không tự viết danh sách nguồn, `## Sources` được append
deterministic từ `state.sources`; (b) Critic so tập marker trong answer với
`state.source_refs()`, marker lạ → `approved=False` → Supervisor cho Writer sửa lại một lần.

**Test.** `tests/test_agents.py::test_critic_flags_hallucinated_citations_and_low_coverage`.

## 3. Mất context ở handoff

**Triệu chứng.** Writer chỉ nhận `analysis_notes`, không thấy snippet gốc, nên không thể
gắn citation chính xác.

**Fix.** Shared state chứa cả `sources`, `research_notes` và `analysis_notes`;
`state.sources_block()` render nguồn kèm marker cho mọi prompt cần nó. Handoff đi qua
Pydantic model nên không có field nào bị "rơi" âm thầm.

**Test.** `tests/test_agents.py::test_writer_appends_resolvable_sources`.

## 4. Provider outage (LLM hoặc search chết giữa chừng)

**Triệu chứng.** Một lần 500 từ OpenAI làm hỏng toàn bộ run; người dùng nhận traceback.

**Fix.** Retry có backoff trong `OpenAIClient`, rồi `LLMError` được `BaseAgent.think()`
bắt lại và thay bằng nhánh offline deterministic; lỗi được ghi vào `state.errors` và
status trở thành `degraded` — người dùng vẫn nhận được câu trả lời **và** biết nó suy giảm.
Search chết thì retrieval rỗng, Supervisor tự degrade sang Writer.

**Test.** `tests/test_workflow.py::test_llm_outage_degrades_instead_of_crashing`,
`::test_search_outage_still_produces_an_answer`.

## 5. Chi phí/thời gian trôi không kiểm soát

**Triệu chứng.** Multi-agent tốn 4x call so với baseline; thêm một vòng revision nữa là
gấp đôi tiếp, không ai nhận ra cho tới lúc xem hóa đơn.

**Fix.** Mỗi call cộng dồn vào `state.usage` (token + USD theo bảng giá trong
`services/llm_client.py`); Supervisor dừng khi vượt `max_cost_usd`; workflow dừng khi vượt
`timeout_seconds`; benchmark in cost/token của cả hai cấu hình cạnh nhau.

**Test.** `tests/test_supervisor.py::test_cost_budget_stops_the_run`,
`tests/test_workflow.py::test_timeout_guard_stops_the_loop`.

## 6. "Benchmark bằng cảm tính"

**Triệu chứng.** Demo một transcript đẹp rồi kết luận multi-agent tốt hơn.

**Fix.** `evaluation/quality.py` chấm 0-10 bằng rubric đo được (completeness, grounding,
structure, relevance, reliability); `evaluation/benchmark.py` chạy cả hai runner trên cùng
tập query và ghi latency/cost/token/coverage/failure-rate; report in cả delta lẫn per-query.
Runner crash được tính là một data point (`failure_rate=1.0`) chứ không làm hỏng benchmark.

**Test.** `tests/test_evaluation.py::test_run_benchmark_records_a_crash_instead_of_raising`,
`::test_multi_agent_scores_above_baseline`.

## Điều còn thiếu (nếu có thêm thời gian)

- Rerank/verify nguồn trước khi cho Analyst dùng (hiện chỉ dedupe theo URL/title).
- LLM-as-judge chạy mặc định (đã có `LLMJudge`, hiện chỉ dùng khi được truyền vào explicit).
- Trace forward sang LangSmith mới ở mức best-effort, chưa có test tích hợp thật.
- Song song hoá Researcher trên nhiều sub-query để giảm latency.
