# Exit ticket

## 1. Case nào nên dùng multi-agent? Vì sao?

Khi task **tách được thành các vai trò có tiêu chí đánh giá khác nhau** và output phải
kiểm chứng được. Ví dụ chính là bài lab này: tìm nguồn (đánh giá bằng recall/độ liên
quan), phân tích (đánh giá bằng việc nêu đúng mâu thuẫn), viết (đánh giá bằng cấu trúc +
citation), kiểm tra (đánh giá bằng citation coverage). Tách vai trò cho ba lợi ích đo được:

1. **Có điểm chèn guardrail.** Critic là một node riêng nên có thể chặn draft trước khi tới
   người dùng. Trong single-agent không có "chỗ" nào để chặn.
2. **Prompt ngắn, ít loãng context.** Mỗi agent chỉ nhận đúng phần state nó cần.
3. **Debug được.** Trace chỉ ra bước nào hỏng; benchmark đo được rằng citation coverage đi
   từ 0% → 100% (xem `reports/benchmark_report.md`).

Các case thực tế tương tự: research assistant có yêu cầu trích dẫn, pipeline soạn tài liệu
tuân thủ (compliance), triage → retrieval → resolution trong customer support.

## 2. Case nào không nên dùng multi-agent? Vì sao?

Khi task là **một bước duy nhất, latency-sensitive, hoặc không có tiêu chí kiểm chứng
riêng cho từng bước**. Ví dụ: dịch một đoạn văn, phân loại intent, trả lời FAQ từ một tài
liệu duy nhất, autocomplete.

Lý do rất cụ thể, đo được ngay trong repo này: multi-agent tốn **4x LLM call và ~15x token**
so với baseline cho cùng một query. Với task một bước, số tiền và độ trễ đó không đổi lại
được chất lượng nào cả — tệ hơn, mỗi handoff là một chỗ có thể mất context, và mỗi vòng
routing là một chỗ có thể oscillate (xem `docs/failure_modes.md`).

**Quy tắc rút ra:** thêm agent chỉ khi trả lời được câu "agent này chịu trách nhiệm cho
metric nào mà agent khác không chịu?". Nếu không trả lời được thì đó là chi phí thuần.
