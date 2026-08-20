# Benchmark Report - single-agent vs multi-agent

_Generated 2026-08-20 05:26 UTC_

## Setup

- **LLM provider**: mock
- **Search provider**: mock
- **Engine**: langgraph
- **Critic**: True
- **Quality scorer**: heuristic rubric (deterministic)
- **Caveat**: offline mock model: latency and cost are not representative, and the templated writer always satisfies the structure/grounding rubric - the quality gap here measures the pipeline shape (retrieval + citation), not model skill
- **Queries**:
  1. Research GraphRAG state-of-the-art and write a 500-word summary
  2. Compare single-agent and multi-agent workflows for customer support
  3. Summarize production guardrails for LLM agents

## Summary (mean per query)

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| single_agent | 0.00 | 0.0000 | 4.2 | 0% | 0% | mean over 3 queries |
| multi_agent | 0.57 | 0.0000 | 10.0 | 100% | 0% | mean over 3 queries |

## Single-agent vs multi-agent

| Dimension | single_agent | multi_agent | Delta |
|---|---:|---:|---:|
| Latency (s) | 0.00 | 0.57 | 1132.80x |
| Cost (USD) | 0.0000 | 0.0000 | n/a |
| Tokens | 156 | 2395 | 15.35x |
| LLM calls | 1 | 4 | 4.00x |
| Quality (0-10) | 4.2 | 10.0 | +5.8 |
| Citation coverage | 0% | 100% | +100% |
| Failure rate | 0% | 0% | +0% |

## Per-query results

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| single_agent[0] | 0.00 | 0.0000 | 4.2 | 0% | 0% | short answer (45 words); no retrieved sources to cite |
| single_agent[1] | 0.00 | 0.0000 | 4.2 | 0% | 0% | short answer (45 words); no retrieved sources to cite |
| single_agent[2] | 0.00 | 0.0000 | 4.2 | 0% | 0% | short answer (43 words); no retrieved sources to cite |
| multi_agent[0] | 1.65 | 0.0000 | 10.0 | 100% | 0% | ok |
| multi_agent[1] | 0.02 | 0.0000 | 10.0 | 100% | 0% | ok |
| multi_agent[2] | 0.02 | 0.0000 | 10.0 | 100% | 0% | ok |

## Route history

- `single_agent`: baseline
- `multi_agent`: researcher -> analyst -> writer -> critic -> done

<details><summary>Sample answer - single_agent</summary>

Research GraphRAG state-of-the-art and write a 500-word summary

Single-agent answer: the topic involves several trade-offs and the right choice depends on data freshness, budget, and latency targets. Without a retrieval step this answer cannot cite sources, which is exactly the gap the multi-agent workflow closes.

</details>

<details><summary>Sample answer - multi_agent</summary>

## Answer
Research GraphRAG state-of-the-art and write a 500-word summary is best addressed by combining retrieval with an explicit division of labour between specialised steps. [S1] [S2] [S3] [S4]

## Key points
- Ground every claim in retrieved evidence. [S1]
- Keep an explicit controller so each step has one job. [S2]
- Measure latency, cost, and quality before adding agents. [S3]

## Limitations
Offline deterministic model: wording is templated, citations are preserved from the retrieved sources.

## Sources
[S1] Community summarisation for global questions - https://example.org/graphrag/community-summaries
[S2] GraphRAG: knowledge-graph grounded retrieval - https://example.org/graphrag/overview
[S3] GraphRAG vs vector RAG: measured trade-offs - https://example.org/graphrag/benchmarks
[S4] Retrieval-augmented generation survey - https://example.org/rag/survey

</details>

