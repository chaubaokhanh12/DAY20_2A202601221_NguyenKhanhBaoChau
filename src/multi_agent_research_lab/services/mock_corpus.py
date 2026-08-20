"""Offline corpus used by `MockSearchClient`.

Keeping the corpus in its own module makes the lab runnable (and gradable) without any
network access or API key, while still exercising real retrieval logic: scoring,
ranking, deduplication, and citation capture.
"""

from typing import Any

MOCK_CORPUS: list[dict[str, Any]] = [
    {
        "title": "GraphRAG: knowledge-graph grounded retrieval",
        "url": "https://example.org/graphrag/overview",
        "snippet": (
            "GraphRAG builds an entity-relation graph over the corpus, then answers global "
            "questions by summarising communities in the graph instead of top-k chunks."
        ),
        "tags": ["graphrag", "rag", "retrieval", "knowledge graph", "state-of-the-art"],
    },
    {
        "title": "GraphRAG vs vector RAG: measured trade-offs",
        "url": "https://example.org/graphrag/benchmarks",
        "snippet": (
            "Graph-based retrieval improves multi-hop and summarisation questions but adds "
            "indexing cost; vector RAG stays cheaper for narrow factoid lookups."
        ),
        "tags": ["graphrag", "benchmark", "rag", "cost", "latency", "comparison"],
    },
    {
        "title": "Community summarisation for global questions",
        "url": "https://example.org/graphrag/community-summaries",
        "snippet": (
            "Hierarchical community summaries let a system answer 'what are the themes' "
            "queries that plain chunk retrieval cannot cover."
        ),
        "tags": ["graphrag", "summary", "retrieval", "state-of-the-art"],
    },
    {
        "title": "Building effective agents: start simple",
        "url": "https://example.org/agents/effective",
        "snippet": (
            "Most production wins come from a single well-prompted model with tools; add "
            "orchestration only when a task decomposes into clearly separable roles."
        ),
        "tags": ["agents", "multi-agent", "single-agent", "design", "workflow"],
    },
    {
        "title": "Supervisor / worker orchestration patterns",
        "url": "https://example.org/agents/supervisor-pattern",
        "snippet": (
            "A supervisor routes work to specialised workers and owns the stop condition, "
            "which keeps each worker prompt short and independently testable."
        ),
        "tags": ["multi-agent", "supervisor", "routing", "orchestration", "handoff"],
    },
    {
        "title": "Multi-agent handoffs in customer support",
        "url": "https://example.org/agents/support-handoff",
        "snippet": (
            "Support deployments route triage, retrieval, and resolution to separate agents; "
            "handoff quality depends on how much context the shared state carries."
        ),
        "tags": ["customer support", "multi-agent", "handoff", "workflow", "single-agent"],
    },
    {
        "title": "When multi-agent systems hurt",
        "url": "https://example.org/agents/anti-patterns",
        "snippet": (
            "Extra agents multiply latency and token cost, and error compounds across hops; "
            "for single-step tasks a single agent is both cheaper and more accurate."
        ),
        "tags": ["multi-agent", "anti-pattern", "cost", "latency", "failure", "comparison"],
    },
    {
        "title": "Production guardrails for LLM agents",
        "url": "https://example.org/guardrails/production",
        "snippet": (
            "Minimum viable guardrails: max iterations, wall-clock timeout, bounded retry "
            "with fallback, schema validation of every agent output, and a cost ceiling."
        ),
        "tags": ["guardrails", "production", "timeout", "retry", "validation", "safety"],
    },
    {
        "title": "Loop and budget control for agent runtimes",
        "url": "https://example.org/guardrails/loop-control",
        "snippet": (
            "Recursion limits plus per-run token budgets stop runaway loops; degrade to the "
            "best partial answer instead of raising to the end user."
        ),
        "tags": ["guardrails", "loop", "budget", "cost", "fallback", "production"],
    },
    {
        "title": "Validating agent output with schemas",
        "url": "https://example.org/guardrails/schema-validation",
        "snippet": (
            "Typed outputs catch malformed handoffs early; unvalidated free text is the most "
            "common source of silent multi-agent failure."
        ),
        "tags": ["guardrails", "validation", "schema", "pydantic", "handoff"],
    },
    {
        "title": "Tracing agent runs end to end",
        "url": "https://example.org/observability/tracing",
        "snippet": (
            "A trace per run with one span per agent call - inputs, outputs, tokens, latency - "
            "is what makes a multi-agent failure explainable after the fact."
        ),
        "tags": ["tracing", "observability", "langsmith", "debug", "latency"],
    },
    {
        "title": "Evaluating agent systems beyond vibes",
        "url": "https://example.org/evaluation/agent-metrics",
        "snippet": (
            "Report latency, cost, task success, and citation coverage per query; a single "
            "cherry-picked transcript is not evidence."
        ),
        "tags": ["benchmark", "evaluation", "metrics", "quality", "citation", "comparison"],
    },
    {
        "title": "RAG vs fine-tuning for domain adaptation",
        "url": "https://example.org/rag/vs-finetuning",
        "snippet": (
            "Retrieval fits fast-changing knowledge and gives citations; fine-tuning fits "
            "stable style, format, and latency-sensitive behaviour."
        ),
        "tags": ["rag", "fine-tuning", "domain adaptation", "comparison", "cost"],
    },
    {
        "title": "Retrieval-augmented generation survey",
        "url": "https://example.org/rag/survey",
        "snippet": (
            "RAG reduces hallucination by grounding generation in retrieved documents, at the "
            "price of retrieval quality becoming the dominant failure mode."
        ),
        "tags": ["rag", "retrieval", "hallucination", "survey", "state-of-the-art"],
    },
    {
        "title": "Cost modelling for LLM pipelines",
        "url": "https://example.org/ops/cost-model",
        "snippet": (
            "Cost scales with calls x context size; multi-agent pipelines typically spend "
            "2-4x the tokens of a single-agent baseline for the same query."
        ),
        "tags": ["cost", "tokens", "benchmark", "multi-agent", "latency"],
    },
    {
        "title": "Failure modes of orchestrated LLM workflows",
        "url": "https://example.org/agents/failure-modes",
        "snippet": (
            "Common failures: routing oscillation, context loss at handoff, silent retrieval "
            "misses, and unbounded retries after a provider outage."
        ),
        "tags": ["failure", "multi-agent", "routing", "handoff", "retry", "guardrails"],
    },
]
