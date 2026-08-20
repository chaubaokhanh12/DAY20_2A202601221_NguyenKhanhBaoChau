from multi_agent_research_lab.services.llm_client import (
    MockLLMClient,
    approx_tokens,
    estimate_cost,
    get_llm_client,
)
from multi_agent_research_lab.services.search_client import (
    MockSearchClient,
    assign_refs,
    get_search_client,
)
from tests.conftest import offline_settings


def test_mock_search_is_relevant_and_deterministic(search: MockSearchClient) -> None:
    first = search.search("production guardrails for LLM agents", max_results=3)
    second = search.search("production guardrails for LLM agents", max_results=3)

    assert [d.title for d in first] == [d.title for d in second]
    assert len(first) == 3
    assert any("guardrail" in d.title.lower() or "guardrail" in d.snippet.lower() for d in first)
    assert [d.ref for d in first] == ["[S1]", "[S2]", "[S3]"]


def test_mock_search_never_returns_empty(search: MockSearchClient) -> None:
    results = search.search("zzz qqq unrelated gibberish", max_results=2)
    assert len(results) == 2
    assert all(doc.ref for doc in results)


def test_assign_refs_is_one_based() -> None:
    docs = assign_refs(MockSearchClient().search("rag", max_results=2))
    assert [doc.ref for doc in docs] == ["[S1]", "[S2]"]


def test_mock_llm_is_role_aware_and_keeps_citations(llm: MockLLMClient) -> None:
    prompt = "Question: what is RAG?\nSources:\n[S1] Retrieval survey (https://x.test)\n"
    writer = llm.complete("ROLE: writer\nwrite", prompt)
    analyst = llm.complete("ROLE: analyst\nanalyse", prompt)

    assert "[S1]" in writer.content
    assert "## Answer" in writer.content
    assert writer.content != analyst.content
    assert "Key claims" in analyst.content
    assert writer.input_tokens and writer.output_tokens
    assert writer.cost_usd == 0.0


def test_mock_llm_is_deterministic(llm: MockLLMClient) -> None:
    args = ("ROLE: writer\nwrite", "Question: q\n[S1] Title\n")
    assert llm.complete(*args).content == llm.complete(*args).content


def test_cost_estimation_uses_pricing_table() -> None:
    assert estimate_cost("gpt-4o-mini", 1_000_000, 0) == 0.15
    assert estimate_cost("gpt-4o", 0, 1_000_000) == 10.0
    assert estimate_cost("mock", 10_000, 10_000) == 0.0
    assert approx_tokens("a" * 400) == 100


def test_factories_fall_back_to_mocks_without_keys() -> None:
    offline = offline_settings(LLM_PROVIDER="auto", SEARCH_PROVIDER="auto")
    assert isinstance(get_llm_client(offline), MockLLMClient)
    assert isinstance(get_search_client(offline), MockSearchClient)
