"""Search client abstraction for `ResearcherAgent`.

`MockSearchClient` scores a local corpus so the lab runs offline; `TavilySearchClient`
talks to a real provider when `TAVILY_API_KEY` is present. Both return the same schema
with stable `[S#]` citation refs, so no agent code changes between modes.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from multi_agent_research_lab.core.config import SearchProvider, Settings, get_settings
from multi_agent_research_lab.core.errors import SearchError
from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.services.mock_corpus import MOCK_CORPUS

logger = logging.getLogger(__name__)

TAVILY_ENDPOINT = "https://api.tavily.com/search"
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "best",
    "by",
    "compare",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "research",
    "state",
    "summarize",
    "the",
    "to",
    "vs",
    "what",
    "when",
    "why",
    "with",
    "write",
    "word",
    "words",
}
_TOKEN_PATTERN = re.compile(r"[a-z0-9\-]+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_PATTERN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]


def assign_refs(documents: list[SourceDocument], start: int = 1) -> list[SourceDocument]:
    """Attach stable `[S1]`, `[S2]`... markers used for citations everywhere downstream."""

    for index, doc in enumerate(documents, start=start):
        doc.ref = f"[S{index}]"
    return documents


class SearchClient(ABC):
    """Provider-agnostic search client."""

    name: str = "search"

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""


class MockSearchClient(SearchClient):
    """Deterministic keyword-scored retrieval over a small curated corpus."""

    name = "mock"

    def __init__(self, corpus: list[dict[str, object]] | None = None) -> None:
        self.corpus = corpus if corpus is not None else MOCK_CORPUS

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        tokens = _tokenize(query)
        scored: list[tuple[float, int, SourceDocument]] = []
        for index, entry in enumerate(self.corpus):
            title = str(entry["title"])
            snippet = str(entry["snippet"])
            tags = [str(tag) for tag in entry.get("tags", [])]
            haystack = f"{title} {snippet} {' '.join(tags)}".lower()
            score = 0.0
            for token in tokens:
                if token in tags:
                    score += 2.0
                elif token in haystack:
                    score += 1.0
            if score <= 0:
                continue
            scored.append(
                (
                    score,
                    -index,  # stable tie-break: earlier corpus entries win
                    SourceDocument(
                        title=title,
                        url=str(entry.get("url") or ""),
                        snippet=snippet,
                        score=score,
                        metadata={"provider": self.name, "tags": tags},
                    ),
                )
            )

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        results = [doc for _, _, doc in scored[:max_results]]
        if not results:  # never hand the Analyst an empty context
            results = [
                SourceDocument(
                    title=str(entry["title"]),
                    url=str(entry.get("url") or ""),
                    snippet=str(entry["snippet"]),
                    score=0.0,
                    metadata={"provider": self.name, "fallback": True},
                )
                for entry in self.corpus[:max_results]
            ]
        return assign_refs(results)


class TavilySearchClient(SearchClient):
    """Minimal Tavily client over stdlib HTTP (no extra dependency)."""

    name = "tavily"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.tavily_api_key:
            raise SearchError("TAVILY_API_KEY is not set; use SEARCH_PROVIDER=mock for offline.")

    def _ssl_context(self) -> ssl.SSLContext:
        # macOS python.org builds do not use the system trust store; prefer certifi when present.
        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            return ssl.create_default_context()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        payload = json.dumps(
            {
                "api_key": self.settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            }
        ).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - fixed https endpoint
            TAVILY_ENDPOINT, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(  # noqa: S310
                request,
                timeout=self.settings.request_timeout_seconds,
                context=self._ssl_context(),
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SearchError(f"Tavily search failed: {exc!r}") from exc

        documents = [
            SourceDocument(
                title=item.get("title") or "Untitled",
                url=item.get("url"),
                snippet=(item.get("content") or "")[:600],
                score=item.get("score"),
                metadata={"provider": self.name},
            )
            for item in body.get("results", [])[:max_results]
        ]
        if not documents:
            raise SearchError("Tavily returned no results.")
        return assign_refs(documents)


def get_search_client(settings: Settings | None = None) -> SearchClient:
    """Factory: real provider when credentials exist, deterministic mock otherwise."""

    settings = settings or get_settings()
    provider = settings.resolved_search_provider()
    if provider is SearchProvider.TAVILY:
        try:
            return TavilySearchClient(settings)
        except SearchError:
            logger.warning("Tavily unavailable - falling back to MockSearchClient.")
    elif settings.search_provider is SearchProvider.AUTO:
        logger.warning("No TAVILY_API_KEY found - falling back to MockSearchClient.")
    return MockSearchClient()
