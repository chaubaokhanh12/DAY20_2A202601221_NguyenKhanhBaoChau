"""Domain-specific errors for the lab."""


class LabError(Exception):
    """Base error for the lab package."""


class StudentTodoError(LabError):
    """Kept for backwards compatibility with the starter skeleton.

    Nothing in `src/` raises it any more: every TODO of the starter repo is implemented.
    """


class AgentExecutionError(LabError):
    """Raised when an agent fails after retries/fallbacks."""


class ValidationError(LabError):
    """Raised when state or output validation fails."""


class ProviderError(LabError):
    """Raised when an external provider (LLM, search) call fails."""


class LLMError(ProviderError):
    """Raised when the LLM provider fails after retries."""


class SearchError(ProviderError):
    """Raised when the search provider fails after retries."""


class BudgetExceededError(LabError):
    """Raised when a run exceeds its iteration / time / cost budget."""
