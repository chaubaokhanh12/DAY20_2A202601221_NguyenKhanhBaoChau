"""Agent implementations."""

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent

__all__ = [
    "AnalystAgent",
    "BaseAgent",
    "CriticAgent",
    "ResearcherAgent",
    "SupervisorAgent",
    "WriterAgent",
]
