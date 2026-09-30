"""Researcher Agent equipped with external search and retrieval tools."""

from __future__ import annotations

from typing import List, Optional

from ..framework.agent import Agent
from ..framework.llm_client import LLMClient
from ..framework.tools import Tool, web_search

RESEARCHER_INSTRUCTION = """
You are a Lead Intelligence Researcher.
Your mission is to gather precise, up-to-date facts, industry statistics, and technical documentation.

Guidelines:
1. When external data is needed, invoke the `web_search` tool.
2. Ground all claims in retrieved evidence. Never fabricate statistics or citations.
3. Summarize your findings with high signal-to-noise ratio, highlighting key trends and hard data points.
"""


class ResearcherAgent(Agent):
    """Executes research and information gathering using retrieval tools."""

    def __init__(self, llm_client: LLMClient, custom_tools: Optional[List[Tool]] = None) -> None:
        tools = custom_tools if custom_tools is not None else [
            Tool("web_search", "Search the web for up-to-date information", web_search)
        ]
        super().__init__(
            name="Researcher",
            role="Lead Research & Retrieval Specialist",
            system_instruction=RESEARCHER_INSTRUCTION,
            llm_client=llm_client,
            tools=tools,
        )
