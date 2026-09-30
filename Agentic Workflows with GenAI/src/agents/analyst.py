"""Analyst Agent equipped with computational tools (Calculator and Python REPL)."""

from __future__ import annotations

from typing import List, Optional

from ..framework.agent import Agent
from ..framework.llm_client import LLMClient
from ..framework.tools import Tool, calculator, python_repl

ANALYST_INSTRUCTION = """
You are a Quantitative Analyst and Computational Engine.
Your mission is to perform numerical modeling, growth forecasting, statistical calculations, and code verification.

Guidelines:
1. Always verify numbers using the `calculator` or `python_repl` tool rather than doing mental math.
2. Structure calculations clearly, explaining assumptions, formulas, and results.
3. Deliver mathematically sound insights and actionable conclusions.
"""


class AnalystAgent(Agent):
    """Executes quantitative, computational, and code-based analysis."""

    def __init__(self, llm_client: LLMClient, custom_tools: Optional[List[Tool]] = None) -> None:
        tools = custom_tools if custom_tools is not None else [
            Tool("calculator", "Evaluate mathematical expressions", calculator),
            Tool("python_repl", "Execute Python code and capture output", python_repl),
        ]
        super().__init__(
            name="Analyst",
            role="Quantitative & Computational Analyst",
            system_instruction=ANALYST_INSTRUCTION,
            llm_client=llm_client,
            tools=tools,
        )
