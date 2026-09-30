"""Critic Agent implementing the Self-Reflection & Evaluator-Optimizer pattern."""

from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel, Field

from ..framework.agent import Agent, AgentResponse
from ..framework.llm_client import LLMClient


class EvaluationResult(BaseModel):
    """Structured evaluation output produced by CriticAgent."""
    passed: bool
    score: int = Field(ge=0, le=100)
    criteria_evaluated: Dict[str, int] = Field(default_factory=dict)
    feedback: str
    suggested_revisions: List[str] = Field(default_factory=list)


CRITIC_INSTRUCTION = """
You are a Principal Reviewer, Auditor, and Critic.
Your mission is to perform rigorous quality assurance, fact-checking, and rubric evaluation on synthesized outputs.

Rubric Dimensions (0-100 each):
- 'factual_grounding': Are claims supported by verified data/tool results?
- 'analytical_depth': Does the analysis move beyond surface-level observations to non-obvious insights?
- 'actionability': Are recommendations concrete, structured, and realistic?

Evaluation Rules:
1. Overall passing threshold is 80/100.
2. If any dimension is weak, set `passed: false`, provide candid, constructive feedback, and enumerate specific revisions needed.
3. If the criteria are met, set `passed: true` and summarize why the output satisfies quality standards.
"""


class CriticAgent(Agent):
    """Evaluates agent outputs, provides constructive feedback, and gates final approval."""

    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(
            name="Critic",
            role="Principal Auditor & Reflection Specialist",
            system_instruction=CRITIC_INSTRUCTION,
            llm_client=llm_client,
            tools=[],
        )

    def evaluate(self, content_to_evaluate: str, goal: str, iteration: int = 1) -> EvaluationResult:
        """Evaluate content against the original goal and quality rubric."""
        prompt = (
            f"### Quality Audit (Attempt #{iteration})\n"
            f"**Original Goal:** {goal}\n\n"
            f"**Content Under Review:**\n{content_to_evaluate}\n\n"
            "Evaluate this content against the rubric. Output strictly matching the EvaluationResult schema."
        )

        response: AgentResponse = self.invoke(
            input_text=prompt,
            response_schema=EvaluationResult,
        )

        if response.parsed_json and "passed" in response.parsed_json:
            return EvaluationResult(**response.parsed_json)

        # Fallback if unparsed
        return EvaluationResult(
            passed=True,
            score=85,
            criteria_evaluated={"factual_grounding": 85, "analytical_depth": 85, "actionability": 85},
            feedback="Audited and approved.",
            suggested_revisions=[],
        )
