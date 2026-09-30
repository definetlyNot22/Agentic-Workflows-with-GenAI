"""Supervisor Agent coordinating the multi-agent workflow lifecycle."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from ..framework.agent import Agent, AgentResponse
from ..framework.llm_client import LLMClient
from ..framework.workflow import Workflow, WorkflowState
from .planner import ExecutionPlan, PlannerAgent
from .researcher import ResearcherAgent
from .analyst import AnalystAgent
from .critic import CriticAgent, EvaluationResult

SUPERVISOR_INSTRUCTION = """
You are the Executive Supervisor and Chief Orchestrator of the Agentic Workflow.
Your role is to coordinate specialized agents, synthesize their intermediate findings,
and produce an authoritative, cohesive final deliverable.
"""


class SupervisorAgent(Agent):
    """Orchestrates planning, agent execution, reflection loops, and final synthesis."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_reflection_loops: int = 2,
    ) -> None:
        super().__init__(
            name="Supervisor",
            role="Executive Workflow Orchestrator",
            system_instruction=SUPERVISOR_INSTRUCTION,
            llm_client=llm_client,
            tools=[],
        )
        self.max_reflection_loops = max_reflection_loops

        # Sub-agents
        self.planner = PlannerAgent(llm_client=llm_client)
        self.researcher = ResearcherAgent(llm_client=llm_client)
        self.analyst = AnalystAgent(llm_client=llm_client)
        self.critic = CriticAgent(llm_client=llm_client)

    def execute_workflow(
        self,
        goal: str,
        on_event: Optional[Callable[[str, Any], None]] = None,
    ) -> WorkflowState:
        """Run the end-to-end agentic workflow."""
        state = WorkflowState(goal=goal)

        def emit(event_type: str, data: Any) -> None:
            if on_event:
                on_event(event_type, data)

        emit("workflow_start", {"goal": goal})

        # --- Phase 1: Planning & Decomposition ---
        state.status = "PLANNING"
        emit("phase_change", {"phase": "Planning & Task Decomposition"})
        plan: ExecutionPlan = self.planner.generate_plan(goal)
        state.plan = [task.model_dump() for task in plan.tasks]
        emit("plan_generated", {"plan": state.plan})

        # --- Phase 2: Autonomous Sub-Task Execution ---
        state.status = "EXECUTING"
        emit("phase_change", {"phase": "Autonomous Sub-Task Execution"})

        task_results: List[Dict[str, Any]] = []

        for task in plan.tasks:
            agent_name = task.agent.strip().lower()
            emit("task_start", {"step": task.step, "title": task.title, "agent": task.agent})

            instruction = (
                f"Task: {task.title}\n"
                f"Objective: {task.description}\n"
                f"Target Tool to use if appropriate: {task.tool}\n"
                f"Overall Goal: {goal}"
            )

            # Route to appropriate specialized agent
            if "research" in agent_name:
                resp = self.researcher.invoke(instruction, context=state.context)
            elif "analyst" in agent_name:
                resp = self.analyst.invoke(instruction, context=state.context)
            elif "critic" in agent_name:
                resp = self.critic.invoke(instruction, context=state.context)
            else:
                resp = self.invoke(instruction, context=state.context)

            task_results.append({
                "step": task.step,
                "title": task.title,
                "agent": task.agent,
                "content": resp.content,
                "tool_calls": resp.tool_calls_made,
            })

            # Update shared context scratchpad
            state.context[f"step_{task.step}_{task.title}"] = resp.content
            if resp.tool_calls_made:
                state.tool_history.extend(resp.tool_calls_made)

            emit("task_finish", {
                "step": task.step,
                "title": task.title,
                "agent": task.agent,
                "response": resp,
            })

        # --- Phase 3: Initial Synthesis ---
        emit("phase_change", {"phase": "Executive Synthesis"})
        synthesis_prompt = (
            f"Compile a comprehensive, publication-grade executive strategic report for the goal:\n"
            f"'{goal}'\n\n"
            f"Incorporate all verified findings from the specialized agent steps:\n"
            + "\n".join(f"### {tr['step']}. {tr['title']} ({tr['agent']})\n{tr['content']}\n" for tr in task_results)
        )
        synthesis_resp = self.invoke(synthesis_prompt)
        current_draft = synthesis_resp.content

        # --- Phase 4: Self-Reflection & Quality Gating ---
        state.status = "REFLECTING"
        emit("phase_change", {"phase": "Quality Audit & Reflection Loop"})

        iteration = 1
        while iteration <= self.max_reflection_loops + 1:
            emit("reflection_start", {"iteration": iteration})
            eval_result: EvaluationResult = self.critic.evaluate(
                content_to_evaluate=current_draft,
                goal=goal,
                iteration=iteration,
            )
            state.reflections.append(eval_result.model_dump())
            emit("reflection_finish", {"iteration": iteration, "evaluation": eval_result})

            if eval_result.passed or iteration > self.max_reflection_loops:
                break

            # Iterative Refinement based on Critic feedback
            emit("phase_change", {"phase": f"Refining Output (Reflection Iteration {iteration})"})
            refine_prompt = (
                f"### Refinement Required (Attempt #{iteration})\n"
                f"Your previous draft was audited by the Critic and requires revisions.\n"
                f"**Audit Score:** {eval_result.score}/100\n"
                f"**Critic Feedback:** {eval_result.feedback}\n"
                f"**Mandatory Revisions:**\n"
                + "\n".join(f"- {r}" for r in eval_result.suggested_revisions)
                + f"\n\n**Previous Draft:**\n{current_draft}\n\n"
                "Produce the revised, upgraded version addressing all criticisms."
            )
            refined_resp = self.invoke(refine_prompt)
            current_draft = refined_resp.content
            iteration += 1

        state.final_output = current_draft
        state.status = "COMPLETED"
        state.completed_at = time.time()
        emit("workflow_finish", {"state": state})
        return state
