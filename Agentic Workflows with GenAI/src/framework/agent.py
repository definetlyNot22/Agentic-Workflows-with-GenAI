"""Base Agent abstraction for multi-agent workflows."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel, Field

from .llm_client import LLMClient, LLMResponse
from .tools import Tool, ToolRegistry


class StepLog(BaseModel):
    """Detailed log record of an agent execution step."""
    step_number: int
    agent_name: str
    action: str  # e.g., "THOUGHT", "TOOL_CALL", "TOOL_RESULT", "OUTPUT"
    details: Dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class AgentResponse(BaseModel):
    """Standardized response produced by an agent."""
    agent_name: str
    content: str
    parsed_json: Optional[Dict[str, Any]] = None
    tool_calls_made: List[Dict[str, Any]] = Field(default_factory=list)
    steps: List[StepLog] = Field(default_factory=list)
    is_simulated: bool = False


class Agent:
    """Autonomous agent capable of reasoning, tool use, and structured outputs."""

    def __init__(
        self,
        name: str,
        role: str,
        system_instruction: str,
        llm_client: LLMClient,
        tools: Optional[List[Tool]] = None,
        max_tool_iterations: int = 5,
    ) -> None:
        self.name = name
        self.role = role
        self.system_instruction = system_instruction
        self.llm_client = llm_client
        self.tools = tools or []
        self.tool_map: Dict[str, Tool] = {t.name: t for t in self.tools}
        self.max_tool_iterations = max_tool_iterations

    def get_tool_callables(self) -> List[Callable[..., Any]]:
        """Extract callables for GenAI tool binding."""
        return [t.func for t in self.tools]

    def invoke(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> AgentResponse:
        """Execute the agent turn with contextual memory and autonomous tool invocation."""
        steps: List[StepLog] = []
        tool_calls_made: List[Dict[str, Any]] = []

        context_str = ""
        if context:
            context_str = "\n\n### Current Shared Workflow Context:\n" + json.dumps(context, indent=2)

        prompt = f"{input_text}{context_str}"
        tool_callables = self.get_tool_callables()

        # Initial LLM Invocation
        llm_resp: LLMResponse = self.llm_client.generate(
            prompt=prompt,
            system_instruction=f"Role: {self.role}\n{self.system_instruction}",
            tools=tool_callables if tool_callables else None,
            response_schema=response_schema,
        )

        steps.append(
            StepLog(
                step_number=len(steps) + 1,
                agent_name=self.name,
                action="INITIAL_REASONING",
                message=f"Agent '{self.name}' initiated reasoning.",
            )
        )

        # Handle tool execution loop if live tool calls were returned
        iteration = 0
        current_text = llm_resp.text

        while llm_resp.tool_calls and iteration < self.max_tool_iterations:
            iteration += 1
            for tc in llm_resp.tool_calls:
                fn_name = tc.get("name")
                fn_args = tc.get("args", {})
                tool = self.tool_map.get(fn_name)

                steps.append(
                    StepLog(
                        step_number=len(steps) + 1,
                        agent_name=self.name,
                        action="TOOL_CALL",
                        details={"tool": fn_name, "args": fn_args},
                        message=f"Invoking tool '{fn_name}' with args {fn_args}",
                    )
                )

                if tool:
                    result = tool.execute(**fn_args)
                else:
                    result = f"Error: Tool '{fn_name}' not available to this agent."

                tool_calls_made.append({"tool": fn_name, "args": fn_args, "result": result})

                steps.append(
                    StepLog(
                        step_number=len(steps) + 1,
                        agent_name=self.name,
                        action="TOOL_RESULT",
                        details={"tool": fn_name, "result": str(result)},
                        message=f"Received result from '{fn_name}'",
                    )
                )

                # Feed result back to model
                continuation_prompt = (
                    f"Tool '{fn_name}' returned: {result}\n\n"
                    "Incorporate this result and continue your task to completion."
                )
                llm_resp = self.llm_client.generate(
                    prompt=continuation_prompt,
                    system_instruction=self.system_instruction,
                    tools=tool_callables,
                )
                current_text = llm_resp.text

        # Parse JSON if requested or if output contains JSON block
        parsed = None
        if response_schema:
            try:
                parsed = json.loads(current_text)
            except Exception:
                pass
        elif current_text.strip().startswith("{") and current_text.strip().endswith("}"):
            try:
                parsed = json.loads(current_text)
            except Exception:
                pass

        steps.append(
            StepLog(
                step_number=len(steps) + 1,
                agent_name=self.name,
                action="OUTPUT",
                message=f"Agent '{self.name}' completed task.",
            )
        )

        return AgentResponse(
            agent_name=self.name,
            content=current_text,
            parsed_json=parsed,
            tool_calls_made=tool_calls_made,
            steps=steps,
            is_simulated=llm_resp.is_simulated,
        )
