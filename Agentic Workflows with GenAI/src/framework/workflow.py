"""Workflow state machine and orchestration manager."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from .agent import Agent, AgentResponse, StepLog


class WorkflowState(BaseModel):
    """Encapsulates the evolving global state of an agentic workflow."""
    goal: str
    status: str = "PENDING"
    plan: List[Dict[str, Any]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    tool_history: List[Dict[str, Any]] = Field(default_factory=list)
    agent_outputs: Dict[str, str] = Field(default_factory=dict)
    reflections: List[Dict[str, Any]] = Field(default_factory=list)
    final_output: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None


class Workflow:
    """Stateful agentic workflow orchestrator."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self.agents: Dict[str, Agent] = {}
        self.listeners: List[Callable[[str, Any], None]] = []

    def register_agent(self, agent: Agent) -> None:
        """Register an agent into the workflow."""
        self.agents[agent.name] = agent

    def get_agent(self, name: str) -> Optional[Agent]:
        """Retrieve an agent by name."""
        return self.agents.get(name)

    def add_listener(self, listener: Callable[[str, Any], None]) -> None:
        """Add event listener for telemetry callbacks (e.g. CLI rendering)."""
        self.listeners.append(listener)

    def emit(self, event_type: str, data: Any) -> None:
        """Notify listeners of an internal workflow event."""
        for listener in self.listeners:
            try:
                listener(event_type, data)
            except Exception:
                pass

    def run_step(
        self,
        agent_name: str,
        instruction: str,
        state: WorkflowState,
        response_schema: Optional[Any] = None,
    ) -> AgentResponse:
        """Execute a single agent's turn within the workflow state."""
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not registered in workflow '{self.name}'.")

        self.emit("agent_start", {"agent": agent_name, "instruction": instruction})

        # Run agent with accumulated context
        resp = agent.invoke(
            input_text=instruction,
            context=state.context,
            response_schema=response_schema,
        )

        # Update state
        state.agent_outputs[agent_name] = resp.content
        if resp.tool_calls_made:
            state.tool_history.extend(resp.tool_calls_made)

        self.emit("agent_finish", {"agent": agent_name, "response": resp})
        return resp
