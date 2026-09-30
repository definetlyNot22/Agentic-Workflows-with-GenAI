"""Comprehensive unit and integration tests for the GenAI Agentic Workflow framework."""

import os
import pytest
from src.framework.llm_client import LLMClient
from src.framework.tools import Tool, ToolRegistry, calculator, python_repl, web_search, file_writer, get_default_registry
from src.framework.agent import Agent
from src.agents.planner import PlannerAgent, ExecutionPlan
from src.agents.researcher import ResearcherAgent
from src.agents.analyst import AnalystAgent
from src.agents.critic import CriticAgent, EvaluationResult
from src.agents.supervisor import SupervisorAgent
from src.framework.workflow import WorkflowState


@pytest.fixture
def sim_client():
    """Provides an offline simulation LLMClient for fast, deterministic unit testing."""
    return LLMClient(force_simulation=True)


def test_tool_calculator():
    """Verify safe calculator execution."""
    res1 = calculator("25 * 4")
    assert res1 == "100"
    res2 = calculator("math.sqrt(144)")
    assert res2 == "12.0"


def test_tool_python_repl():
    """Verify Python code execution in local sandbox."""
    code = "vals = [10, 20, 30]\nprint(f'Sum: {sum(vals)}')"
    output = python_repl(code)
    assert "Sum: 60" in output


def test_tool_web_search():
    """Verify web search tool retrieval."""
    output = web_search("enterprise agentic AI market")
    assert "Search Results for" in output
    assert "CAGR" in output or "market" in output.lower()


def test_tool_file_writer(tmp_path):
    """Verify file writer tool."""
    target_file = tmp_path / "test_artifact.txt"
    content = "Agentic workflows deliver autonomous reasoning."
    result = file_writer(str(target_file), content)
    assert "Successfully saved" in result
    assert target_file.read_text(encoding="utf-8") == content


def test_tool_registry():
    """Verify tool registration and dynamic dispatch."""
    registry = get_default_registry()
    assert registry.get("calculator") is not None
    assert registry.get("python_repl") is not None
    assert registry.get("web_search") is not None
    assert registry.get("file_writer") is not None

    res = registry.execute("calculator", expression="10 + 5")
    assert res == "15"


def test_planner_agent(sim_client):
    """Verify Planner decomposes a high-level goal into structured tasks."""
    planner = PlannerAgent(llm_client=sim_client)
    plan: ExecutionPlan = planner.generate_plan("Build an autonomous agent for financial forecasting")
    assert len(plan.tasks) >= 3
    assert plan.tasks[0].step == 1
    assert any(task.agent == "Researcher" for task in plan.tasks)
    assert any(task.agent == "Analyst" for task in plan.tasks)


def test_researcher_agent(sim_client):
    """Verify Researcher agent can be invoked and reports findings."""
    researcher = ResearcherAgent(llm_client=sim_client)
    resp = researcher.invoke("Investigate current trends in AI agents")
    assert resp.agent_name == "Researcher"
    assert len(resp.content) > 50


def test_analyst_agent(sim_client):
    """Verify Analyst agent computational reasoning."""
    analyst = AnalystAgent(llm_client=sim_client)
    resp = analyst.invoke("Calculate growth rates and verify code")
    assert resp.agent_name == "Analyst"
    assert len(resp.content) > 20


def test_critic_reflection_loop(sim_client):
    """Verify Critic agent reflection and evaluation logic."""
    critic = CriticAgent(llm_client=sim_client)
    
    # First attempt: expect critical audit & revision suggestions
    res1: EvaluationResult = critic.evaluate(
        content_to_evaluate="AI agents are useful.",
        goal="Provide in-depth market analysis",
        iteration=1,
    )
    assert isinstance(res1.score, int)
    assert len(res1.criteria_evaluated) > 0
    assert len(res1.feedback) > 0

    # Revised attempt: expect higher score or approval
    res2: EvaluationResult = critic.evaluate(
        content_to_evaluate="Revised comprehensive market analysis with $28.5B market size at 42% CAGR...",
        goal="Provide in-depth market analysis",
        iteration=2,
    )
    assert res2.passed is True
    assert res2.score >= 80


def test_end_to_end_workflow(sim_client, tmp_path):
    """Verify complete multi-agent workflow lifecycle."""
    supervisor = SupervisorAgent(llm_client=sim_client, max_reflection_loops=1)
    events_captured = []

    def on_event(event_type, data):
        events_captured.append(event_type)

    state: WorkflowState = supervisor.execute_workflow(
        goal="Develop a strategic roadmap for deploying GenAI agents",
        on_event=on_event,
    )

    assert state.status == "COMPLETED"
    assert len(state.plan) > 0
    assert len(state.reflections) > 0
    assert state.final_output is not None
    assert "workflow_start" in events_captured
    assert "plan_generated" in events_captured
    assert "reflection_start" in events_captured
    assert "workflow_finish" in events_captured
