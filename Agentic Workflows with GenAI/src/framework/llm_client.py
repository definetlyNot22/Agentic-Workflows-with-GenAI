"""LLM client abstraction for Google GenAI with graceful mock/simulation fallback."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


class LLMResponse(BaseModel):
    """Normalized response from LLM generation."""
    text: str
    model: str
    tool_calls: List[Dict[str, Any]] = []
    finish_reason: Optional[str] = None
    is_simulated: bool = False


class LLMClient:
    """Wrapper client for Google GenAI with live execution and simulation modes."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        force_simulation: Optional[bool] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.force_simulation = (
            force_simulation
            if force_simulation is not None
            else os.getenv("SIMULATION_MODE", "false").lower() in ("1", "true", "yes")
        )
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Google GenAI SDK client if API key is present."""
        if not self.force_simulation and self.api_key and self.api_key != "your_gemini_api_key_here":
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[Warning] Failed to initialize live Google GenAI client: {e}. Falling back to simulation mode.")
                self.client = None
        else:
            self.client = None

    @property
    def is_live(self) -> bool:
        """True if connected to the live Gemini API."""
        return self.client is not None

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Callable[..., Any]]] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate response using Google GenAI or intelligent simulation."""
        if self.is_live:
            return self._generate_live(
                prompt=prompt,
                system_instruction=system_instruction,
                tools=tools,
                response_schema=response_schema,
                temperature=temperature,
            )
        else:
            return self._generate_simulated(
                prompt=prompt,
                system_instruction=system_instruction,
                response_schema=response_schema,
            )

    def _generate_live(
        self,
        prompt: str,
        system_instruction: Optional[str],
        tools: Optional[List[Callable[..., Any]]],
        response_schema: Optional[Type[BaseModel]],
        temperature: float,
    ) -> LLMResponse:
        """Generate using live google-genai SDK."""
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )

        if tools:
            config.tools = tools

        if response_schema:
            config.response_mime_type = "application/json"
            config.response_schema = response_schema

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        tool_calls: List[Dict[str, Any]] = []
        # Extract function calls if present
        if response.function_calls:
            for fc in response.function_calls:
                tool_calls.append({
                    "name": fc.name,
                    "args": dict(fc.args) if hasattr(fc, "args") else {},
                })

        return LLMResponse(
            text=response.text or "",
            model=self.model,
            tool_calls=tool_calls,
            is_simulated=False,
        )

    def _generate_simulated(
        self,
        prompt: str,
        system_instruction: Optional[str],
        response_schema: Optional[Type[BaseModel]],
    ) -> LLMResponse:
        """Intelligent simulation engine for agent workflows without requiring API keys."""
        sys_str = (system_instruction or "").lower()
        p_str = prompt.lower()

        # 1. Planner Agent Simulation
        if "planner" in sys_str or "decompose" in sys_str or "plan" in p_str:
            simulated_plan = {
                "tasks": [
                    {
                        "step": 1,
                        "title": "Market Landscape & Scale Research",
                        "agent": "Researcher",
                        "tool": "web_search",
                        "description": "Gather current market size, growth trajectory, and key player metrics in enterprise agentic AI.",
                    },
                    {
                        "step": 2,
                        "title": "Quantitative & Performance Analysis",
                        "agent": "Analyst",
                        "tool": "calculator",
                        "description": "Calculate compound growth projections and ROI improvements from multi-agent deployment.",
                    },
                    {
                        "step": 3,
                        "title": "Code & Technical Feasibility Check",
                        "agent": "Analyst",
                        "tool": "python_repl",
                        "description": "Execute Python verification of multi-agent state transition probabilities.",
                    },
                    {
                        "step": 4,
                        "title": "Quality Audit & Reflection",
                        "agent": "Critic",
                        "tool": "none",
                        "description": "Evaluate findings for factual rigor, completeness, and strategic clarity.",
                    },
                ]
            }
            if response_schema:
                text_out = json.dumps(simulated_plan)
            else:
                text_out = (
                    "### Actionable Implementation Plan\n\n"
                    "1. **Market Landscape & Scale Research** [Researcher -> web_search]: "
                    "Gather current market size and adoption stats.\n"
                    "2. **Quantitative Analysis** [Analyst -> calculator]: "
                    "Compute 3-year market projections and efficiency gains.\n"
                    "3. **Technical Validation** [Analyst -> python_repl]: "
                    "Run computational verification of agent loop convergence.\n"
                    "4. **Synthesis & Reflection** [Critic]: "
                    "Review findings against high-quality industry benchmarks.\n"
                )
            return LLMResponse(
                text=text_out,
                model=f"{self.model} (simulation)",
                is_simulated=True,
            )

        # 2. Critic Agent Simulation (Reflection & Self-Correction)
        elif "critic" in sys_str or "auditor" in sys_str or "evaluat" in sys_str or "rubric" in sys_str:
            # Check if this is a first-pass critique or a revised critique
            is_revised = any(term in p_str for term in ["attempt #2", "attempt 2", "attempt #3", "attempt 3", "revised", "revision", "addressing all criticisms"])

            if not is_revised:
                # Trigger reflection feedback loop on first attempt to showcase self-correction
                critique = {
                    "passed": False,
                    "score": 68,
                    "criteria_evaluated": {
                        "factual_grounding": 75,
                        "analytical_depth": 65,
                        "actionability": 64,
                    },
                    "feedback": (
                        "The preliminary report outlines broad concepts but lacks specific quantitative projections "
                        "and concrete comparisons of agent orchestration architectures. "
                        "Recommendation: Add concrete mathematical projections and a clear multi-tier architectural breakdown."
                    ),
                    "suggested_revisions": [
                        "Include projected 3-year valuation with compound growth.",
                        "Add explicit comparative breakdown between single-turn LLMs and autonomous multi-agent loops.",
                    ],
                }
            else:
                # Passed quality gate on revised iteration
                critique = {
                    "passed": True,
                    "score": 94,
                    "criteria_evaluated": {
                        "factual_grounding": 95,
                        "analytical_depth": 92,
                        "actionability": 95,
                    },
                    "feedback": (
                        "Exceptional quality. The report now integrates precise market metrics ($28.5B market size at 42% CAGR), "
                        "reproducible mathematical forecasts, and thorough architectural comparison with error reduction data."
                    ),
                    "suggested_revisions": [],
                }

            if response_schema:
                text_out = json.dumps(critique)
            else:
                status_str = "PASSED" if critique["passed"] else "NEEDS REVISION"
                text_out = (
                    f"### Evaluation Result: {status_str} (Score: {critique['score']}/100)\n\n"
                    f"**Feedback:** {critique['feedback']}\n\n"
                    f"**Revisions Required:**\n" + "\n".join(f"- {r}" for r in critique["suggested_revisions"])
                )
            return LLMResponse(
                text=text_out,
                model=f"{self.model} (simulation)",
                is_simulated=True,
            )

        # 3. Researcher Agent Simulation
        elif "research" in sys_str:
            text_out = (
                "Based on latest industry intelligence retrieved from search:\n\n"
                "- Enterprise Agentic AI is expanding at a 42% CAGR, with market valuation hitting $28.5B.\n"
                "- 68% of Fortune 500 organizations are migrating from static prompt-chains to autonomous multi-agent workflows.\n"
                "- Implementing self-reflection (Critic agents) and tool execution loops reduces hallucination rates by 62%.\n"
                "- Top multi-agent patterns include Supervisor Orchestration, Autonomous Goal Decomposition, and Reflection loops."
            )
            return LLMResponse(
                text=text_out,
                model=f"{self.model} (simulation)",
                is_simulated=True,
            )

        # 4. Analyst Agent Simulation
        elif "analyst" in sys_str or "computational" in sys_str or "quantitative" in sys_str:
            text_out = (
                "Quantitative & Computational Analysis Results:\n\n"
                "- Baseline 2026 Enterprise Valuation: $28.50B\n"
                "- Projected 2029 Market Size (Compound 42% CAGR): $28.5 * (1.42)^3 = $81.79B\n"
                "- Task Success Benchmark: Multi-Agent Loops achieved 89.2% accuracy vs 54.1% for Zero-Shot LLMs (+35.1% net lift)\n"
                "- Cost-Efficiency: Fast-tier Gemini 3.8 Flash yields 4.2x lower latency than monolithic legacy models."
            )
            return LLMResponse(
                text=text_out,
                model=f"{self.model} (simulation)",
                is_simulated=True,
            )

        # 5. Default / Synthesizer Simulation
        else:
            text_out = (
                "# Executive Strategic Dossier: Agentic Workflows with GenAI\n\n"
                "## 1. Executive Summary\n"
                "Modern Generative AI has transitioned from passive text generation to autonomous **Agentic Workflows**. "
                "By pairing Gemini 3.8 Flash with structured planning, autonomous tool calling, and self-reflective feedback loops, "
                "enterprise tasks achieve unprecedented accuracy (+34% over single-shot prompts) and operational autonomy.\n\n"
                "## 2. Core Architectural Pillars\n"
                "1. **Autonomous Planning**: Deconstructing ambiguous high-level goals into deterministic sub-tasks.\n"
                "2. **Tool Execution (Grounding)**: Dynamic interaction with live environments, APIs, databases, and Python interpreters.\n"
                "3. **Multi-Agent Specialization**: Distributing cognitive roles across Supervisor, Researcher, Analyst, and Critic.\n"
                "4. **Self-Reflection (Reflection Loop)**: Continuous critique against domain rubrics to refine and eliminate errors.\n\n"
                "## 3. Quantitative Projections & Benchmark Analysis\n"
                "- Current Market Base: $28.5B (2026)\n"
                "- Projected 3-Year Market (42% CAGR): $81.79B (2029)\n"
                "- Multi-Agent Task Completion Precision: 89% (vs. 54% single-turn)\n"
                "- Hallucination Reduction: 62% reduction via Critic agent verification.\n\n"
                "## 4. Strategic Recommendations\n"
                "- Adopt standard multi-agent orchestration frameworks with state persistence.\n"
                "- Implement automated reflection gates before publishing mission-critical outputs.\n"
                "- Maintain a unified tool registry with strict sandbox execution boundaries."
            )
            return LLMResponse(
                text=text_out,
                model=f"{self.model} (simulation)",
                is_simulated=True,
            )
