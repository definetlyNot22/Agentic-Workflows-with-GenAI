# Agentic Workflows with Google GenAI

A modular, extensible, and production-ready **Agentic Workflow Framework** powered by the **Google GenAI SDK** (`google-genai`) and **Gemini 3.8 Flash**.

This framework demonstrates the four foundational agentic design patterns:
1. **Planning & Goal Decomposition**: Breaking ambiguous objectives into structured sub-tasks.
2. **Autonomous Tool Use (Grounding)**: Dynamic invocation of external search, calculators, and sandboxed Python code.
3. **Multi-Agent Collaboration**: Specialized roles (Supervisor, Researcher, Analyst, Critic) cooperating towards a shared goal.
4. **Self-Reflection & Quality Gates**: The Critic agent audits outputs against structured rubrics, enforcing iterative refinement loops before final approval.

---

## 🏛️ System Architecture

```
                             ┌─────────────────────────────┐
                             │       User Objective        │
                             └──────────────┬──────────────┘
                                            │
                                            ▼
                             ┌─────────────────────────────┐
                             │       Supervisor Agent      │
                             │   (Orchestrator & Router)   │
                             └──────────────┬──────────────┘
                                            │
                   ┌────────────────────────┼────────────────────────┐
                   ▼                        ▼                        ▼
        ┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
        │    Planner Agent     │ │   Researcher Agent   │ │    Analyst Agent     │
        │ (Task Decomposition) │ │  (Web & Tool Ground) │ │ (Code & Math Exec)   │
        └──────────┬───────────┘ └──────────┬───────────┘ └──────────┬───────────┘
                   │                        │                        │
                   └────────────────────────┼────────────────────────┘
                                            │ Intermediate Results & Context
                                            ▼
                             ┌─────────────────────────────┐
                             │         Critic Agent        │
                             │ (Reflection & Quality Gate) │
                             └──────────────┬──────────────┘
                                            │
                             ┌──────────────┴──────────────┐
                   [Needs Revision]                [Passed Quality Gate]
                             │                                     │
                             ▼                                     ▼
                   Iterative Refinement               Final Synthesized Deliverable
```

---

## 📦 Project Structure

```
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies (google-genai, rich, pydantic, pytest)
├── pytest.ini                # Pytest configuration
├── README.md                 # Project documentation
├── agentic_workflow_report.md# Sample generated executive deliverable
├── src/
│   ├── __init__.py
│   ├── main.py               # Interactive CLI runner with Rich visualization
│   ├── framework/            # Core agentic framework abstractions
│   │   ├── __init__.py
│   │   ├── agent.py          # Base Agent class with tool-loop execution
│   │   ├── llm_client.py     # Google GenAI wrapper with dual Live/Simulation engine
│   │   ├── tools.py          # Tool registry & standard tools (search, calc, python, writer)
│   │   └── workflow.py       # Shared state, memory, and orchestration hooks
│   └── agents/               # Specialized multi-agent implementations
│       ├── __init__.py
│       ├── planner.py        # Goal decomposition into executable sub-tasks
│       ├── researcher.py     # External information discovery agent
│       ├── analyst.py        # Mathematical & computational code analyst
│       ├── critic.py         # Self-reflection & rubric evaluation agent
│       └── supervisor.py     # Master coordinator and executive synthesizer
└── tests/
    └── test_workflow.py      # Unit & integration test suite (100% passing)
```

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10+
- (Optional) Gemini API Key from [Google AI Studio](https://aistudio.google.com/)

### 2. Environment Setup
```powershell
# Clone/navigate to directory
cd "Agentic Workflows with GenAI"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key
Copy the template and configure your Gemini API key:
```powershell
cp .env.example .env
```
Inside `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```

> **Dual-Mode Engine**: If no API key is provided or `SIMULATION_MODE=true`, the framework runs on a built-in deterministic simulation engine so you can test, inspect, and benchmark workflows offline!

---

## 💻 Running the Workflow

### Option A: Interactive Web UI Dashboard (Recommended)
Launch the local web server:
```powershell
python src/server.py
```
Then open your browser to **[http://127.0.0.1:8000](http://127.0.0.1:8000)** to experience the real-time visual dashboard featuring:
- Live multi-stage pipeline ribbon (Planning -> Execution -> Reflection -> Deliverable)
- Real-time Server-Sent Events (SSE) telemetry feed
- Critic reflection rubric scorecard with animated gauges
- Markdown preview of executive deliverable with 1-click clipboard copy

---

### Option B: Interactive CLI Runner
```powershell
# Run with a custom objective
python src/main.py --goal "Analyze market opportunities for Autonomous AI Agents in Healthcare"

# Run in offline simulation mode
python src/main.py --simulation

# Specify custom output path
python src/main.py --output "my_strategy_dossier.md"
```

---

### Running Automated Tests
```powershell
pytest -v tests/
```

---

## 🧩 Programmatic Usage

You can also import and run the workflow directly in your own Python applications:

```python
from src.framework.llm_client import LLMClient
from src.agents.supervisor import SupervisorAgent

# Initialize client (uses GEMINI_API_KEY from environment)
client = LLMClient(model="gemini-3.8-flash")

# Initialize master supervisor
supervisor = SupervisorAgent(llm_client=client, max_reflection_loops=2)

# Execute workflow
state = supervisor.execute_workflow(
    goal="Design an agentic pipeline for automated compliance auditing"
)

# Access final deliverable and execution history
print(state.final_output)
print("Total sub-tasks executed:", len(state.plan))
print("Audit reflections recorded:", len(state.reflections))
```

---

## 🛠️ Adding Custom Tools

To add custom tools, define a Python function with docstrings and type annotations, then wrap it in a `Tool`:

```python
from src.framework.tools import Tool

def get_stock_price(ticker: str) -> str:
    """Retrieve the current market price for a given stock ticker.
    
    Args:
        ticker: The stock ticker symbol (e.g. 'GOOGL').
    """
    # Custom API / database lookup here
    return f"{ticker}: $182.40 USD"

stock_tool = Tool(
    name="get_stock_price",
    description="Fetch real-time stock price quotes",
    func=get_stock_price,
)

# Register with agent
researcher.tools.append(stock_tool)
```

---

## 🔬 Key Agentic Patterns Implemented

| Pattern | Module | Purpose |
|---|---|---|
| **Planning** | [`src/agents/planner.py`](file:///d:/projects/Agentic%20Workflows%20with%20GenAI/src/agents/planner.py) | Deconstructs ambiguous goals into a DAG of verifiable sub-tasks with assigned agents and tools. |
| **Tool Execution** | [`src/framework/agent.py`](file:///d:/projects/Agentic%20Workflows%20with%20GenAI/src/framework/agent.py) | Autonomous tool execution loop that invokes tools and returns observation steps back to the LLM. |
| **Multi-Agent Router** | [`src/agents/supervisor.py`](file:///d:/projects/Agentic%20Workflows%20with%20GenAI/src/agents/supervisor.py) | Routes tasks to specialized agents (Researcher, Analyst, Critic) and aggregates shared context. |
| **Reflection / Critic** | [`src/agents/critic.py`](file:///d:/projects/Agentic%20Workflows%20with%20GenAI/src/agents/critic.py) | Audits drafts against Factual Grounding, Analytical Depth, and Actionability, triggering refinement cycles. |
