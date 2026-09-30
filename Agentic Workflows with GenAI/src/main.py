"""Main CLI entrypoint for the GenAI Agentic Workflow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path for direct execution
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown

from src.framework.llm_client import LLMClient
from src.framework.workflow import WorkflowState
from src.agents.supervisor import SupervisorAgent

console = Console()


def print_banner() -> None:
    """Render startup banner."""
    title = Text("Agentic Workflow with Google GenAI", style="bold cyan")
    subtitle = Text("Planning | Autonomous Tool Use | Multi-Agent Collaboration | Reflection", style="dim white")
    console.print(Panel(Text.assemble(title, "\n", subtitle), border_style="cyan"))


def format_event(event_type: str, data: Any) -> None:
    """Format and print real-time workflow events."""
    if event_type == "workflow_start":
        console.print(f"\n[bold green]>>> Workflow Initiated:[/bold green] {data['goal']}\n")

    elif event_type == "phase_change":
        console.print(f"[bold magenta]=== {data['phase']} ===[/bold magenta]")

    elif event_type == "plan_generated":
        table = Table(title="Generated Execution Plan", border_style="blue", show_header=True)
        table.add_column("Step", style="cyan", width=6)
        table.add_column("Title", style="bold white")
        table.add_column("Agent", style="yellow")
        table.add_column("Tool", style="green")
        table.add_column("Objective", style="dim")

        for task in data["plan"]:
            table.add_row(
                str(task["step"]),
                task["title"],
                task["agent"],
                task["tool"],
                task["description"],
            )
        console.print(table)
        console.print()

    elif event_type == "task_start":
        console.print(f"  [cyan]> Running Step {data['step']}:[/cyan] [bold]{data['title']}[/bold] ([yellow]{data['agent']}[/yellow])")

    elif event_type == "task_finish":
        resp = data["response"]
        if resp.tool_calls_made:
            for tc in resp.tool_calls_made:
                console.print(f"    [dim green]-> Tool Execution:[/dim green] [bold]{tc['tool']}[/bold] -> args: {tc['args']}")
        console.print(f"    [dim white][v] Step {data['step']} completed.[/dim white]\n")

    elif event_type == "reflection_start":
        console.print(f"  [yellow]* Critic auditing draft output (Attempt #{data['iteration']})...[/yellow]")

    elif event_type == "reflection_finish":
        eval_res = data["evaluation"]
        status_color = "green" if eval_res.passed else "red"
        status_label = "PASSED" if eval_res.passed else "NEEDS REVISION"
        
        table = Table(title=f"Critic Reflection Audit (Score: {eval_res.score}/100 - {status_label})", border_style=status_color)
        table.add_column("Criterion", style="white")
        table.add_column("Score", style="bold cyan")
        
        for crit, score in eval_res.criteria_evaluated.items():
            table.add_row(crit.replace("_", " ").title(), f"{score}/100")
            
        console.print(table)
        console.print(f"    [bold {status_color}]Feedback:[/bold {status_color}] {eval_res.feedback}")
        if eval_res.suggested_revisions:
            console.print("    [bold yellow]Required Revisions:[/bold yellow]")
            for r in eval_res.suggested_revisions:
                console.print(f"      - {r}")
        console.print()


def main() -> None:
    """Run CLI application."""
    parser = argparse.ArgumentParser(description="GenAI Agentic Workflow System")
    parser.add_argument(
        "--goal",
        type=str,
        default="Conduct an enterprise market analysis and technical roadmap for Agentic AI workflows in 2026",
        help="High-level goal or task to execute",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        help="Gemini model to use (default: gemini-3.8-flash)",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Force simulation mode even if an API key is present",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="agentic_workflow_report.md",
        help="Path to save the final report markdown file",
    )

    args = parser.parse_args()

    print_banner()

    # Initialize LLM Client
    client = LLMClient(
        model=args.model,
        force_simulation=args.simulation or None,
    )

    mode_text = "[green]Live Gemini API[/green]" if client.is_live else "[yellow]Deterministic Simulation Engine (No API Key Required)[/yellow]"
    console.print(f"[bold]Active Engine:[/bold] {mode_text} (Model: [bold cyan]{args.model}[/bold cyan])\n")

    supervisor = SupervisorAgent(llm_client=client)

    state: WorkflowState = supervisor.execute_workflow(
        goal=args.goal,
        on_event=format_event,
    )

    # Display final deliverable
    console.print(Panel(Markdown(state.final_output or "No output generated."), title="[bold green]Final Executive Deliverable[/bold green]", border_style="green"))

    # Save deliverable
    if state.final_output and args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(state.final_output)
            console.print(f"\n[bold green]✓ Deliverable exported to:[/bold green] {args.output}\n")
        except Exception as e:
            console.print(f"[red]Failed to save report to '{args.output}': {e}[/red]")


if __name__ == "__main__":
    main()
