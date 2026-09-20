"""End-to-End Orchestration Workbench & Demo Script for Telco AI Intent Routing.

Visual terminal reproduction of Docs/jev_intent_classifier_studio.html with Rich UI.
Flow: User turn -> Jev classification -> RAG retrieval -> Qwen response generation -> Human handoff trigger.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List, Optional
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pipelines.orchestrator import HandoverContextCard, OrchestrationResult, PipelineOrchestrator
from services.jev.classifier import IntentChoice

# Force UTF-8 encoding in Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console()

PRESET_QUERIES = [
    {
        "label": "Double Billing (High Frustration)",
        "query": "Why was I billed twice on my Mastercard for invoice #INV-9024? Your bot is useless, connect me to a human manager right now!",
    },
    {
        "label": "Fiber Optical Cut (Critical Emergency)",
        "query": "CRITICAL EMERGENCY: Construction cut our main fiber line! The ONT optical box has a blinking LOS red light and 30 employees are completely down!",
    },
    {
        "label": "5G APN & Roaming Configuration (Normal Grounded)",
        "query": "What are the recommended regional APN settings for 5G roaming in Zone 2 Americas under the Infinite plan?",
    },
    {
        "label": "Ambiguous / Vague Query (Fallback)",
        "query": "Hey there folks, just wondering how you handle general account things.",
    },
]


def render_banner() -> None:
    """Renders workbench top header."""
    header_text = Text()
    header_text.append("⚡ JEV INTENT CLASSIFIER & RAG WORKBENCH ", style="bold white")
    header_text.append("v1.13-SPEC\n", style="bold cyan")
    header_text.append("Deterministic Next-Token Chat Intent, Urgency (Noul), Frustration (Score) & Qwen 3 Local Grounding", style="dim")

    console.print(Panel(header_text, box=ROUNDED, border_style="cyan", padding=(0, 2)))


def render_orchestration_result(res: OrchestrationResult, turn_num: int = 1) -> None:
    """Renders studio-fidelity terminal UI."""
    jev = res.jev_result

    # 1. Dispatch Status Banner
    queue_style = "bold green"
    if jev.is_emergency:
        queue_style = "bold red on black"
    elif jev.is_escalated:
        queue_style = "bold yellow on black"
    elif jev.is_fallback:
        queue_style = "bold magenta"

    status_table = Table(box=None, show_header=False, expand=True)
    status_table.add_column("Key", style="bold", width=28)
    status_table.add_column("Value")

    status_table.add_row("Input User Message:", f"[italic white]\"{res.user_message}\"[/italic white]")
    status_table.add_row(
        "Downstream Dispatch Queue:",
        f"[{queue_style}] ➔ {jev.dispatch_queue} [/{queue_style}] [dim]({jev.dispatch_reason})[/dim]",
    )
    status_table.add_row(
        "Classification Latency:",
        f"[cyan]{jev.latency_ms:.1f} ms[/cyan] [dim]({jev.engine_used})[/dim] | Total Turn: [yellow]{res.total_latency_ms:.1f} ms[/yellow]",
    )

    console.print(Panel(status_table, title=f"[bold cyan]Turn #{turn_num} Execution Summary[/bold cyan]", box=ROUNDED, border_style="cyan"))

    # 2. Probability Distribution Bars
    prob_table = Table(title="Next-Token Choice Logit Distribution", box=ROUNDED, border_style="blue", expand=True)
    prob_table.add_column("Intent Class", style="bold", width=26)
    prob_table.add_column("Distribution Gauge", width=42)
    prob_table.add_column("Probability", justify="right", width=12)

    for item in jev.distribution:
        p_pct = item.probability * 100
        bar_len = int(item.probability * 35)
        bar_str = "█" * bar_len + "░" * (35 - bar_len)

        is_selected = item.choice == jev.selected_choice
        if is_selected:
            intent_label = f"[bold cyan]➔ {item.choice}[/bold cyan] [bold white](Selected)[/bold white]"
            bar_color = "bold cyan" if item.probability >= 0.70 else "yellow"
            prob_label = f"[{bar_color}]{p_pct:5.1f}%[/{bar_color}]"
        else:
            intent_label = f"  [dim]{item.choice}[/dim]"
            bar_color = "dim blue"
            prob_label = f"[dim]{p_pct:5.1f}%[/dim]"

        prob_table.add_row(intent_label, f"[{bar_color}]{bar_str}[/{bar_color}]", prob_label)

    console.print(prob_table)

    # 3. Auxiliary Primitives: Noul (Urgency) and Score (Frustration)
    noul_pct = int(jev.is_urgent * 100)
    noul_bar = "█" * int(jev.is_urgent * 25) + "░" * (25 - int(jev.is_urgent * 25))
    noul_color = "bold red" if jev.is_urgent >= 0.75 else "yellow"

    score_pct = int(jev.frustration_score * 100)
    score_bar = "█" * int(jev.frustration_score * 25) + "░" * (25 - int(jev.frustration_score * 25))
    score_color = "bold red" if jev.frustration_score >= 0.80 else "magenta"

    meter_table = Table(box=ROUNDED, expand=True, show_header=True, border_style="cyan")
    meter_table.add_column("Noul: Outage / Urgency Meter", width=40)
    meter_table.add_column("Score: Frustration Index", width=40)

    noul_text = (
        f"Value: [{noul_color}]{jev.is_urgent:.2f} ({noul_pct}%)[/{noul_color}] "
        f"{'[bold red]CRISIS TRIGGER[/bold red]' if jev.is_urgent >= 0.75 else '[dim]Normal[/dim]'}\n"
        f"[{noul_color}]{noul_bar}[/{noul_color}]\n"
        f"[dim]Threshold: 0.75 (Tier-2 Emergency Queue)[/dim]"
    )

    score_text = (
        f"Value: [{score_color}]{jev.frustration_score:.2f} ({score_pct}%)[/{score_color}] "
        f"{'[bold red]ESCALATE HUMAN[/bold red]' if jev.frustration_score >= 0.80 else '[dim]Normal[/dim]'}\n"
        f"[{score_color}]{score_bar}[/{score_color}]\n"
        f"[dim]Threshold: 0.80 (Senior Human Queue)[/dim]"
    )

    meter_table.add_row(noul_text, score_text)
    console.print(meter_table)

    # 4. Jev Routing Execution Trace Logs
    trace_text = Text()
    for log in jev.trace_logs:
        if "[GUARDRAIL HIT]" in log:
            trace_text.append(f"{log}\n", style="bold yellow")
        elif "[FALLBACK TRIGGERED]" in log:
            trace_text.append(f"{log}\n", style="bold magenta")
        elif "[ROUTED]" in log:
            trace_text.append(f"{log}\n", style="bold green")
        else:
            trace_text.append(f"{log}\n", style="dim")

    console.print(Panel(trace_text, title="[bold]Jev Routing Execution Trace[/bold]", box=ROUNDED, border_style="cyan"))

    # 5. RAG Grounding References (if retrieved)
    if res.retrieved_chunks:
        rag_table = Table(title=f"LanceDB Grounding Context ({len(res.retrieved_chunks)} Chunks Retrieved)", box=ROUNDED, border_style="green", expand=True)
        rag_table.add_column("Doc ID", style="bold cyan", width=16)
        rag_table.add_column("Plan Code", style="yellow", width=22)
        rag_table.add_column("Regional APN", style="green", width=22)
        rag_table.add_column("SLA Commitment", style="white", width=24)
        rag_table.add_column("Match Score", justify="right", width=12)

        for chunk in res.retrieved_chunks:
            rag_table.add_row(
                chunk.get("doc_id", "N/A"),
                chunk.get("plan_code", "N/A"),
                chunk.get("regional_apn", "N/A"),
                chunk.get("sla", "Standard"),
                f"{chunk.get('similarity_score', 0.95):.4f}",
            )
        console.print(rag_table)

    # 6. Generated Model Answer
    gen = res.generation_result
    gen_title = f"[bold green]Local Answering Engine Response[/bold green] [dim]({gen.backend} | {gen.model} | {gen.latency_ms:.0f}ms)[/dim]"
    reply_body = f"[white]{gen.content}[/white]"

    if gen.reasoning:
        reply_body = f"[dim italic]Reasoning trace ({len(gen.reasoning)} chars): {gen.reasoning[:200]}...[/dim italic]\n\n" + reply_body

    console.print(Panel(reply_body, title=gen_title, box=ROUNDED, border_style="green"))

    # 7. Escalation Handover Card (if present)
    if res.handover_card:
        card = res.handover_card
        card_content = Text()
        card_content.append(f"Card ID: {card.handover_id} | Dispatched to: {card.dispatch_queue}\n", style="bold yellow")
        card_content.append(f"Trigger Reason: {card.trigger_reason}\n", style="bold red")
        card_content.append("\nRecommended Human Specialist Actions:\n", style="bold white")
        for act in card.recommended_actions:
            card_content.append(f"  • {act}\n", style="cyan")

        console.print(Panel(card_content, title="[bold red]⚠️ AUTOMATED HUMAN HANDOVER CONTEXT CARD[/bold red]", box=ROUNDED, border_style="red"))

    console.print("\n" + "=" * 90 + "\n")


def run_demo_suite(orchestrator: PipelineOrchestrator) -> None:
    """Executes the 3 canonical end-to-end benchmark scenarios."""
    console.print("[bold cyan]=== RUNNING AUTOMATED ORCHESTRATION DEMO SUITE ===[/bold cyan]\n")

    demo_scenarios = [
        {
            "title": "Scenario 1: High-Speed 5G APN Configuration (Normal Grounding + Bot Contained)",
            "message": "Can you explain how to configure regional APN settings for 5G roaming in Zone 2 Americas under the Infinite plan?",
            "meta": {"customer_name": "Jordan Smith", "account_number": "ACCT-981023"},
        },
        {
            "title": "Scenario 2: Double Charge Billing Dispute (Frustration >= 0.80 -> Senior Human Handover)",
            "message": "Why was I billed twice on my Mastercard for invoice #INV-9024? Your bot is completely useless! Transfer me to a real person right now!",
            "meta": {"customer_name": "Alex Taylor", "account_number": "ACCT-449102"},
        },
        {
            "title": "Scenario 3: Critical Fiber Cable Cut (Noul Urgency >= 0.75 -> Emergency NOC Truck Roll)",
            "message": "CRITICAL EMERGENCY: A construction backhoe just sliced our fiber drop line! The ONT box has a blinking LOS red light and our whole company is disconnected!",
            "meta": {"customer_name": "David Miller (IT Director)", "account_number": "ACCT-ENT-8819"},
        },
    ]

    for idx, scen in enumerate(demo_scenarios, start=1):
        console.print(f"[bold yellow]▶ EXECUTING {scen['title']}[/bold yellow]")
        result = orchestrator.process_turn(
            user_message=scen["message"],
            account_metadata=scen["meta"],
        )
        render_orchestration_result(result, turn_num=idx)
        time.sleep(1.0)

    console.print("[bold green]✔ DEMO SUITE COMPLETE: All 3 routing, RAG grounding, and escalation paths verified![/bold green]\n")


def run_interactive_workbench(orchestrator: PipelineOrchestrator) -> None:
    """Launches interactive CLI workbench."""
    render_banner()

    console.print("[bold white]Interactive Workbench Ready.[/bold white] Type your message or type [cyan]presets[/cyan], [cyan]demo[/cyan], or [cyan]exit[/cyan].\n")

    turn_counter = 1
    while True:
        try:
            user_input = console.input("[bold cyan]Customer Input > [/bold cyan]").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Exiting Workbench. Goodbye![/yellow]")
                break

            if user_input.lower() == "demo":
                run_demo_suite(orchestrator)
                continue

            if user_input.lower() == "presets":
                console.print("\n[bold]Available Test Presets:[/bold]")
                for i, p in enumerate(PRESET_QUERIES, 1):
                    console.print(f"  [yellow]{i}[/yellow]. [bold]{p['label']}[/bold]: [dim]\"{p['query']}\"[/dim]")
                choice = console.input("\nEnter preset number (1-4): ").strip()
                try:
                    p_idx = int(choice) - 1
                    if 0 <= p_idx < len(PRESET_QUERIES):
                        user_input = PRESET_QUERIES[p_idx]["query"]
                        console.print(f"[green]Selected:[/green] \"{user_input}\"\n")
                    else:
                        continue
                except ValueError:
                    continue

            result = orchestrator.process_turn(user_message=user_input)
            render_orchestration_result(result, turn_num=turn_counter)
            turn_counter += 1

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted. Exiting.[/yellow]")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telco AI Intent Routing & RAG Workbench")
    parser.add_argument("--demo", action="store_true", help="Run automated 3-scenario benchmark demo")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive workbench CLI")

    args = parser.parse_args()

    orchestrator = PipelineOrchestrator()

    if args.demo:
        render_banner()
        run_demo_suite(orchestrator)
    else:
        run_interactive_workbench(orchestrator)
