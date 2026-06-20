"""a2a-hack CLI: run | score | smoke."""

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.tasks import Task

from a2a_hack.domain import get_hack_task_splits, get_hack_tasks
from a2a_hack.env_api.sessions import SessionManager
from a2a_hack.runner import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
    DEFAULT_OUTER_RETRIES,
    DEFAULT_TASK_TIMEOUT_S,
    run_batch,
    run_one,
    start_env_api,
)
from a2a_hack.scoring import score_pairings

app = typer.Typer(help="A2A hackathon harness on tau2-bench banking_knowledge")
console = Console()

DEFAULT_USER_LLM = "vertex_ai/gemini-3.5-flash"
VERTEX_EXPRESS_BASE = "https://aiplatform.googleapis.com/v1/publishers/google"
DEFAULT_API_PORT = 8090

# Static per-job bearer tokens; the marking worker injects random ones, these
# defaults match the template's .env.example for the local dev loop.
DEFAULT_USER_TOKEN = os.environ.get("ENV_API_USER_TOKEN", "dev-user-token")
DEFAULT_AGENT_TOKEN = os.environ.get("ENV_API_AGENT_TOKEN", "dev-agent-token")


def _resolve_tasks(tasks_arg: str) -> list[Task]:
    """Resolve --tasks: a split name (test/train/feedback) or comma-separated ids."""
    splits = get_hack_task_splits()
    if tasks_arg in splits:
        return get_hack_tasks(tasks_arg)
    wanted = [t.strip() for t in tasks_arg.split(",") if t.strip()]
    by_id = {t.id: t for t in get_hack_tasks()}
    missing = [t for t in wanted if t not in by_id]
    if missing:
        raise typer.BadParameter(
            f"Unknown task id(s): {missing}. Valid splits: {sorted(splits)}"
        )
    return [by_id[t] for t in wanted]


def _parse_llm_args(value: Optional[str]) -> Optional[dict]:
    if not value:
        return None
    return json.loads(value)


def _resolve_user_llm(user_llm: str, llm_args: Optional[dict]) -> tuple[str, Optional[dict]]:
    """With GOOGLE_API_KEY set (and no explicit overrides), route the user sim
    through the Vertex express endpoint — LiteLLM's vertex_ai provider only
    takes ADC credentials, so a Vertex API key alone would otherwise not work."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if user_llm == DEFAULT_USER_LLM and llm_args is None and api_key:
        return "gemini/gemini-3.5-flash", {"api_base": VERTEX_EXPRESS_BASE, "api_key": api_key}
    return user_llm, llm_args


@app.command()
def run(
    personal_url: str = typer.Option(..., help="A2A URL of the team's personal agent"),
    cs_url: str = typer.Option(..., help="Real CS agent URL the gateway forwards to"),
    tasks: str = typer.Option("train", help="Split name or comma-separated task ids"),
    save_to: Path = typer.Option(..., help="Results directory (tau2 dir format)"),
    concurrency: int = typer.Option(DEFAULT_CONCURRENCY),
    user_llm: str = typer.Option(DEFAULT_USER_LLM),
    user_llm_args: Optional[str] = typer.Option(None, help="JSON dict of LLM args"),
    seed: int = typer.Option(42),
    max_steps: int = typer.Option(DEFAULT_MAX_STEPS),
    max_errors: int = typer.Option(DEFAULT_MAX_ERRORS),
    task_timeout: float = typer.Option(DEFAULT_TASK_TIMEOUT_S, help="Whole-task wall-clock seconds"),
    max_retries: int = typer.Option(DEFAULT_OUTER_RETRIES),
    auto_resume: bool = typer.Option(False, "--auto-resume"),
    api_host: str = typer.Option("0.0.0.0", help="Env API bind host"),
    api_port: int = typer.Option(DEFAULT_API_PORT, help="Env API bind port"),
    advertise_base: Optional[str] = typer.Option(
        None, help="Externally reachable env API base URL (defaults to request host)"
    ),
    user_token: str = typer.Option(DEFAULT_USER_TOKEN, envvar="ENV_API_USER_TOKEN"),
    agent_token: str = typer.Option(DEFAULT_AGENT_TOKEN, envvar="ENV_API_AGENT_TOKEN"),
):
    """Run a batch of tasks against a team's agent pair."""
    task_list = _resolve_tasks(tasks)
    user_llm, llm_args = _resolve_user_llm(user_llm, _parse_llm_args(user_llm_args))
    results = run_batch(
        tasks=task_list,
        personal_url=personal_url,
        cs_url=cs_url,
        save_to=save_to,
        user_llm=user_llm,
        user_llm_args=llm_args,
        user_token=user_token,
        agent_token=agent_token,
        concurrency=concurrency,
        seed=seed,
        max_steps=max_steps,
        max_errors=max_errors,
        task_timeout=task_timeout,
        max_retries=max_retries,
        auto_resume=auto_resume,
        api_host=api_host,
        api_port=api_port,
        advertise_base=advertise_base,
    )
    rewards = [
        s.reward_info.reward for s in results.simulations if s.reward_info is not None
    ]
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    console.print(
        f"[bold]Done:[/bold] {len(results.simulations)} sims, mean reward {mean:.3f}. "
        f"Browse with: tau2 view {save_to}"
    )
    # Non-zero exit on remaining INFRA errors so callers (the marking worker)
    # retry with --auto-resume, which requeues exactly those sims.
    from tau2.data_model.simulation import TerminationReason

    infra = [
        s.task_id
        for s in results.simulations
        if s.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
    ]
    if infra:
        console.print(f"[red]{len(infra)} sim(s) ended in INFRASTRUCTURE_ERROR: {infra}[/red]")
        raise typer.Exit(2)


@app.command()
def score(
    a: Path = typer.Option(..., help="Results dir: team personal x team CS"),
    b: Path = typer.Option(..., help="Results dir: team personal x held-out CS"),
    c: Path = typer.Option(..., help="Results dir: held-out personal x team CS"),
    out: Path = typer.Option(Path("scores.json")),
):
    """Combine three pairing runs into the 50/25/25 final score."""
    scores = score_pairings(a, b, c)
    out.write_text(json.dumps(scores, indent=2))
    console.print(
        f"a={scores['a']:.3f} b={scores['b']:.3f} c={scores['c']:.3f} "
        f"[bold]final={scores['final']:.3f}[/bold] -> {out}"
    )


@app.command()
def smoke(
    task_id: Optional[str] = typer.Option(None, help="Task id (default: first feedback task)"),
    personal_url: str = typer.Option(...),
    cs_url: str = typer.Option(...),
    user_llm: str = typer.Option(DEFAULT_USER_LLM),
    user_llm_args: Optional[str] = typer.Option(None, help="JSON dict of LLM args"),
    seed: int = typer.Option(42),
    max_steps: int = typer.Option(DEFAULT_MAX_STEPS),
    api_host: str = typer.Option("0.0.0.0"),
    api_port: int = typer.Option(DEFAULT_API_PORT),
    advertise_base: Optional[str] = typer.Option(None),
    user_token: str = typer.Option(DEFAULT_USER_TOKEN, envvar="ENV_API_USER_TOKEN"),
    agent_token: str = typer.Option(DEFAULT_AGENT_TOKEN, envvar="ENV_API_AGENT_TOKEN"),
):
    """Run one task and print both conversation legs, tool calls, and reward."""
    if task_id is None:
        task_id = get_hack_task_splits()["feedback"][0]
    by_id = {t.id: t for t in get_hack_tasks()}
    if task_id not in by_id:
        raise typer.BadParameter(f"Unknown task id: {task_id}")
    task = by_id[task_id]

    manager = SessionManager(user_token=user_token, agent_token=agent_token, cs_url=cs_url)
    server = start_env_api(manager, api_host, api_port, advertise_base)
    user_llm, llm_args = _resolve_user_llm(user_llm, _parse_llm_args(user_llm_args))
    try:
        simulation = run_one(
            task=task,
            manager=manager,
            personal_url=personal_url,
            user_llm=user_llm,
            user_llm_args=llm_args,
            seed=seed,
            max_steps=max_steps,
        )
    finally:
        server.should_exit = True

    console.rule(f"[bold]Events (task {task.id})")
    events = sorted(
        ((simulation.info or {}).get("events") or []),
        key=lambda event: event.get("sequence", 0),
    )
    for event in events:
        channel = event.get("channel")
        actor = event.get("actor")
        if event.get("type") == "message":
            console.print(
                f"[bold]{channel}/{actor}:[/bold] {(event.get('content') or '')[:500]}"
            )
        elif event.get("type") == "tool":
            tool_call = event.get("tool_call") or {}
            console.print(
                f"[yellow]{channel}/{actor} tool:[/yellow] "
                f"{tool_call.get('name')}({json.dumps(tool_call.get('arguments') or {})[:200]})"
            )

    console.rule("[bold]Tau2 evaluator tool trajectory")
    for msg in simulation.messages or []:
        if isinstance(msg, ToolMessage):
            content = (msg.content or "")[:300]
            console.print(f"  [dim]tool result ({msg.requestor}):[/dim] {content}")
        elif isinstance(msg, (UserMessage, AssistantMessage)):
            if msg.is_tool_call():
                for tc in msg.tool_calls:
                    who = (
                        "personal agent (user tools)"
                        if tc.requestor == "user"
                        else "CS agent (bank tools)"
                    )
                    console.print(
                        f"[yellow]{who}:[/yellow] "
                        f"{tc.name}({json.dumps(tc.arguments)[:200]})"
                    )

    console.rule("[bold]Checks")
    num_tool_calls = (simulation.info or {}).get("num_env_tool_calls", 0)
    if num_tool_calls == 0:
        console.print(
            "[red]No env tool calls were recorded for this contextId. "
            "Check that both agents reuse the incoming A2A contextId on env API calls.[/red]"
        )
    else:
        console.print(f"[green]{num_tool_calls} env tool call(s) recorded under the contextId.[/green]")
    if not any(event.get("channel") == "personal_cs" for event in events):
        console.print(
            "[red]No personal->CS messages were captured. Check that the personal agent "
            "calls CS_AGENT_URL and propagates the incoming contextId.[/red]"
        )
    else:
        cs_events = [event for event in events if event.get("channel") == "personal_cs"]
        console.print(f"[green]{len(cs_events)} personal-CS event(s) captured by the gateway.[/green]")

    reward = simulation.reward_info.reward if simulation.reward_info else 0.0
    console.print(
        f"[bold]Reward: {reward}[/bold] (termination: {simulation.termination_reason})"
    )


if __name__ == "__main__":
    app()
