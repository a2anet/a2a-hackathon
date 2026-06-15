"""Registers the "banking_hackathon" domain: tau2 banking_knowledge with the
hackathon task splits and the A2A_HACK_TASKS_DIR override (the marking VM points
this at the held-out task set at final-marking time)."""

import json
import os
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.data_model import TransactionalDB
from tau2.domains.banking_knowledge.environment import get_environment
from tau2.domains.banking_knowledge.utils import KNOWLEDGE_DB_PATH
from tau2.environment.environment import Environment
from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator

DOMAIN_NAME = "banking_hackathon"

SPLITS_PATH = Path(__file__).parent / "data" / "banking_hackathon_splits.json"
# The harness ships its own task copies under its own task numbering.
HACK_TASKS_DIR = Path(__file__).parent / "data" / "tasks"

# Appended to every task's user scenario: the simulated user talks to a
# personal assistant instead of operating banking tools directly.
USER_SIM_ADDENDUM = """

## Important: you are talking to YOUR OWN personal assistant

You are NOT talking to the bank directly. You are talking to your own
personal AI assistant, which can contact the bank's customer service on
your behalf and can also perform account actions for you (e.g. submitting
applications or referrals) when you ask it to.

- You cannot operate any tools or websites yourself. When your instructions
  say to take an action (e.g. submit a referral, apply for a card), ask your
  assistant to do it for you and confirm any details it needs.
- Talk to your assistant naturally, the same way you would talk to the bank,
  and share personal details only when asked.
- When your goals are met (or you decide to give up), end the conversation
  as usual.
""".rstrip()


def get_hack_environment(solo_mode: bool = False, **env_kwargs) -> Environment:
    """Environment constructor for the banking_hackathon domain.

    Same constructor serves the live session and both evaluator envs, so
    recording via env.get_response guarantees byte-identical replay.
    """
    env = get_environment(
        db=TransactionalDB.load(str(KNOWLEDGE_DB_PATH)),
        retrieval_variant="no_knowledge",
        solo_mode=solo_mode,
        **env_kwargs,
    )
    env.domain_name = DOMAIN_NAME
    return env


def get_hack_task_splits() -> dict[str, list[str]]:
    """Load the hackathon task splits (test/train/feedback)."""
    with open(SPLITS_PATH) as fp:
        return json.load(fp)


def get_hack_tasks(task_split_name: Optional[str] = None) -> list[Task]:
    """Load hackathon tasks, honoring the A2A_HACK_TASKS_DIR override.

    Args:
        task_split_name: Optional split name from the splits file
            ("test", "train", "feedback"). None returns all tasks.

    Returns:
        The tasks, filtered to the split if one is given.
    """
    tasks_dir = Path(os.environ.get("A2A_HACK_TASKS_DIR", str(HACK_TASKS_DIR)))
    tasks = []
    for task_file in sorted(tasks_dir.glob("task_*.json")):
        with open(task_file) as fp:
            tasks.append(Task.model_validate(json.load(fp)))
    if task_split_name is None:
        return tasks
    splits = get_hack_task_splits()
    if task_split_name not in splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits: {list(splits)}"
        )
    split_ids = set(splits[task_split_name])
    return [task for task in tasks if task.id in split_ids]


def _is_anthropic_llm(llm: str) -> bool:
    """LiteLLM model strings that route to the Anthropic provider."""
    return llm.startswith("anthropic/") or llm.startswith("claude-")


def build_user_sim(llm: str, task: Task, llm_args: Optional[dict] = None) -> UserSimulator:
    """Build the user simulator for a hackathon task.

    tools=None selects tau2's no-tools simulation guidelines; the addendum
    re-frames the scenario as acting through a personal assistant.

    For Anthropic user sims (the marking pipeline; default here is gemini),
    inject Anthropic prompt-caching breakpoints via LiteLLM's
    cache_control_injection_points hook: one anchor on the (stable) system
    prompt, one rolling on the last message so each turn's growing
    [system + tools + history] prefix is served from cache on the next turn
    (5-min ephemeral, ~90% cheaper reads).
    """
    final_args = dict(llm_args or {})
    if _is_anthropic_llm(llm) and "cache_control_injection_points" not in final_args:
        final_args["cache_control_injection_points"] = [
            {"location": "message", "role": "system"},
            {"location": "message", "index": -1},
        ]
    return UserSimulator(
        llm=llm,
        instructions=str(task.user_scenario) + USER_SIM_ADDENDUM,
        tools=None,
        llm_args=final_args or None,
    )


_registered = False


def register() -> None:
    """Register the banking_hackathon domain and tasks with the tau2 registry.

    Idempotent: safe to call from multiple entry points (CLI, tests).
    """
    global _registered
    if _registered:
        return
    registry.register_domain(get_hack_environment, DOMAIN_NAME)
    registry.register_tasks(get_hack_tasks, DOMAIN_NAME, get_task_splits=get_hack_task_splits)
    _registered = True
