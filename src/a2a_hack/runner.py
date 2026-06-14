"""Run-one flow and slim batch loop for hackathon simulations.

Reuses tau2's checkpoint/resume, retry, and evaluator machinery; the agent
side is the A2A bridge talking to the team's personal agent, and env tool
calls arrive out-of-band through the env API."""

import multiprocessing
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import uvicorn
from loguru import logger
from tau2.data_model.simulation import (
    AgentInfo,
    Info,
    Results,
    SimulationRun,
    UserInfo,
)
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner.checkpoint import create_checkpoint_fns, try_resume
from tau2.runner.progress import StatusMonitor, run_with_retry
from tau2.user.user_simulator import get_global_user_sim_guidelines
from tau2.utils.utils import get_commit_hash

from a2a_hack.bridge import A2ABridgeAgent
from a2a_hack.domain import DOMAIN_NAME, build_user_sim
from a2a_hack.env_api.server import create_app
from a2a_hack.env_api.sessions import SessionManager
from a2a_hack.merge import merge_trajectory

DEFAULT_MAX_STEPS = 60
DEFAULT_MAX_ERRORS = 10
# Each sim drives 3 models on one Vertex key; a high default 429s a single
# express key. Raise --concurrency if your key has more quota.
DEFAULT_CONCURRENCY = 2
DEFAULT_OUTER_RETRIES = 2
# Whole-task wall-clock; the orchestrator ends a slower sim as TIMEOUT (reward
# 0), checked at turn boundaries (so a turn already running may overrun by up
# to one per-turn budget). Paired with the bridge's per-turn timeout.
DEFAULT_TASK_TIMEOUT_S = 600.0

BRIDGE_IMPLEMENTATION = "a2a_bridge"


def start_env_api(manager: SessionManager, host: str, port: int, advertise_base: Optional[str] = None) -> uvicorn.Server:
    """Start the env API server in a background thread; returns the server
    (set ``should_exit`` to stop it)."""
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(manager, advertise_base=advertise_base),
            host=host,
            port=port,
            log_level="warning",
        )
    )
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            return server
        time.sleep(0.1)
    raise RuntimeError(f"Env API server failed to start on {host}:{port}")


def run_one(
    task: Task,
    manager: SessionManager,
    personal_url: str,
    user_llm: str,
    user_llm_args: Optional[dict] = None,
    seed: Optional[int] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_errors: int = DEFAULT_MAX_ERRORS,
    task_timeout: float = DEFAULT_TASK_TIMEOUT_S,
) -> SimulationRun:
    """Run a single simulation: session -> orchestrator -> merge -> evaluate.

    A fresh uuid is both the env session id and the A2A contextId; the merged
    trajectory (conversation + out-of-band tool calls) is the evaluation
    input, so this can't reuse tau2's run_simulation (which evaluates before
    the merge could happen).
    """
    sid = uuid.uuid4().hex
    session = manager.create_session(sid, task)
    bridge = A2ABridgeAgent(personal_url=personal_url, context_id=sid)
    user_sim = build_user_sim(user_llm, task, user_llm_args)
    orchestrator = Orchestrator(
        domain=DOMAIN_NAME,
        agent=bridge,
        user=user_sim,
        environment=session.env,
        task=task,
        max_steps=max_steps,
        max_errors=max_errors,
        timeout=task_timeout,
        seed=seed,
        simulation_id=sid,
    )
    try:
        simulation = orchestrator.run()
    finally:
        manager.close(sid)

    simulation.messages = merge_trajectory(simulation.messages, session.records)
    simulation.reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=DOMAIN_NAME,
    )
    simulation.info = {
        "context_id": sid,
        "leg2": [r.model_dump(exclude={"raw"}) for r in session.chat_records],
        "num_env_tool_calls": len(session.records),
    }
    return simulation


def build_info(
    user_llm: str,
    user_llm_args: Optional[dict],
    max_steps: int,
    max_errors: int,
    seed: int,
) -> Info:
    """Run metadata for Results; also the config-change fingerprint try_resume
    compares on resume."""
    from a2a_hack.domain import get_hack_environment

    return Info(
        git_commit=get_commit_hash(),
        num_trials=1,
        max_steps=max_steps,
        max_errors=max_errors,
        user_info=UserInfo(
            implementation="user_simulator",
            llm=user_llm,
            llm_args=user_llm_args or {},
            global_simulation_guidelines=get_global_user_sim_guidelines(),
        ),
        agent_info=AgentInfo(implementation=BRIDGE_IMPLEMENTATION),
        environment_info=get_hack_environment().get_info(),
        seed=seed,
    )


def run_batch(
    tasks: list[Task],
    personal_url: str,
    cs_url: str,
    save_to: Path,
    user_llm: str,
    user_token: str,
    agent_token: str,
    user_llm_args: Optional[dict] = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    seed: int = 42,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_errors: int = DEFAULT_MAX_ERRORS,
    task_timeout: float = DEFAULT_TASK_TIMEOUT_S,
    max_retries: int = DEFAULT_OUTER_RETRIES,
    auto_resume: bool = False,
    api_host: str = "0.0.0.0",
    api_port: int = 8090,
    advertise_base: Optional[str] = None,
) -> Results:
    """Run a batch of tasks against a team's agents with resume + retries.

    Results are written incrementally in tau2's dir format (browsable with
    ``tau2 view``); failed-after-retries sims are stored as
    INFRASTRUCTURE_ERROR and re-queued on the next resume.
    """
    manager = SessionManager(user_token=user_token, agent_token=agent_token, cs_url=cs_url)
    server = start_env_api(manager, api_host, api_port, advertise_base)
    try:
        # Single trial; seed derivation mirrors tau2's batch for stable
        # (trial, task_id, seed) resume keys.
        random.seed(seed)
        trial_seed = random.randint(0, 1000000)

        info = build_info(user_llm, user_llm_args, max_steps, max_errors, seed)
        results = Results(info=info, tasks=tasks, simulations=[])
        save_path = Path(save_to) / "results.json"
        results, done_runs, tasks = try_resume(
            save_path=save_path,
            simulation_results=results,
            tasks=tasks,
            num_trials=1,
            auto_resume=auto_resume,
            results_format="dir",
        )
        save_fn, _replace_fn = create_checkpoint_fns(save_path, multiprocessing.Lock())

        todo = [t for t in tasks if (0, t.id, trial_seed) not in done_runs]
        monitor = StatusMonitor(len(tasks), initial_completed=len(done_runs))
        monitor.set_results(results)
        monitor.start()

        def _run_tracked(task: Task) -> SimulationRun:
            task_key = f"{task.id}.0"
            monitor.task_started(task_key, 0)
            try:
                simulation = run_with_retry(
                    lambda: run_one(
                        task=task,
                        manager=manager,
                        personal_url=personal_url,
                        user_llm=user_llm,
                        user_llm_args=user_llm_args,
                        seed=trial_seed,
                        max_steps=max_steps,
                        max_errors=max_errors,
                        task_timeout=task_timeout,
                    ),
                    task=task,
                    trial=0,
                    seed=trial_seed,
                    max_retries=max_retries,
                    retry_delay=2.0,
                    save_fn=save_fn,
                    on_retry=lambda: monitor.task_restarted(task_key),
                )
                results.simulations.append(simulation)
                return simulation
            finally:
                monitor.task_finished(task_key)

        try:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                list(executor.map(_run_tracked, todo))
        finally:
            monitor.stop()

        final = Results.load(save_path)
        logger.info(
            f"Batch complete: {len(final.simulations)}/{len(tasks)} sims saved to {save_to}"
        )
        return final
    finally:
        server.should_exit = True
