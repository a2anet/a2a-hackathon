"""M3 verification: the batch loop checkpoints and resumes (new tasks only),
and the 50/25/25 scoring combines pairing dirs with missing/INFRA = 0."""

from datetime import datetime

import pytest
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from tau2.data_model.message import UserMessage
from tau2.data_model.simulation import (
    AgentInfo,
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.environment.environment import EnvironmentInfo

import a2a_hack.runner as runner_module
import a2a_hack.domain as domain_module
from a2a_hack.domain import get_hack_tasks
from a2a_hack.runner import run_batch
from a2a_hack.scoring import score_pairings

from conftest import SimpleEchoAgent, free_port, start_server


class ScriptedUser:
    """Minimal half-duplex user: fixed script ending in ###STOP###."""

    def __init__(self, script: list[str]):
        self.script = script

    def get_init_state(self, message_history=None) -> dict:
        return {"i": 0}

    def generate_next_message(self, message, state) -> tuple[UserMessage, dict]:
        text = self.script[min(state["i"], len(self.script) - 1)]
        state["i"] += 1
        return UserMessage(role="user", content=text), state

    def set_seed(self, seed: int) -> None:
        pass

    def stop(self, message=None, state=None) -> None:
        pass


@pytest.fixture(scope="module")
def personal_echo():
    port = free_port()
    app = to_a2a(SimpleEchoAgent(name="personal_echo"), host="127.0.0.1", port=port)
    server = start_server(app, port)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


def test_batch_checkpoint_and_resume(tmp_path, monkeypatch, personal_echo):
    monkeypatch.setattr(
        runner_module,
        "build_user_sim",
        lambda llm, task, llm_args=None: ScriptedUser(["hi there", "ok bye ###STOP###"]),
    )
    all_tasks = get_hack_tasks()
    save_to = tmp_path / "results-a"

    def _run(tasks):
        return run_batch(
            tasks=tasks,
            personal_url=personal_echo,
            cs_url="http://127.0.0.1:1",  # echo agent never calls the CS leg
            save_to=save_to,
            user_llm="scripted",
            user_token="u",
            agent_token="a",
            concurrency=2,
            auto_resume=True,
            api_host="127.0.0.1",
            api_port=free_port(),
        )

    first = _run(all_tasks[:3])
    assert len(first.simulations) == 3
    first_ids = {s.id for s in first.simulations}

    # Resume with two extra tasks: the original three are not re-run.
    second = _run(all_tasks[:5])
    assert len(second.simulations) == 5
    assert first_ids <= {s.id for s in second.simulations}

    # Echo agents never touch tools, so the DB check fails -> reward 0,
    # but every sim terminated cleanly via the user's ###STOP###.
    for sim in second.simulations:
        assert sim.termination_reason == TerminationReason.USER_STOP
        assert sim.reward_info is not None and sim.reward_info.reward == 0.0


def test_vertex_claude_user_sim_gets_anthropic_cache_args(monkeypatch) -> None:
    captured: dict = {}

    class FakeUserSimulator:
        """Capture build_user_sim args without constructing tau2's simulator."""

        def __init__(
            self,
            llm: str,
            instructions: str,
            tools: object,
            llm_args: dict | None,
        ) -> None:
            captured["llm"] = llm
            captured["instructions"] = instructions
            captured["tools"] = tools
            captured["llm_args"] = llm_args

    class TaskStub:
        """Minimal task shape used by build_user_sim."""

        user_scenario = "complete the task"

    monkeypatch.setattr(domain_module, "UserSimulator", FakeUserSimulator)

    domain_module.build_user_sim(
        "vertex_ai/claude-sonnet-4-6",
        TaskStub(),
        {"vertex_project": "agent-project"},
    )

    assert captured["llm"] == "vertex_ai/claude-sonnet-4-6"
    assert captured["tools"] is None
    assert captured["llm_args"]["vertex_project"] == "agent-project"
    assert captured["llm_args"]["cache_control_injection_points"] == [
        {"location": "message", "role": "system"},
        {"location": "message", "index": -1},
    ]


def _minimal_info() -> Info:
    return Info(
        git_commit="test",
        num_trials=1,
        max_steps=60,
        max_errors=10,
        user_info=UserInfo(implementation="user_simulator"),
        agent_info=AgentInfo(implementation="a2a_bridge"),
        environment_info=EnvironmentInfo(domain_name="banking_hackathon", policy="p"),
    )


def _write_pairing(path, tasks, rewards: dict[str, float | None]):
    """rewards: task_id -> reward, or None for an INFRASTRUCTURE_ERROR sim."""
    now = datetime.now().isoformat()
    sims = []
    for i, (task_id, reward) in enumerate(rewards.items()):
        infra = reward is None
        sims.append(
            SimulationRun(
                id=f"{path.name}-{i}",
                task_id=task_id,
                start_time=now,
                end_time=now,
                duration=1.0,
                termination_reason=(
                    TerminationReason.INFRASTRUCTURE_ERROR
                    if infra
                    else TerminationReason.USER_STOP
                ),
                reward_info=None if infra else RewardInfo(reward=reward),
                messages=[],
                trial=0,
                seed=1,
            )
        )
    results = Results(
        info=_minimal_info(),
        tasks=[t for t in tasks if t.id in rewards],
        simulations=sims,
    )
    results.save(path / "results.json", format="dir")


def test_score_pairings_math(tmp_path):
    tasks = [t for t in get_hack_tasks() if t.id in {"task_001", "task_002"}]
    # a: both tasks; b: one INFRA (counts 0); c: task_002 missing entirely.
    _write_pairing(tmp_path / "a", tasks, {"task_001": 1.0, "task_002": 0.5})
    _write_pairing(tmp_path / "b", tasks, {"task_001": None, "task_002": 1.0})
    _write_pairing(tmp_path / "c", tasks, {"task_001": 1.0})

    scores = score_pairings(tmp_path / "a", tmp_path / "b", tmp_path / "c")
    assert scores["a"] == 0.75
    assert scores["b"] == 0.5
    assert scores["c"] == 0.5
    assert scores["final"] == 0.5 * 0.75 + 0.25 * 0.5 + 0.25 * 0.5
    assert scores["per_task"]["task_002"]["c"] == 0.0
