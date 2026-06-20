"""M0 verification: golden actions through the HTTP env API -> evaluator
trajectory -> reward must reproduce the exact task reward, with auth and
session lifecycle enforced."""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

from a2a_hack.domain import DOMAIN_NAME, get_hack_tasks
from a2a_hack.env_api.server import create_app
from a2a_hack.env_api.sessions import SessionManager
from a2a_hack.merge import build_evaluation_trajectory

USER_TOKEN = "user-secret"
AGENT_TOKEN = "agent-secret"

# The canned dialogue below matches the referral task; select it by golden
# action content so split regeneration (renumbering) can't repoint the test.
REFERRAL_ACTIONS = {"log_verification", "get_referrals_by_user", "submit_referral"}


@pytest.fixture()
def task():
    for t in get_hack_tasks():
        names = {a.name for a in (t.evaluation_criteria.actions or [])}
        if names == REFERRAL_ACTIONS and not t.evaluation_criteria.nl_assertions:
            return t
    pytest.fail("Referral task not found in the shipped task set")


def _init_env(session, task) -> None:
    """Mirror the orchestrator: apply the task's initial state to the live
    env before any tool calls (the evaluator's replay env gets it too)."""
    initial = task.initial_state
    session.env.set_state(
        initialization_data=initial.initialization_data if initial else None,
        initialization_actions=initial.initialization_actions if initial else None,
        message_history=[],
    )


@pytest.fixture()
def manager():
    return SessionManager(user_token=USER_TOKEN, agent_token=AGENT_TOKEN)


@pytest.fixture()
def client(manager):
    with TestClient(create_app(manager)) as client:
        yield client


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _simulation(task, merged) -> SimulationRun:
    now = datetime.now().isoformat()
    return SimulationRun(
        id="sim-m0",
        task_id=task.id,
        start_time=now,
        end_time=now,
        duration=1.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=merged,
    )


def _golden_action_calls(task):
    """(scope, name, arguments) per golden action, mapping requestor to scope."""
    calls = []
    for action in task.evaluation_criteria.actions:
        scope = "user" if action.requestor == "user" else "agent"
        calls.append((scope, action.name, action.arguments))
    return calls


def test_golden_replay_reward_one(client, manager, task):
    session = manager.create_session("ctx-m0", task)
    _init_env(session, task)

    for scope, name, arguments in _golden_action_calls(task):
        token = USER_TOKEN if scope == "user" else AGENT_TOKEN
        resp = client.post(
            f"/sessions/{session.id}/tools/{name}",
            json={"arguments": arguments},
            headers=_auth(token),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["error"] is False, resp.json()

    tool_events = [event for event in session.events if event.type == "tool"]
    assert [event.sequence for event in tool_events] == [
        record.sequence for record in session.records
    ]
    assert {event.channel for event in tool_events} <= {"user_personal", "personal_cs"}

    manager.close(session.id)
    merged = build_evaluation_trajectory(session.records)
    reward_info = evaluate_simulation(
        simulation=_simulation(task, merged),
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=DOMAIN_NAME,
    )
    assert reward_info.reward == 1.0, reward_info
    assert all(message.role == "tool" or message.is_tool_call() for message in merged)
    assert all(message.content is None for message in merged if message.role != "tool")


def test_skipped_action_reward_zero(client, manager, task):
    session = manager.create_session("ctx-m0-skip", task)
    _init_env(session, task)

    # Only the first golden action (log_verification); skip submit_referral.
    scope, name, arguments = _golden_action_calls(task)[0]
    resp = client.post(
        f"/sessions/{session.id}/tools/{name}",
        json={"arguments": arguments},
        headers=_auth(AGENT_TOKEN),
    )
    assert resp.status_code == 200

    manager.close(session.id)
    merged = build_evaluation_trajectory(session.records)
    reward_info = evaluate_simulation(
        simulation=_simulation(task, merged),
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=DOMAIN_NAME,
    )
    assert reward_info.reward == 0.0, reward_info


def test_tool_listing_by_scope(client, manager, task):
    session = manager.create_session("ctx-m0-tools", task)

    user_tools = client.get(f"/sessions/{session.id}/tools", headers=_auth(USER_TOKEN))
    assert user_tools.status_code == 200
    user_names = {t["function"]["name"] for t in user_tools.json()["tools"]}
    assert user_names == set(task.user_tools)

    agent_tools = client.get(f"/sessions/{session.id}/tools", headers=_auth(AGENT_TOKEN))
    assert agent_tools.status_code == 200
    agent_names = {t["function"]["name"] for t in agent_tools.json()["tools"]}
    assert "log_verification" in agent_names
    assert "submit_referral" not in agent_names


def test_auth_and_scope_errors(client, manager, task):
    session = manager.create_session("ctx-m0-auth", task)

    # Invalid bearer token -> 401
    resp = client.get(f"/sessions/{session.id}/tools", headers=_auth("wrong"))
    assert resp.status_code == 401
    resp = client.post(
        f"/sessions/{session.id}/tools/log_verification",
        json={"arguments": {}},
        headers=_auth("wrong"),
    )
    assert resp.status_code == 401

    # Cross-scope tool name -> 404, not recorded
    resp = client.post(
        f"/sessions/{session.id}/tools/submit_referral",
        json={"arguments": {}},
        headers=_auth(AGENT_TOKEN),
    )
    assert resp.status_code == 404
    resp = client.post(
        f"/sessions/{session.id}/tools/log_verification",
        json={"arguments": {}},
        headers=_auth(USER_TOKEN),
    )
    assert resp.status_code == 404
    assert session.records == []

    # Unknown session -> 404
    resp = client.get("/sessions/nope/tools", headers=_auth(USER_TOKEN))
    assert resp.status_code == 404


def test_closed_session_409(client, manager, task):
    session = manager.create_session("ctx-m0-closed", task)
    manager.close(session.id)
    resp = client.post(
        f"/sessions/{session.id}/tools/get_current_time",
        json={"arguments": {}},
        headers=_auth(AGENT_TOKEN),
    )
    assert resp.status_code == 409
