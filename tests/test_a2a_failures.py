"""A2A task failures must fail the harness without affecting env tool errors."""

from collections.abc import Callable
from typing import Any, NoReturn

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from tau2.data_model.message import UserMessage
from tau2.data_model.simulation import TerminationReason
from tau2.data_model.tasks import Task

import a2a_hack.runner as runner_module
from a2a_hack.a2a_errors import A2ATaskFailure
from a2a_hack.a2a_errors import is_provider_error_text
from a2a_hack.domain import get_hack_tasks
from a2a_hack.env_api.server import create_app
from a2a_hack.env_api.sessions import SessionManager
from a2a_hack.runner import run_one

from conftest import free_port, start_server


class ScriptedUser:
    """Minimal user participant that emits one message."""

    def get_init_state(self, message_history: list[Any] | None = None) -> dict[str, int]:
        """Return the initial script index."""
        return {"i": 0}

    def generate_next_message(
        self,
        message: object,
        state: dict[str, int],
    ) -> tuple[UserMessage, dict[str, int]]:
        """Return the next scripted user message."""
        state["i"] += 1
        return UserMessage(role="user", content="trigger failure"), state

    def set_seed(self, seed: int) -> None:
        """No-op: this user is deterministic."""

    def stop(self, message: object | None = None, state: object | None = None) -> None:
        """No-op: this user has no resources to release."""


class FailingBridge:
    """Bridge stand-in that records then raises an upstream A2A task failure."""

    def __init__(
        self,
        personal_url: str,
        context_id: str,
        record_message: Callable[[str, str], None],
        record_failure: Callable[[str, str, str], None],
        check_failures: Callable[[], None],
    ) -> None:
        self.record_message = record_message
        self.record_failure = record_failure

    def get_init_state(self, message_history: list[Any] | None = None) -> None:
        """Return no bridge state."""
        return None

    def generate_next_message(self, message: UserMessage, state: None) -> NoReturn:
        """Record the failure text and raise the structured failure."""
        error_text = "429 RESOURCE_EXHAUSTED"
        self.record_message("simulated_user", message.content or "")
        self.record_message("personal_agent", error_text)
        self.record_failure("personal_agent", "failed", error_text)
        raise A2ATaskFailure("personal_agent", "failed", error_text)

    def set_seed(self, seed: int) -> None:
        """No-op: this bridge is deterministic."""

    @classmethod
    def is_stop(cls, message: object) -> bool:
        """This bridge never emits a stop message."""
        return False


def _task() -> Task:
    """Return a stable task fixture."""
    return next(t for t in get_hack_tasks() if t.id == "task_010")


def _failed_task_payload(context_id: str) -> dict[str, Any]:
    """Build a JSON-RPC A2A task payload with failed status."""
    return {
        "kind": "task",
        "id": "cs-task",
        "contextId": context_id,
        "status": {
            "state": "failed",
            "message": {
                "kind": "message",
                "messageId": "cs-failed-message",
                "role": "agent",
                "parts": [{"kind": "text", "text": "429 RESOURCE_EXHAUSTED"}],
            },
        },
    }


def _provider_error_text() -> str:
    """Return the ADK/Gemini 429 text surfaced by the A2A SDK queue race."""
    return (
        "On how to mitigate this issue, please refer to:\n\n"
        "https://google.github.io/adk-docs/agents/models/google-gemini/"
        "#error-code-429-resource_exhausted\n\n"
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
        "'Resource exhausted. Please try again later.', "
        "'status': 'RESOURCE_EXHAUSTED'}}"
    )


def _completed_message_payload(text: str) -> dict[str, Any]:
    """Build an A2A message payload that carries text without failed state."""
    return {
        "kind": "message",
        "messageId": "cs-provider-error-message",
        "role": "agent",
        "parts": [{"kind": "text", "text": text}],
    }


def test_gateway_records_failed_customer_service_task() -> None:
    """A failed CS A2A task is recorded even when HTTP/JSON-RPC succeeds."""
    context_id = "ctx-cs-failed"
    port = free_port()
    upstream = FastAPI()

    @upstream.post("/")
    async def message_send(request: Request) -> dict:
        body = await request.json()
        return {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": _failed_task_payload(context_id),
        }

    upstream_server = start_server(upstream, port)
    manager = SessionManager(
        user_token="user-secret",
        agent_token="agent-secret",
        cs_url=f"http://127.0.0.1:{port}",
    )
    session = manager.create_session(context_id, _task())

    try:
        with TestClient(create_app(manager)) as client:
            response = client.post(
                "/cs-agent",
                json={
                    "jsonrpc": "2.0",
                    "id": "req-1",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "kind": "message",
                            "messageId": "personal-message",
                            "role": "user",
                            "contextId": context_id,
                            "parts": [{"kind": "text", "text": "please help"}],
                        }
                    },
                },
            )
    finally:
        upstream_server.should_exit = True

    assert response.status_code == 200
    assert len(session.a2a_failures) == 1
    failure = session.a2a_failures[0]
    assert failure.channel == "personal_cs"
    assert failure.actor == "customer_service_agent"
    assert failure.state == "failed"
    assert failure.message == "429 RESOURCE_EXHAUSTED"

    assert [event.actor for event in session.events] == [
        "personal_agent",
        "customer_service_agent",
    ]
    assert session.events[-1].content == "429 RESOURCE_EXHAUSTED"


def test_gateway_records_provider_error_text_as_failure() -> None:
    """Provider errors that ADK/A2A surfaces as text still fail the session."""
    context_id = "ctx-cs-provider-error"
    port = free_port()
    upstream = FastAPI()
    error_text = _provider_error_text()

    @upstream.post("/")
    async def message_send(request: Request) -> dict[str, Any]:
        body = await request.json()
        return {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": _completed_message_payload(error_text),
        }

    upstream_server = start_server(upstream, port)
    manager = SessionManager(
        user_token="user-secret",
        agent_token="agent-secret",
        cs_url=f"http://127.0.0.1:{port}",
    )
    session = manager.create_session(context_id, _task())

    try:
        with TestClient(create_app(manager)) as client:
            response = client.post(
                "/cs-agent",
                json={
                    "jsonrpc": "2.0",
                    "id": "req-1",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "kind": "message",
                            "messageId": "personal-message",
                            "role": "user",
                            "contextId": context_id,
                            "parts": [{"kind": "text", "text": "please help"}],
                        }
                    },
                },
            )
    finally:
        upstream_server.should_exit = True

    assert response.status_code == 200
    assert len(session.a2a_failures) == 1
    failure = session.a2a_failures[0]
    assert failure.channel == "personal_cs"
    assert failure.actor == "customer_service_agent"
    assert failure.state == "failed"
    assert failure.message == error_text
    assert session.events[-1].content == error_text


def test_provider_error_text_detection_is_narrow() -> None:
    """Bank-domain 429 text is not enough to trip the provider-error backstop."""
    assert is_provider_error_text(_provider_error_text())
    assert not is_provider_error_text("Error 429 - too many attempts. Try later.")


def test_run_one_converts_a2a_failure_to_infrastructure_error(monkeypatch) -> None:
    """A2A task failures become failed simulations with transcript events."""
    monkeypatch.setattr(runner_module, "A2ABridgeAgent", FailingBridge)
    monkeypatch.setattr(
        runner_module,
        "build_user_sim",
        lambda _llm, _task, _llm_args=None: ScriptedUser(),
    )
    manager = SessionManager(user_token="user-secret", agent_token="agent-secret")

    simulation = run_one(
        task=_task(),
        manager=manager,
        personal_url="http://personal-agent.invalid",
        user_llm="scripted",
        max_steps=2,
    )

    assert simulation.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
    assert simulation.reward_info is None
    assert simulation.info["error_type"] == "A2ATaskFailure"
    assert "429 RESOURCE_EXHAUSTED" in simulation.info["error"]
    assert [event["actor"] for event in simulation.info["events"]] == [
        "simulated_user",
        "personal_agent",
    ]
