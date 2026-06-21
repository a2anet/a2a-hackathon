"""Model usage normalization tests."""

from fastapi.testclient import TestClient
from tau2.data_model.message import AssistantMessage, UserMessage

from a2a_hack.domain import get_hack_tasks
from a2a_hack.env_api.server import create_app
from a2a_hack.env_api.sessions import SessionManager
from a2a_hack.model_usage import aggregate_model_usage, usage_records_from_messages


def test_usage_records_from_messages_extracts_cache_tokens() -> None:
    """LiteLLM raw usage cache fields are normalized into run usage."""
    message = UserMessage(
        role="user",
        content="hello",
        cost=0.25,
        usage={"prompt_tokens": 100, "completion_tokens": 20},
        raw_data={
            "model": "vertex_ai/claude-sonnet-4-6",
            "usage": {
                "prompt_tokens_details": {"cached_tokens": 40},
                "cache_creation_input_tokens": 10,
            },
        },
    )
    assistant_message = AssistantMessage(
        role="assistant",
        content="not tracked",
        cost=1.0,
        usage={"prompt_tokens": 1000, "completion_tokens": 1000},
        raw_data={"model": "team-model", "usage": {}},
    )

    records = usage_records_from_messages([message, assistant_message])
    aggregate = aggregate_model_usage(records)

    assert aggregate["calls"] == 1
    assert aggregate["input_tokens"] == 100
    assert aggregate["output_tokens"] == 20
    assert aggregate["cache_read_input_tokens"] == 40
    assert aggregate["cache_write_input_tokens"] == 10
    assert aggregate["estimated_cost_usd"] == 0.0006495
    assert aggregate["models"]["vertex_ai/claude-sonnet-4-6"]["calls"] == 1
    assert aggregate["actors"]["user_simulator"]["calls"] == 1


def test_model_usage_endpoint_records_session_usage() -> None:
    """Held-out agents can report normalized usage through the env API."""
    task = next(t for t in get_hack_tasks() if t.id == "task_010")
    manager = SessionManager(user_token="user-secret", agent_token="agent-secret")
    manager.create_session("ctx-usage", task)

    with TestClient(create_app(manager)) as client:
        response = client.post(
            "/sessions/ctx-usage/model-usage",
            headers={"Authorization": "Bearer user-secret"},
            json={
                "actor": "heldout_personal",
                "model": "claude-sonnet-4-6",
                "input_tokens": 1000,
                "output_tokens": 100,
                "cache_read_input_tokens": 750,
                "cache_write_input_tokens": 200,
            },
        )

    assert response.status_code == 200
    records = manager.get("ctx-usage").model_usage_records
    assert len(records) == 1
    assert records[0].actor == "heldout_personal"
    assert records[0].cache_read_input_tokens == 750

    aggregate = aggregate_model_usage(records)
    assert aggregate["estimated_cost_usd"] == 0.005475
