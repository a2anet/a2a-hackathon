"""Helpers for preserving upstream A2A task failures as harness failures."""

from collections.abc import Mapping
from typing import Any

from a2a.types import Message as A2AMessage
from a2a.types import Task
from a2a.types import TextPart

FAILED_TASK_STATES = {"failed", "canceled", "rejected"}
PROVIDER_ERROR_SIGNATURES = (
    "google.github.io/adk-docs/agents/models/google-gemini/#error-code-429",
    "resource_exhausted",
    "rate_limit_error",
)


class A2ATaskFailure(RuntimeError):
    """Raised when an upstream A2A task ends in a terminal failure state."""

    def __init__(self, actor: str, state: str, message: str) -> None:
        self.actor = actor
        self.state = state
        self.message = message.strip() or f"{actor} A2A task ended with state {state}"
        super().__init__(f"{actor} A2A task {state}: {self.message}")


def text_from_a2a_message(message: A2AMessage) -> str:
    """Return joined text parts from an A2A SDK message."""
    texts = []
    for part in message.parts or []:
        root = getattr(part, "root", part)
        if isinstance(root, TextPart) and root.text:
            texts.append(root.text)
    return "\n".join(texts)


def text_from_a2a_task(task: Task) -> str:
    """Return joined artifact and status-message text from an A2A SDK task."""
    texts = []
    for artifact in task.artifacts or []:
        for part in artifact.parts or []:
            root = getattr(part, "root", part)
            if isinstance(root, TextPart) and root.text:
                texts.append(root.text)
    if task.status is not None and task.status.message is not None:
        text = text_from_a2a_message(task.status.message)
        if text:
            texts.append(text)
    return "\n".join(texts)


def failed_task_state(task: Task) -> str | None:
    """Return a terminal failure state for an A2A SDK task, if present."""
    status = task.status
    state = _state_value(status.state if status is not None else None)
    return state if state in FAILED_TASK_STATES else None


def failed_payload_state(result: Mapping[str, Any]) -> str | None:
    """Return a terminal failure state for an A2A JSON task payload, if present."""
    status = result.get("status")
    if not isinstance(status, Mapping):
        return None
    state = _state_value(status.get("state"))
    return state if state in FAILED_TASK_STATES else None


def failure_from_task(task: Task, actor: str) -> A2ATaskFailure | None:
    """Build an exception for a failed A2A SDK task, if it failed."""
    state = failed_task_state(task)
    if state is None:
        return None
    return A2ATaskFailure(actor, state, text_from_a2a_task(task))


def failure_from_payload(
    result: Mapping[str, Any],
    actor: str,
    message: str,
) -> A2ATaskFailure | None:
    """Build an exception for a failed A2A JSON task payload, if it failed."""
    state = failed_payload_state(result)
    if state is None:
        return None
    return A2ATaskFailure(actor, state, message)


def failure_from_provider_error_text(actor: str, message: str) -> A2ATaskFailure | None:
    """Build an exception when agent text contains a provider/runtime failure."""
    if not is_provider_error_text(message):
        return None
    return A2ATaskFailure(actor, "failed", message)


def is_provider_error_text(message: str) -> bool:
    """Return whether text is a surfaced model-provider/runtime error."""
    text = message.lower()
    if "429" not in text:
        return False
    return any(signature in text for signature in PROVIDER_ERROR_SIGNATURES)


def _state_value(state: object) -> str:
    if state is None:
        return ""
    value = getattr(state, "value", state)
    return str(value).lower()
