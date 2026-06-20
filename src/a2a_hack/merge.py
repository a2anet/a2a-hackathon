"""Build the tau2 evaluator trajectory from recorded tool calls.

The human-readable transcript lives in session events. tau2 scoring only needs
tool-call carrier messages immediately followed by matching ToolMessage
results, so regular chat text is intentionally excluded from this trajectory.
"""

from datetime import datetime, timedelta

from tau2.data_model.message import AssistantMessage, Message, UserMessage

from a2a_hack.env_api.sessions import RecordedCall


def _carrier_message(record: RecordedCall) -> Message:
    """Build the message that carries a recorded tool call (requestor by scope)."""
    cls = UserMessage if record.scope == "user" else AssistantMessage
    return cls(
        role=record.tool_call.requestor,
        content=None,
        tool_calls=[record.tool_call],
        timestamp=record.timestamp,
    )


def build_evaluation_trajectory(records: list[RecordedCall]) -> list[Message]:
    """Build a tau2-compatible evaluator trajectory from recorded tool calls.

    Args:
        records: The session's recorded tool calls, in execution order.

    Returns:
        Tool-call carrier and ToolMessage pairs with strictly monotonic
        timestamps and renumbered turn indexes.
    """
    merged: list[Message] = []
    for record in sorted(records, key=lambda item: item.sequence):
        merged.append(_carrier_message(record))
        merged.append(record.tool_message)

    # Rewrite timestamps strictly monotonic; keeps downstream sorts stable.
    base = datetime.now() - timedelta(seconds=len(merged))
    trajectory = []
    for i, msg in enumerate(merged):
        msg = msg.model_copy(deep=True)
        msg.timestamp = (base + timedelta(seconds=i)).isoformat()
        msg.turn_idx = i
        trajectory.append(msg)
    return trajectory
