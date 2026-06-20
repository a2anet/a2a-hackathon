"""Event timeline invariants for transcript display and audit."""

from a2a_hack.domain import get_hack_tasks
from a2a_hack.env_api.sessions import SessionManager


def test_events_share_one_sequence_across_channels():
    """Interleaved user/personal and personal/CS messages stay in record order."""
    task = next(t for t in get_hack_tasks() if t.id == "task_010")
    manager = SessionManager(user_token="user-secret", agent_token="agent-secret")
    session = manager.create_session("ctx-events", task)

    session.record_user_personal_message("simulated_user", "first user turn")
    session.record_personal_cs_message("personal_agent", "ask CS")
    session.record_personal_cs_message("customer_service_agent", "CS reply")
    session.record_user_personal_message("personal_agent", "first personal reply")
    session.record_user_personal_message("simulated_user", "second user turn")

    assert [event.sequence for event in session.events] == [1, 2, 3, 4, 5]
    assert [event.channel for event in session.events] == [
        "user_personal",
        "personal_cs",
        "personal_cs",
        "user_personal",
        "user_personal",
    ]
    assert [event.actor for event in session.events] == [
        "simulated_user",
        "personal_agent",
        "customer_service_agent",
        "personal_agent",
        "simulated_user",
    ]
