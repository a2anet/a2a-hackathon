"""Session management and on-the-wire recording for the env tools API.

A session is keyed by the A2A contextId. The evaluator trajectory and the
human-readable run timeline are both recorded here from one live sequence
counter so UI ordering never depends on rewritten tau2 timestamps.
"""

import threading
from typing import Literal, Optional

from pydantic import BaseModel, Field
from tau2.data_model.message import ToolCall, ToolMessage
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.utils.utils import get_now

from a2a_hack.domain import get_hack_environment
from a2a_hack.model_usage import ModelUsageRecord

Scope = Literal["user", "agent"]
EventType = Literal["message", "tool"]
EventChannel = Literal["user_personal", "personal_cs"]
EventActor = Literal["simulated_user", "personal_agent", "customer_service_agent"]

SCOPE_TO_REQUESTOR: dict[Scope, str] = {"user": "user", "agent": "assistant"}
SCOPE_TO_CHANNEL: dict[Scope, EventChannel] = {
    "user": "user_personal",
    "agent": "personal_cs",
}
SCOPE_TO_ACTOR: dict[Scope, EventActor] = {
    "user": "personal_agent",
    "agent": "customer_service_agent",
}


class SessionError(Exception):
    """Base class for session errors, carrying an HTTP-friendly code."""

    status_code = 500


class UnknownSessionError(SessionError):
    status_code = 404


class UnknownToolError(SessionError):
    status_code = 404


class SessionClosedError(SessionError):
    status_code = 409


class RecordedCall(BaseModel):
    """One env tool call executed through the API, with its result."""

    sequence: int
    timestamp: str
    scope: Scope
    tool_call: ToolCall
    tool_message: ToolMessage


class RecordedEvent(BaseModel):
    """One ordered transcript/audit event for a simulation run."""

    sequence: int
    timestamp: str
    type: EventType
    channel: EventChannel
    actor: EventActor
    target: Optional[EventActor] = None
    content: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    tool_message: Optional[ToolMessage] = None


class RecordedA2AFailure(BaseModel):
    """One upstream A2A task failure observed during a simulation run."""

    timestamp: str
    channel: EventChannel
    actor: EventActor
    state: str
    message: str


class Session(BaseModel):
    """Live state for one simulation, keyed by contextId."""

    model_config = {"arbitrary_types_allowed": True}

    id: str
    task: Task
    env: Environment
    records: list[RecordedCall] = Field(default_factory=list)
    events: list[RecordedEvent] = Field(default_factory=list)
    a2a_failures: list[RecordedA2AFailure] = Field(default_factory=list)
    model_usage_records: list[ModelUsageRecord] = Field(default_factory=list)
    closed: bool = False

    def model_post_init(self, __context) -> None:
        self._lock = threading.Lock()
        self._sequence = 0

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def tools(self, scope: Scope) -> list[Tool]:
        """Tools visible to a scope: user scope mirrors tau2's build_user
        (filtered to task.user_tools); agent scope gets all agent tools."""
        if scope == "user":
            return self.env.get_user_tools(include=self.task.user_tools)
        return self.env.get_tools()

    def execute_tool(self, scope: Scope, name: str, arguments: dict) -> ToolMessage:
        """Execute a tool under the session lock, record it, return the result.

        Unknown tools 404 without being recorded so replay only ever sees
        calls the live env actually executed.
        """
        with self._lock:
            if self.closed:
                raise SessionClosedError(f"Session {self.id} is closed")
            if name not in {t.name for t in self.tools(scope)}:
                raise UnknownToolError(f"Unknown tool for {scope} scope: {name}")
            sequence = self._next_sequence()
            tool_call = ToolCall(
                id=f"oob-{self.id}-{sequence}",
                name=name,
                arguments=arguments,
                requestor=SCOPE_TO_REQUESTOR[scope],
            )
            tool_message = self.env.get_response(tool_call)
            timestamp = tool_message.timestamp or get_now()
            self.records.append(
                RecordedCall(
                    sequence=sequence,
                    timestamp=timestamp,
                    scope=scope,
                    tool_call=tool_call,
                    tool_message=tool_message,
                )
            )
            self.events.append(
                RecordedEvent(
                    sequence=sequence,
                    timestamp=timestamp,
                    type="tool",
                    channel=SCOPE_TO_CHANNEL[scope],
                    actor=SCOPE_TO_ACTOR[scope],
                    tool_call=tool_call,
                    tool_message=tool_message,
                )
            )
            return tool_message

    def record_user_personal_message(
        self,
        actor: Literal["simulated_user", "personal_agent"],
        content: str,
    ) -> None:
        """Record one message on the simulated-user/personal-agent channel."""
        with self._lock:
            target: EventActor = (
                "personal_agent" if actor == "simulated_user" else "simulated_user"
            )
            self.events.append(
                RecordedEvent(
                    sequence=self._next_sequence(),
                    timestamp=get_now(),
                    type="message",
                    channel="user_personal",
                    actor=actor,
                    target=target,
                    content=content,
                )
            )

    def record_personal_cs_message(
        self,
        actor: Literal["personal_agent", "customer_service_agent"],
        content: str,
    ) -> None:
        """Record one message on the personal-agent/customer-service channel."""
        with self._lock:
            target: EventActor = (
                "customer_service_agent"
                if actor == "personal_agent"
                else "personal_agent"
            )
            self.events.append(
                RecordedEvent(
                    sequence=self._next_sequence(),
                    timestamp=get_now(),
                    type="message",
                    channel="personal_cs",
                    actor=actor,
                    target=target,
                    content=content,
                )
            )

    def record_user_personal_failure(
        self,
        actor: Literal["personal_agent"],
        state: str,
        message: str,
    ) -> None:
        """Record an A2A task failure on the simulated-user/personal-agent leg."""
        self._record_a2a_failure("user_personal", actor, state, message)

    def record_personal_cs_failure(
        self,
        actor: Literal["customer_service_agent"],
        state: str,
        message: str,
    ) -> None:
        """Record an A2A task failure on the personal-agent/customer-service leg."""
        self._record_a2a_failure("personal_cs", actor, state, message)

    def first_a2a_failure(self) -> Optional[RecordedA2AFailure]:
        """Return the first observed upstream A2A task failure, if any."""
        with self._lock:
            return self.a2a_failures[0] if self.a2a_failures else None

    def raise_if_a2a_failed(self) -> None:
        """Raise when an upstream A2A task failure has been recorded."""
        from a2a_hack.a2a_errors import A2ATaskFailure

        failure = self.first_a2a_failure()
        if failure is not None:
            raise A2ATaskFailure(failure.actor, failure.state, failure.message)

    def _record_a2a_failure(
        self,
        channel: EventChannel,
        actor: EventActor,
        state: str,
        message: str,
    ) -> None:
        with self._lock:
            self.a2a_failures.append(
                RecordedA2AFailure(
                    timestamp=get_now(),
                    channel=channel,
                    actor=actor,
                    state=state,
                    message=message,
                )
            )

    def record_model_usage(self, record: ModelUsageRecord) -> None:
        """Record one normalized model usage record for this session."""
        with self._lock:
            self.model_usage_records.append(record)


class SessionManager:
    """Creates and owns sessions; holds the static per-job bearer tokens."""

    def __init__(self, user_token: str, agent_token: str, cs_url: Optional[str] = None):
        self.user_token = user_token
        self.agent_token = agent_token
        self.cs_url = cs_url
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def scope_for_token(self, token: str) -> Optional[Scope]:
        """Map a bearer token to its scope; None if invalid."""
        if token == self.user_token:
            return "user"
        if token == self.agent_token:
            return "agent"
        return None

    def create_session(self, session_id: str, task: Task) -> Session:
        """Create a session with a fresh environment for the given task."""
        with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"Session {session_id} already exists")
            session = Session(id=session_id, task=task, env=get_hack_environment())
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise UnknownSessionError(f"Unknown session: {session_id}")
        return session

    def find(self, session_id: str) -> Optional[Session]:
        """Like get() but returns None for unknown sessions (gateway capture)."""
        with self._lock:
            return self._sessions.get(session_id)

    def close(self, session_id: str) -> Session:
        """Close a session: subsequent tool calls 409. Records stay readable."""
        session = self.get(session_id)
        with session._lock:
            session.closed = True
        return session
