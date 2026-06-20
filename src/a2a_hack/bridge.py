"""A2ABridgeAgent: the tau2 "agent" participant that forwards each user-sim
message to the team's personal agent over A2A.

Session identity rides entirely on the A2A contextId (= env session id);
there is no A2A metadata. The bridge is the A2A *client* for leg 1, so the
harness records that leg itself — no task stores, no SDK coupling on the
team side."""

import asyncio
import uuid
from collections.abc import Callable
from typing import Literal, Optional

import httpx
from a2a.client import ClientConfig, ClientFactory, minimal_agent_card
from a2a.types import AgentCard, Message as A2AMessage, Part, Role, Task, TextPart
from loguru import logger
from tau2.agent.base.participant import HalfDuplexParticipant
from tau2.data_model.message import AssistantMessage, Message, UserMessage

# Per-turn budget (one personal turn runs a whole CS sub-loop: RAG + tools).
# The whole-task budget is enforced separately by the orchestrator timeout.
DEFAULT_TURN_TIMEOUT_S = 300.0

# Orchestrator validate() rejects empty messages; coerce so one silent A2A
# reply doesn't kill the sim.
EMPTY_REPLY_PLACEHOLDER = "[no response]"
BridgeActor = Literal["simulated_user", "personal_agent"]
BridgeRecorder = Callable[[BridgeActor, str], None]


def _text_from_a2a_message(message: A2AMessage) -> str:
    texts = []
    for part in message.parts or []:
        root = getattr(part, "root", part)
        if isinstance(root, TextPart) and root.text:
            texts.append(root.text)
    return "\n".join(texts)


def _text_from_task(task: Task) -> str:
    texts = []
    for artifact in task.artifacts or []:
        for part in artifact.parts or []:
            root = getattr(part, "root", part)
            if isinstance(root, TextPart) and root.text:
                texts.append(root.text)
    if task.status is not None and task.status.message is not None:
        text = _text_from_a2a_message(task.status.message)
        if text:
            texts.append(text)
    return "\n".join(texts)


class A2ABridgeAgent(HalfDuplexParticipant[UserMessage, AssistantMessage, None]):
    """Forwards user-sim text to the personal agent and returns its reply.

    Never emits tool calls and never stops the conversation itself; the
    user sim's ###STOP### produces USER_STOP, which the evaluator accepts.
    """

    def __init__(
        self,
        personal_url: str,
        context_id: str,
        timeout: float = DEFAULT_TURN_TIMEOUT_S,
        record_message: Optional[BridgeRecorder] = None,
    ):
        self.personal_url = personal_url
        self.context_id = context_id
        self.timeout = timeout
        self.record_message = record_message
        # Minimal card: always POST to the URL we were given, ignoring
        # whatever URL the agent's own card advertises (docker networking).
        self._card: AgentCard = minimal_agent_card(personal_url, ["JSONRPC"])

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> None:
        """The bridge is stateless; the personal agent holds its own session
        state keyed by contextId."""
        return None

    def set_seed(self, seed: int) -> None:
        """No-op: the bridge has no randomness."""

    def generate_next_message(
        self, message: UserMessage, state: None
    ) -> tuple[AssistantMessage, None]:
        """Send one user turn to the personal agent and wait for the reply."""
        text = message.content or ""
        if self.record_message is not None:
            self.record_message("simulated_user", text)
        reply = asyncio.run(self._send(text))
        if not reply.strip():
            reply = EMPTY_REPLY_PLACEHOLDER
        if self.record_message is not None:
            self.record_message("personal_agent", reply)
        return AssistantMessage(role="assistant", content=reply), None

    async def _send(self, text: str) -> str:
        outgoing = A2AMessage(
            message_id=uuid.uuid4().hex,
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
            context_id=self.context_id,
        )
        async with httpx.AsyncClient(timeout=self.timeout) as http_client:
            client = ClientFactory(
                ClientConfig(streaming=False, httpx_client=http_client)
            ).create(self._card)
            reply = ""
            async for event in client.send_message(outgoing):
                if isinstance(event, A2AMessage):
                    reply = _text_from_a2a_message(event) or reply
                elif isinstance(event, tuple):
                    task = event[0]
                    if isinstance(task, Task):
                        reply = _text_from_task(task) or reply
            return reply
