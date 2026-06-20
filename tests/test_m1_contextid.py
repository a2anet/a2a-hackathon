"""M1 verification: ADK's built-in A2A executor honors the client-supplied
contextId end-to-end, and the harness records a single ordered event stream."""

import uuid
from typing import AsyncGenerator

import httpx
import pytest
from a2a.client import ClientConfig, ClientFactory, minimal_agent_card
from a2a.types import Message as A2AMessage, Part, Role, Task as A2ATask, TextPart
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from tau2.data_model.message import UserMessage

from a2a_hack.bridge import A2ABridgeAgent
from a2a_hack.domain import get_hack_tasks
from a2a_hack.env_api.server import create_app
from a2a_hack.env_api.sessions import SessionManager

from conftest import free_port, incoming_text, start_server, text_event

CTX_ID = "m1-ctx-1"


class CsEchoAgent(BaseAgent):
    """Echoes its ADK session id, which must equal the A2A contextId."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        yield text_event(self, ctx, f"cs sid={ctx.session.id}")


class PersonalEchoAgent(BaseAgent):
    """Echoes its session id and relays a message to the CS agent through the
    gateway, propagating the same contextId."""

    cs_agent_url: str

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        incoming = incoming_text(ctx)
        cs_reply = await self._ask_cs(ctx.session.id, f"relay:{incoming}")
        yield text_event(
            self, ctx, f"personal sid={ctx.session.id} | {cs_reply}"
        )

    async def _ask_cs(self, context_id: str, text: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            client = ClientFactory(
                ClientConfig(streaming=False, httpx_client=http_client)
            ).create(minimal_agent_card(self.cs_agent_url, ["JSONRPC"]))
            message = A2AMessage(
                message_id=uuid.uuid4().hex,
                role=Role.user,
                parts=[Part(root=TextPart(text=text))],
                context_id=context_id,
            )
            reply = ""
            async for event in client.send_message(message):
                if isinstance(event, A2AMessage):
                    texts = [p.root.text for p in event.parts if isinstance(p.root, TextPart)]
                    reply = "\n".join(texts) or reply
                elif isinstance(event, tuple) and isinstance(event[0], A2ATask):
                    task = event[0]
                    texts = []
                    for artifact in task.artifacts or []:
                        for p in artifact.parts or []:
                            if isinstance(p.root, TextPart) and p.root.text:
                                texts.append(p.root.text)
                    reply = "\n".join(texts) or reply
            return reply


@pytest.fixture(scope="module")
def stack():
    env_port, personal_port, cs_port = free_port(), free_port(), free_port()

    manager = SessionManager(
        user_token="user-secret",
        agent_token="agent-secret",
        cs_url=f"http://127.0.0.1:{cs_port}",
    )
    env_server = start_server(create_app(manager), env_port)

    cs_app = to_a2a(CsEchoAgent(name="cs_echo"), host="127.0.0.1", port=cs_port)
    cs_server = start_server(cs_app, cs_port)

    gateway_url = f"http://127.0.0.1:{env_port}/cs-agent"
    personal_app = to_a2a(
        PersonalEchoAgent(name="personal_echo", cs_agent_url=gateway_url),
        host="127.0.0.1",
        port=personal_port,
    )
    personal_server = start_server(personal_app, personal_port)

    yield manager, f"http://127.0.0.1:{personal_port}"

    for server in (env_server, cs_server, personal_server):
        server.should_exit = True


def test_contextid_end_to_end(stack):
    manager, personal_url = stack
    task = next(t for t in get_hack_tasks() if t.id == "task_010")
    session = manager.create_session(CTX_ID, task)

    bridge = A2ABridgeAgent(
        personal_url=personal_url,
        context_id=CTX_ID,
        record_message=session.record_user_personal_message,
    )
    reply, _ = bridge.generate_next_message(
        UserMessage(role="user", content="hello agents"), None
    )

    # ADK keyed both sessions on the A2A contextId.
    assert f"personal sid={CTX_ID}" in (reply.content or ""), reply.content
    assert f"cs sid={CTX_ID}" in (reply.content or ""), reply.content

    # One ordered event stream captures both channels.
    events = session.events
    assert [event.sequence for event in events] == sorted(event.sequence for event in events)
    assert [event.channel for event in events] == [
        "user_personal",
        "personal_cs",
        "personal_cs",
        "user_personal",
    ]
    assert [event.actor for event in events] == [
        "simulated_user",
        "personal_agent",
        "customer_service_agent",
        "personal_agent",
    ]
    assert events[0].content == "hello agents"
    assert "relay:hello agents" in (events[1].content or "")
    assert f"cs sid={CTX_ID}" in (events[2].content or "")
    assert f"personal sid={CTX_ID}" in (events[3].content or "")
