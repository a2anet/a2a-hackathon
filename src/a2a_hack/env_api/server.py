"""FastAPI app for the env tools API and the CS-agent gateway.

Tools API (bearer-scoped, session = contextId in the path):
- GET  /sessions/{cid}/tools          -> tool schemas for the caller's scope
- POST /sessions/{cid}/tools/{name}   -> execute a tool, record it

Gateway (no auth -- only reachable inside the job network):
- POST /cs-agent                       -> transparent A2A JSON-RPC passthrough
  to the real CS agent, recording both legs of the message under its contextId
- GET  /cs-agent/.well-known/...       -> agent card proxy, URL rewritten to
  the gateway so clients keep talking through us
"""

import json
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from a2a_hack.a2a_errors import failure_from_payload
from a2a_hack.env_api.sessions import Scope, SessionError, SessionManager
from a2a_hack.model_usage import ModelUsageRecord

GATEWAY_PATH = "/cs-agent"
AGENT_CARD_PATHS = [
    "/.well-known/agent-card.json",
    "/.well-known/agent.json",
]
CS_FORWARD_TIMEOUT_S = 300.0


def _bearer_scope(request: Request, manager: SessionManager) -> Optional[Scope]:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    return manager.scope_for_token(auth[7:].strip())


def _extract_text(message: Optional[dict]) -> str:
    """Join the text parts of an A2A message dict ('kind' or legacy 'type')."""
    if not isinstance(message, dict):
        return ""
    texts = []
    for part in message.get("parts") or []:
        if not isinstance(part, dict):
            continue
        if part.get("kind") == "text" or part.get("type") == "text":
            texts.append(part.get("text") or "")
    return "\n".join(t for t in texts if t)


def _extract_reply_text(result: Optional[dict]) -> str:
    """Best-effort text extraction from a message/send result (Message or Task)."""
    if not isinstance(result, dict):
        return ""
    kind = result.get("kind") or result.get("type")
    if kind == "message":
        return _extract_text(result)
    # Task result: artifacts plus the final status message
    texts = []
    for artifact in result.get("artifacts") or []:
        text = _extract_text(artifact)
        if text:
            texts.append(text)
    status_message = (result.get("status") or {}).get("message")
    text = _extract_text(status_message)
    if text:
        texts.append(text)
    return "\n".join(texts)


def create_app(manager: SessionManager, advertise_base: Optional[str] = None) -> FastAPI:
    """Build the env API app around a SessionManager.

    Args:
        manager: Owns sessions, tokens, and the real CS agent URL.
        advertise_base: Externally-reachable base URL of this server, used to
            rewrite the proxied agent card so A2A clients keep calling the
            gateway. Defaults to the per-request base URL.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(timeout=CS_FORWARD_TIMEOUT_S)
        yield
        await app.state.client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.manager = manager
    app.state.advertise_base = advertise_base

    @app.exception_handler(SessionError)
    async def session_error_handler(request: Request, exc: SessionError):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    # ------------------------------------------------------------------
    # Env tools API
    # ------------------------------------------------------------------

    # Sync endpoints: tool execution holds a per-session threading lock, so
    # run in FastAPI's threadpool instead of blocking the event loop.

    @app.get("/sessions/{cid}/tools")
    def list_tools(cid: str, request: Request):
        scope = _bearer_scope(request, manager)
        if scope is None:
            return JSONResponse(status_code=401, content={"detail": "Invalid token"})
        session = manager.get(cid)
        return {"tools": [tool.openai_schema for tool in session.tools(scope)]}

    @app.post("/sessions/{cid}/tools/{name}")
    def call_tool(cid: str, name: str, body: dict | None = None, request: Request = None):
        scope = _bearer_scope(request, manager)
        if scope is None:
            return JSONResponse(status_code=401, content={"detail": "Invalid token"})
        session = manager.get(cid)
        arguments = (body or {}).get("arguments") or {}
        if not isinstance(arguments, dict):
            return JSONResponse(
                status_code=422, content={"detail": "arguments must be an object"}
            )
        tool_message = session.execute_tool(scope, name, arguments)
        return {"content": tool_message.content, "error": tool_message.error}

    @app.post("/sessions/{cid}/model-usage")
    def record_model_usage(cid: str, request: Request, body: dict | None = None):
        scope = _bearer_scope(request, manager)
        if scope is None:
            return JSONResponse(status_code=401, content={"detail": "Invalid token"})
        session = manager.get(cid)
        record = ModelUsageRecord.model_validate(body or {})
        session.record_model_usage(record)
        return {"ok": True}

    # ------------------------------------------------------------------
    # CS-agent gateway
    # ------------------------------------------------------------------

    def _gateway_base(request: Request) -> str:
        base = app.state.advertise_base or str(request.base_url).rstrip("/")
        return f"{base}{GATEWAY_PATH}"

    @app.get(GATEWAY_PATH + "/.well-known/{card_path:path}")
    async def agent_card(card_path: str, request: Request):
        if manager.cs_url is None:
            return JSONResponse(status_code=502, content={"detail": "No CS agent configured"})
        upstream = await app.state.client.get(
            f"{manager.cs_url.rstrip('/')}/.well-known/{card_path}"
        )
        try:
            card = upstream.json()
        except json.JSONDecodeError:
            return Response(content=upstream.content, status_code=upstream.status_code)
        # Keep the client talking through the gateway for all later requests.
        if isinstance(card, dict) and "url" in card:
            card["url"] = _gateway_base(request)
        return JSONResponse(status_code=upstream.status_code, content=card)

    def _record_request_message(body: Any) -> Optional[str]:
        """Record the outgoing personal->CS message; returns the contextId."""
        if not isinstance(body, dict):
            return None
        message = (body.get("params") or {}).get("message") or {}
        context_id = message.get("contextId")
        if not context_id:
            return None
        session = manager.find(context_id)
        if session is not None:
            text = _extract_text(message)
            session.record_personal_cs_message("personal_agent", text)
        return context_id

    def _record_reply(context_id: Optional[str], result: Any) -> None:
        if not context_id:
            return
        session = manager.find(context_id)
        if session is None:
            return
        text = _extract_reply_text(result)
        session.record_personal_cs_message("customer_service_agent", text)
        if isinstance(result, dict):
            failure = failure_from_payload(result, "customer_service_agent", text)
            if failure is not None:
                session.record_personal_cs_failure(
                    "customer_service_agent",
                    failure.state,
                    failure.message,
                )

    @app.post(GATEWAY_PATH)
    async def gateway(request: Request):
        if manager.cs_url is None:
            return JSONResponse(status_code=502, content={"detail": "No CS agent configured"})
        raw_body = await request.body()
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            body = None
        method = body.get("method") if isinstance(body, dict) else None
        context_id = _record_request_message(body)

        headers = {"content-type": request.headers.get("content-type", "application/json")}
        accept = request.headers.get("accept")
        if accept:
            headers["accept"] = accept
        cs_url = manager.cs_url.rstrip("/")

        if method == "message/stream":
            # Byte passthrough for SSE; capture the final event best-effort.
            upstream_request = app.state.client.build_request(
                "POST", cs_url, content=raw_body, headers=headers
            )
            upstream = await app.state.client.send(upstream_request, stream=True)
            buffer = bytearray()

            async def relay():
                try:
                    async for chunk in upstream.aiter_bytes():
                        buffer.extend(chunk)
                        yield chunk
                finally:
                    await upstream.aclose()
                    _record_stream_reply(context_id, bytes(buffer))

            return StreamingResponse(
                relay(),
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "text/event-stream"),
            )

        upstream = await app.state.client.post(cs_url, content=raw_body, headers=headers)
        try:
            payload = upstream.json()
        except json.JSONDecodeError:
            payload = None
        if method == "message/send" and isinstance(payload, dict):
            _record_reply(context_id, payload.get("result"))
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    def _record_stream_reply(context_id: Optional[str], buffer: bytes) -> None:
        """Parse buffered SSE events and record the last meaningful reply text."""
        if not context_id:
            return
        last_result = None
        try:
            for line in buffer.decode("utf-8", errors="replace").splitlines():
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                result = event.get("result") if isinstance(event, dict) else None
                if isinstance(result, dict) and _extract_reply_text(result):
                    last_result = result
        except Exception as e:
            logger.warning(f"Gateway stream capture failed for {context_id}: {e}")
        if last_result is not None:
            _record_reply(context_id, last_result)

    return app
