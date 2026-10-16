"""Graphiti MCP client + config for the Hermes memory provider's working tier.

The deployed Graphiti server exposes **only** an MCP surface over streamable
HTTP — there is no REST API — so this is a small JSON-RPC client rather than a
wrapper around the MCP SDK: the provider is synchronous and threaded, the SDK is
async, and httpx is already a dependency.

Ops doc for the server: homelab ``docs/truenas-graphiti.md``.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2025-06-18"

# Both media types are mandatory. `Accept: application/json` alone is rejected
# with 406 "Client must accept both application/json and text/event-stream",
# and the server then answers in `text/event-stream` regardless.
_MCP_ACCEPT = "application/json, text/event-stream"

# cognee partitions by `node_set`; Graphiti partitions by `group_id`, and the
# partitioning is *physical* — one FalkorDB graph per group, so a mismatch is
# not a filtering bug, it is a different graph. Only `hermes_agent` renames.
GROUP_ID_BY_NODE_SET = {
    "homelab": "homelab",
    "paperclip": "paperclip",
    "hermes_agent": "hermes",
    "meta": "meta",
}


def group_id_for_node_set(node_set: str) -> str:
    """Map a cognee ``node_set`` onto its Graphiti ``group_id``.

    An unmapped partition passes through unchanged rather than falling back to
    a default: quietly collapsing an unknown agent into ``hermes`` would mix
    two agents' memories into one graph.
    """
    return GROUP_ID_BY_NODE_SET.get(node_set, node_set)


class GraphitiError(RuntimeError):
    """A Graphiti MCP call failed at the protocol or tool level."""


@dataclass
class GraphitiConfig:
    base_url: str = "http://truenas.lan:8001"
    # Empty means "derive from the cognee node_set" — see ``resolve_group_id``.
    group_id: str = ""
    request_timeout: float = 30.0
    # How many episodes to accumulate before a flush. Defaults to 1 — an
    # episode is one MCP call regardless, so holding turns back only delays
    # when their facts become recallable. Raise it only to trade freshness for
    # fewer background threads. (The queue itself matters either way — see
    # ``_flush`` in provider.py.)
    episode_buffer_size: int = 1

    _CONFIG_PREFIX = "graphiti_"

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> "GraphitiConfig":
        """Read ``graphiti_``-prefixed keys out of the shared ``memory.cognee`` block.

        The provider is registered under one Hermes config namespace, so both
        tiers' settings arrive in the same mapping; the prefix keeps
        ``base_url`` (cognee) and ``graphiti_base_url`` distinct.
        """
        data = data or {}
        known = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if not key.startswith(cls._CONFIG_PREFIX):
                continue
            name = key[len(cls._CONFIG_PREFIX) :]
            if name in known and value not in (None, ""):
                kwargs[name] = value
        return cls(**kwargs)

    @classmethod
    def from_hermes_config(cls) -> "GraphitiConfig":
        data: Dict[str, Any] = {}
        try:
            from hermes_cli.config import cfg_get, load_config

            data = cfg_get(load_config(), "memory", "cognee", default={}) or {}
        except Exception:
            logger.debug("Could not load Hermes config for graphiti", exc_info=True)
        return cls.from_mapping(data)

    def resolve_group_id(self, node_set: str) -> str:
        """The group this agent writes to: explicit config wins, else the mapping."""
        return self.group_id or group_id_for_node_set(node_set)


class GraphitiClient:
    """Thin synchronous MCP client for the remote Graphiti server.

    Thread-safe: ``httpx.Client`` is, and the MCP session handshake is guarded
    so concurrent flush threads establish it exactly once.
    """

    def __init__(self, config: GraphitiConfig, *, transport: Optional[httpx.BaseTransport] = None):
        self._config = config
        self._http = httpx.Client(
            base_url=config.base_url,
            timeout=config.request_timeout,
            headers={"Content-Type": "application/json", "Accept": _MCP_ACCEPT},
            transport=transport,
        )
        self._session_id = ""
        self._session_lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._next_id = 0

    @property
    def config(self) -> GraphitiConfig:
        return self._config

    def close(self) -> None:
        self._http.close()

    # -- public tool surface -------------------------------------------------

    def add_episode(
        self,
        *,
        name: str,
        body: str,
        source_description: str,
        source: str = "text",
        reference_time: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write one episode (``add_memory``).

        Returns as soon as the server has *queued* the episode; extraction lands
        roughly ten seconds later in the server's own per-group sequential
        worker, so this is not a completion signal.

        ``reference_time`` should be stamped when the turn happened, not when
        the flush runs — after an outage requeue those differ, and it is the
        timestamp the bi-temporal model reasons about.
        """
        # `add_memory` takes group_id SINGULAR while the read tools take
        # group_ids plural. Unknown keys are dropped silently and the call still
        # reports success, so these names are load-bearing, not cosmetic.
        args: Dict[str, Any] = {
            "name": name,
            "episode_body": body,
            "group_id": group_id or self._config.group_id,
            "source": source,
            "source_description": source_description,
        }
        if reference_time:
            args["reference_time"] = reference_time
        return self._call_tool("add_memory", args)

    def search_facts(
        self,
        query: str,
        *,
        max_facts: int = 10,
        group_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search extracted facts (``RELATES_TO`` edges) via RRF — no LLM call."""
        payload = self._call_tool(
            "search_memory_facts",
            {
                "query": query,
                "group_ids": group_ids or [self._config.group_id],
                "max_facts": max_facts,
            },
        )
        return payload.get("facts") or []

    def get_episodes(
        self,
        *,
        max_episodes: int = 10,
        group_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Read raw episodes back — the un-extracted turn text, verbatim."""
        payload = self._call_tool(
            "get_episodes",
            {
                "group_ids": group_ids or [self._config.group_id],
                # `max_episodes`, NOT `last_n`: the latter is dropped silently
                # and the empty result reads as "the tier is empty".
                "max_episodes": max_episodes,
            },
        )
        return payload.get("episodes") or []

    def status(self) -> Dict[str, Any]:
        """Server + graph-store reachability (``get_status``)."""
        return self._call_tool("get_status", {})

    # -- MCP plumbing --------------------------------------------------------

    def _new_id(self) -> str:
        with self._id_lock:
            self._next_id += 1
            return f"hermes-{self._next_id}"

    def _post(self, payload: Dict[str, Any], session_id: str = "") -> httpx.Response:
        headers = {"MCP-Protocol-Version": MCP_PROTOCOL_VERSION}
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        return self._http.post("/mcp", content=json.dumps(payload), headers=headers)

    def _ensure_session(self) -> str:
        with self._session_lock:
            if self._session_id:
                return self._session_id
            request_id = self._new_id()
            resp = self._post(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {},
                        # Identifies this MCP client implementation to the
                        # server; deliberately not the plugin version, which
                        # would drift from plugin.yaml unnoticed.
                        "clientInfo": {"name": "hermes-plugin-cognee", "version": "1"},
                    },
                }
            )
            resp.raise_for_status()
            self._envelope(resp, request_id)  # surfaces a handshake-level error
            session_id = resp.headers.get("mcp-session-id", "")
            if not session_id:
                raise GraphitiError("Graphiti MCP handshake returned no session id")
            # The spec requires this notification before any request. This
            # server does not enforce it; sending it keeps a stricter build
            # from rejecting every call.
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, session_id)
            self._session_id = session_id
            return session_id

    def _invalidate_session(self, stale: str) -> None:
        with self._session_lock:
            if self._session_id == stale:
                self._session_id = ""

    def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        session_id = self._ensure_session()
        try:
            return self._tool_call(session_id, name, arguments)
        except _SessionExpired:
            # The server drops sessions on restart. Re-handshake once; a second
            # expiry is a real fault, not a stale handle, so let it surface.
            self._invalidate_session(session_id)
            try:
                return self._tool_call(self._ensure_session(), name, arguments)
            except _SessionExpired as exc:
                raise GraphitiError(f"Graphiti MCP session rejected twice for {name}") from exc

    def _tool_call(self, session_id: str, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        request_id = self._new_id()
        resp = self._post(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            session_id,
        )
        if resp.status_code == 404:
            raise _SessionExpired(name)
        resp.raise_for_status()
        result = self._envelope(resp, request_id).get("result") or {}
        text = _content_text(result)
        if result.get("isError"):
            raise GraphitiError(f"Graphiti tool {name} failed: {text or 'no detail'}")
        return _as_payload(text)

    def _envelope(self, resp: httpx.Response, request_id: str) -> Dict[str, Any]:
        """Pull our JSON-RPC message out of an SSE- or JSON-framed response."""
        for message in _decode_messages(resp):
            if message.get("id") != request_id:
                continue
            if "error" in message:
                error = message["error"] or {}
                raise GraphitiError(f"Graphiti MCP error: {error.get('message', error)}")
            return message
        raise GraphitiError(f"No Graphiti MCP response for request {request_id}")


class _SessionExpired(RuntimeError):
    """The server no longer recognises our Mcp-Session-Id."""


def _decode_messages(resp: httpx.Response) -> List[Dict[str, Any]]:
    if "text/event-stream" not in resp.headers.get("content-type", ""):
        body = resp.json()
        return [body] if isinstance(body, dict) else []
    messages = []
    for line in resp.text.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            message = json.loads(line[len("data:") :].strip())
        except ValueError:
            continue  # keepalive / non-JSON event payload
        if isinstance(message, dict):
            messages.append(message)
    return messages


def _content_text(result: Dict[str, Any]) -> str:
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text") or ""
    return ""


def _as_payload(text: str) -> Dict[str, Any]:
    """Tool results arrive as a JSON string inside a text content block."""
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except ValueError:
        return {"message": text}
    return payload if isinstance(payload, dict) else {"result": payload}
