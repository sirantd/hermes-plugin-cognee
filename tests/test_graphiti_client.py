"""Wire-contract tests for the Graphiti MCP client.

These lean hard on the exact JSON-RPC payload because the Graphiti MCP server
**silently drops unknown arguments and returns an empty success** — a misspelt
`group_ids` yields `{"episodes": []}` with `isError: false`, which reads as "the
tier is empty" rather than "you called it wrong". Asserting the params dict is
the only regression guard for that.
"""

import json

import httpx
import pytest

from graphiti_client import (
    GraphitiClient,
    GraphitiConfig,
    GraphitiError,
    group_id_for_node_set,
)

SESSION_ID = "sess-abc123"


def _sse(payload: dict) -> httpx.Response:
    """Frame a JSON-RPC message the way the live server does (SSE, not JSON)."""
    body = f"event: message\ndata: {json.dumps(payload)}\n\n"
    return httpx.Response(
        200, text=body, headers={"content-type": "text/event-stream"}
    )


def _tool_result(request_id, payload: dict) -> httpx.Response:
    return _sse(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "isError": False,
            },
        }
    )


class Recorder:
    """MockTransport handler that answers the MCP handshake and records calls."""

    def __init__(self, tool_payloads=None):
        self.requests = []
        self.calls = []  # (tool_name, arguments)
        self.initializes = 0
        self.tool_payloads = tool_payloads or {}
        self.session_id = SESSION_ID

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request, body))
        method = body.get("method")
        if method == "initialize":
            self.initializes += 1
            return httpx.Response(
                200,
                text=f'event: message\ndata: {json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {"protocolVersion": "2025-06-18", "capabilities": {}}})}\n\n',
                headers={
                    "content-type": "text/event-stream",
                    "mcp-session-id": self.session_id,
                },
            )
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/call":
            name = body["params"]["name"]
            self.calls.append((name, body["params"]["arguments"]))
            payload = self.tool_payloads.get(name, {"ok": True})
            return _tool_result(body["id"], payload)
        raise AssertionError(f"unexpected MCP method {method}")


def _client(handler, **cfg_over) -> GraphitiClient:
    cfg = GraphitiConfig(base_url="http://test", group_id="hermes", **cfg_over)
    return GraphitiClient(cfg, transport=httpx.MockTransport(handler))


# -- config + partitioning ---------------------------------------------------


def test_config_defaults_point_at_the_deployed_server():
    cfg = GraphitiConfig()
    assert cfg.base_url == "http://truenas.lan:8001"
    assert cfg.request_timeout == 30.0
    assert cfg.episode_buffer_size == 1
    assert cfg.group_id == ""  # resolved from node_set unless overridden


def test_node_set_maps_onto_the_documented_group_ids():
    # docs/truenas-graphiti.md "Partitioning" — hermes_agent is the one that renames.
    assert group_id_for_node_set("hermes_agent") == "hermes"
    assert group_id_for_node_set("homelab") == "homelab"
    assert group_id_for_node_set("paperclip") == "paperclip"
    assert group_id_for_node_set("meta") == "meta"


def test_unmapped_node_set_passes_through_rather_than_defaulting():
    # Silently collapsing an unknown partition into `hermes` would cross-contaminate.
    assert group_id_for_node_set("some_other_agent") == "some_other_agent"


def test_resolve_group_id_prefers_explicit_override():
    assert GraphitiConfig(group_id="custom").resolve_group_id("hermes_agent") == "custom"
    assert GraphitiConfig().resolve_group_id("hermes_agent") == "hermes"


def test_config_from_mapping_strips_the_graphiti_prefix():
    cfg = GraphitiConfig.from_mapping(
        {
            "base_url": "http://cognee:8000",  # the cognee half — must not leak in
            "graphiti_base_url": "http://gr:8001",
            "graphiti_episode_buffer_size": 4,
        }
    )
    assert cfg.base_url == "http://gr:8001"
    assert cfg.episode_buffer_size == 4


# -- MCP transport -----------------------------------------------------------


def test_accept_header_lists_both_media_types():
    # Verified against the live server: `Accept: application/json` alone is
    # rejected 406 "Client must accept both application/json and text/event-stream".
    rec = Recorder()
    _client(rec).status()
    accept = rec.requests[0][0].headers["accept"]
    assert "application/json" in accept and "text/event-stream" in accept


def test_handshake_runs_once_and_session_id_is_echoed():
    rec = Recorder()
    client = _client(rec)
    client.status()
    client.status()
    assert rec.initializes == 1  # session reused, not re-established per call
    tool_requests = [r for r, b in rec.requests if b.get("method") == "tools/call"]
    assert all(r.headers.get("mcp-session-id") == SESSION_ID for r in tool_requests)


def test_initialized_notification_is_sent_after_handshake():
    rec = Recorder()
    _client(rec).status()
    methods = [b.get("method") for _, b in rec.requests]
    assert methods[:3] == ["initialize", "notifications/initialized", "tools/call"]


def test_expired_session_is_reestablished_and_the_call_retried():
    """A 404 on a tool call means the server dropped the session — re-handshake."""
    state = {"first": True}
    rec = Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("method") == "tools/call" and state["first"]:
            state["first"] = False
            return httpx.Response(404, json={"error": "session not found"})
        return rec(request)

    client = GraphitiClient(
        GraphitiConfig(base_url="http://test", group_id="hermes"),
        transport=httpx.MockTransport(handler),
    )
    client.status()  # must not raise
    assert rec.initializes == 2  # original handshake + one re-establish


def test_a_second_session_expiry_is_not_retried_forever():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return httpx.Response(
                200,
                text=f'event: message\ndata: {json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {}})}\n\n',
                headers={
                    "content-type": "text/event-stream",
                    "mcp-session-id": SESSION_ID,
                },
            )
        if body.get("method") == "notifications/initialized":
            return httpx.Response(202)
        return httpx.Response(404, json={"error": "session not found"})

    with pytest.raises(GraphitiError):
        _client(handler).status()


def test_json_framed_response_is_accepted_too():
    """The server frames SSE today, but MCP permits a plain JSON body."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"], "result": {}},
                headers={"mcp-session-id": SESSION_ID},
            )
        if body.get("method") == "notifications/initialized":
            return httpx.Response(202)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"status": "ok"})}],
                    "isError": False,
                },
            },
        )

    assert _client(handler).status() == {"status": "ok"}


def test_jsonrpc_error_is_raised_not_returned():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"], "result": {}},
                headers={"mcp-session-id": SESSION_ID},
            )
        if body.get("method") == "notifications/initialized":
            return httpx.Response(202)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "error": {"code": -32602, "message": "bad params"},
            },
        )

    with pytest.raises(GraphitiError, match="bad params"):
        _client(handler).status()


def test_tool_level_error_flag_is_raised():
    """`isError: true` carries the failure in the content, not the JSON-RPC envelope."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"], "result": {}},
                headers={"mcp-session-id": SESSION_ID},
            )
        if body.get("method") == "notifications/initialized":
            return httpx.Response(202)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "content": [{"type": "text", "text": "group_id is required"}],
                    "isError": True,
                },
            },
        )

    with pytest.raises(GraphitiError, match="group_id is required"):
        _client(handler).status()


def test_response_for_a_different_request_id_is_rejected():
    """Never hand a caller another request's result."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"], "result": {}},
                headers={"mcp-session-id": SESSION_ID},
            )
        if body.get("method") == "notifications/initialized":
            return httpx.Response(202)
        return _sse({"jsonrpc": "2.0", "id": "someone-else", "result": {"content": []}})

    with pytest.raises(GraphitiError):
        _client(handler).status()


# -- tool argument contracts -------------------------------------------------


def test_add_episode_sends_the_exact_add_memory_arguments():
    rec = Recorder(tool_payloads={"add_memory": {"message": "queued"}})
    client = _client(rec)
    client.add_episode(
        name="hermes turn 1",
        body="User:\nhi\n\nAssistant:\nhello",
        source_description="hermes:session-turn",
        source="message",
        reference_time="2026-08-07T05:00:00+00:00",
    )
    tool, args = rec.calls[0]
    assert tool == "add_memory"
    assert args == {
        "name": "hermes turn 1",
        "episode_body": "User:\nhi\n\nAssistant:\nhello",
        # SINGULAR here. `search_memory_facts`/`get_episodes` take `group_ids`.
        "group_id": "hermes",
        "source": "message",
        "source_description": "hermes:session-turn",
        "reference_time": "2026-08-07T05:00:00+00:00",
    }


def test_add_episode_omits_reference_time_when_unset():
    rec = Recorder()
    _client(rec).add_episode(
        name="n", body="b", source_description="d", source="text"
    )
    _, args = rec.calls[0]
    assert "reference_time" not in args


def test_add_episode_group_id_can_be_overridden_per_call():
    rec = Recorder()
    _client(rec).add_episode(
        name="n", body="b", source_description="d", group_id="hermes-probe"
    )
    assert rec.calls[0][1]["group_id"] == "hermes-probe"


def test_search_facts_sends_plural_group_ids_and_max_facts():
    rec = Recorder(tool_payloads={"search_memory_facts": {"facts": [{"fact": "a"}]}})
    out = _client(rec).search_facts("who am i", max_facts=4)
    tool, args = rec.calls[0]
    assert tool == "search_memory_facts"
    # PLURAL, and `max_facts` — not `top_k`, not `limit`.
    assert args == {"query": "who am i", "group_ids": ["hermes"], "max_facts": 4}
    assert out == [{"fact": "a"}]


def test_get_episodes_sends_plural_group_ids_and_max_episodes():
    rec = Recorder(
        tool_payloads={"get_episodes": {"episodes": [{"content": "raw turn"}]}}
    )
    out = _client(rec).get_episodes(max_episodes=3)
    tool, args = rec.calls[0]
    assert tool == "get_episodes"
    # `max_episodes` — NOT `last_n`, which is silently dropped for an empty success.
    assert args == {"group_ids": ["hermes"], "max_episodes": 3}
    assert out == [{"content": "raw turn"}]


def test_search_facts_returns_empty_list_when_the_tier_has_nothing():
    rec = Recorder(
        tool_payloads={
            "search_memory_facts": {"message": "No relevant facts found", "facts": []}
        }
    )
    assert _client(rec).search_facts("q") == []


def test_status_unwraps_the_content_payload():
    rec = Recorder(
        tool_payloads={"get_status": {"status": "ok", "message": "connected"}}
    )
    assert _client(rec).status()["status"] == "ok"
