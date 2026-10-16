from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parent.parent


def test_plugin_manifest_valid():
    manifest = yaml.safe_load((REPO / "plugin.yaml").read_text())
    assert manifest["name"] == "cognee"
    assert "httpx" in " ".join(manifest["pip_dependencies"])
    assert "on_session_end" in manifest["hooks"]
    assert "on_memory_write" in manifest["hooks"]
    # The periodic cognify tick died with the retarget — nothing this provider
    # writes on a turn reaches cognee any more.
    assert "on_turn_start" not in manifest["hooks"]


from client import CogneeConfig


def test_config_defaults():
    cfg = CogneeConfig()
    assert cfg.base_url == "http://truenas.lan:8000"
    assert cfg.dataset == "main_dataset"
    assert cfg.node_set == "hermes_agent"
    assert cfg.prefetch_search_type == "CHUNKS"
    assert cfg.tool_search_type == "GRAPH_COMPLETION"
    assert cfg.request_timeout == 30.0
    assert cfg.auth_token == ""


def test_config_from_mapping_overrides_defaults():
    cfg = CogneeConfig.from_mapping(
        {"base_url": "http://x:9000", "dataset": "d", "tool_search_type": "CHUNKS"}
    )
    assert cfg.base_url == "http://x:9000"
    assert cfg.dataset == "d"
    assert cfg.tool_search_type == "CHUNKS"
    assert cfg.node_set == "hermes_agent"  # untouched default


def test_working_tier_keys_do_not_leak_into_the_archive_config():
    # Both tiers share one Hermes config block; the `graphiti_` prefix is what
    # keeps `base_url` and `graphiti_base_url` from colliding.
    cfg = CogneeConfig.from_mapping(
        {"base_url": "http://cognee:8000", "graphiti_base_url": "http://graphiti:8001"}
    )
    assert cfg.base_url == "http://cognee:8000"
    assert not hasattr(cfg, "graphiti_base_url")


import httpx
from client import CogneeClient


def _client(handler):
    transport = httpx.MockTransport(handler)
    cfg = CogneeConfig(base_url="http://test")
    return CogneeClient(cfg, transport=transport)


import json


def test_search_posts_camelcase_json_and_returns_list():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=[{"text": "a"}, {"text": "b"}])

    client = _client(handler)
    out = client.search("who am i", search_type="GRAPH_COMPLETION", top_k=3, only_context=True)

    assert seen["url"] == "http://test/api/v1/search"
    assert seen["payload"] == {
        "searchType": "GRAPH_COMPLETION",
        "datasets": ["main_dataset"],
        "query": "who am i",
        "topK": 3,
        "onlyContext": True,
    }
    assert out == [{"text": "a"}, {"text": "b"}]


def test_search_wraps_non_list_response_in_list():
    def handler(request):
        return httpx.Response(200, json={"answer": "42"})

    out = _client(handler).search("q", search_type="CHUNKS")
    assert out == [{"answer": "42"}]


def test_remember_posts_multipart_with_session_and_background():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["content_type"] = request.headers.get("content-type", "")
        seen["body"] = request.content
        return httpx.Response(200, json={"status": "ok"})

    client = _client(handler)
    client.remember(["a durable fact"], session_id="sess-42")

    assert seen["url"] == "http://test/api/v1/remember"
    assert "multipart/form-data" in seen["content_type"]
    body = seen["body"]
    assert b"main_dataset" in body
    assert b'name="node_set"' in body and b"hermes_agent" in body
    assert b'name="session_id"' in body and b"sess-42" in body
    assert b'name="run_in_background"' in body and b"true" in body
    assert b"a durable fact" in body


def test_remember_omits_session_id_when_unset():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        return httpx.Response(200)

    _client(handler).remember(["fact"])
    assert b'name="session_id"' not in seen["body"]


def test_remember_empty_list_is_noop():
    def handler(request):
        raise AssertionError("should not call server for empty remember")

    _client(handler).remember([])  # no exception, no request


def test_recall_posts_camelcase_json_with_session_and_node_scope():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=[{"text": "a"}])

    client = _client(handler)
    out = client.recall(
        "who am i",
        search_type="CHUNKS",
        top_k=5,
        only_context=True,
        node_name=["hermes_agent"],
        session_id="sess-42",
    )

    assert seen["url"] == "http://test/api/v1/recall"
    assert seen["payload"] == {
        "searchType": "CHUNKS",
        "datasets": ["main_dataset"],
        "query": "who am i",
        "topK": 5,
        "onlyContext": True,
        "nodeName": ["hermes_agent"],
        "sessionId": "sess-42",
    }
    assert out == [{"text": "a"}]


def test_recall_omits_optional_scoping_fields():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _client(handler).recall("q", search_type="CHUNKS")
    assert "nodeName" not in seen["payload"]
    assert "sessionId" not in seen["payload"]


def test_forget_posts_dataset_and_memory_only_flag():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "ok"})

    _client(handler).forget("main_dataset", memory_only=True)

    assert seen["url"] == "http://test/api/v1/forget"
    assert seen["payload"] == {"dataset": "main_dataset", "memoryOnly": True}


def test_forget_defaults_to_full_deletion():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200)

    _client(handler).forget("main_dataset")
    assert seen["payload"] == {"dataset": "main_dataset", "memoryOnly": False}


def test_requests_send_auth_header_when_token_set():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization", "")
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    cfg = CogneeConfig(base_url="http://test", auth_token="tok123")
    CogneeClient(cfg, transport=transport).remember(["x"])
    assert seen["auth"] == "Bearer tok123"
