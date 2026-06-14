from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parent.parent


def test_plugin_manifest_valid():
    manifest = yaml.safe_load((REPO / "plugin.yaml").read_text())
    assert manifest["name"] == "cognee"
    assert "httpx" in " ".join(manifest["pip_dependencies"])
    assert "on_session_end" in manifest["hooks"]
    assert "on_memory_write" in manifest["hooks"]


from client import CogneeConfig


def test_config_defaults():
    cfg = CogneeConfig()
    assert cfg.base_url == "http://truenas.local:8000"
    assert cfg.dataset == "main_dataset"
    assert cfg.node_set == "hermes"
    assert cfg.prefetch_search_type == "CHUNKS"
    assert cfg.tool_search_type == "GRAPH_COMPLETION"
    assert cfg.cognify_every_n_turns == 10
    assert cfg.add_buffer_size == 5
    assert cfg.request_timeout == 30.0
    assert cfg.auth_token == ""


def test_config_from_mapping_overrides_defaults():
    cfg = CogneeConfig.from_mapping(
        {"base_url": "http://x:9000", "dataset": "d", "cognify_every_n_turns": 3}
    )
    assert cfg.base_url == "http://x:9000"
    assert cfg.dataset == "d"
    assert cfg.cognify_every_n_turns == 3
    assert cfg.node_set == "hermes"  # untouched default


import httpx
from client import CogneeClient


def _client(handler):
    transport = httpx.MockTransport(handler)
    cfg = CogneeConfig(base_url="http://test")
    return CogneeClient(cfg, transport=transport)


def test_add_posts_multipart_with_dataset_and_node_set():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["content_type"] = request.headers.get("content-type", "")
        seen["body"] = request.content
        return httpx.Response(200, json={"status": "ok"})

    client = _client(handler)
    client.add(["fact one", "fact two"])

    assert seen["url"] == "http://test/api/v1/add"
    assert "multipart/form-data" in seen["content_type"]
    body = seen["body"]
    assert b"main_dataset" in body          # datasetName field
    assert b'name="node_set"' in body
    assert b"hermes" in body
    assert b"fact one" in body and b"fact two" in body


def test_add_empty_list_is_noop():
    def handler(request):
        raise AssertionError("should not call server for empty add")

    _client(handler).add([])  # no exception, no request


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


def test_cognify_posts_camelcase_background_json():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "started"})

    _client(handler).cognify()

    assert seen["url"] == "http://test/api/v1/cognify"
    assert seen["payload"] == {"datasets": ["main_dataset"], "runInBackground": True}


def test_delete_dataset_resolves_name_to_id():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/api/v1/datasets":
            return httpx.Response(200, json=[
                {"id": "ds-1", "name": "other"},
                {"id": "ds-2", "name": "main_dataset"},
            ])
        if request.method == "DELETE" and request.url.path == "/api/v1/datasets/ds-2":
            return httpx.Response(200, json={"deleted": True})
        return httpx.Response(404)

    deleted = _client(handler).delete_dataset_by_name("main_dataset")
    assert deleted is True
    assert ("GET", "/api/v1/datasets") in calls
    assert ("DELETE", "/api/v1/datasets/ds-2") in calls


def test_delete_dataset_missing_name_returns_false():
    def handler(request):
        return httpx.Response(200, json=[{"id": "x", "name": "nope"}])

    assert _client(handler).delete_dataset_by_name("absent") is False


def test_add_sends_auth_header_when_token_set():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization", "")
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    cfg = CogneeConfig(base_url="http://test", auth_token="tok123")
    CogneeClient(cfg, transport=transport).add(["x"])
    assert seen["auth"] == "Bearer tok123"
