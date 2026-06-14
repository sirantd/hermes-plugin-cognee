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
