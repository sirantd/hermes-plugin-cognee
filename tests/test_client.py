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
