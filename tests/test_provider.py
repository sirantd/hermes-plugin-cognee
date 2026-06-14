import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("agent.memory_provider")  # needs a hermes-agent checkout on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for flat import

from provider import CogneeMemoryProvider  # noqa: E402
from client import CogneeConfig  # noqa: E402


class FakeClient:
    def __init__(self, config):
        self.config = config
        self.added = []
        self.cognified = 0
        self.searched = []
        self.deleted = []
        self.search_return = [{"text": "recalled context"}]

    def add(self, texts):
        self.added.append(list(texts))

    def cognify(self):
        self.cognified += 1

    def search(self, query, *, search_type, top_k=10, only_context=False):
        self.searched.append((query, search_type, top_k, only_context))
        return list(self.search_return)

    def delete_dataset_by_name(self, name):
        self.deleted.append(name)
        return True

    def close(self):
        pass


def make_provider(**cfg_over):
    cfg = CogneeConfig(**cfg_over)
    fake = FakeClient(cfg)
    provider = CogneeMemoryProvider(config=cfg, client=fake)
    provider.initialize("sess-1", agent_context="primary", hermes_home="/tmp/h")
    return provider, fake


def test_name_and_availability():
    provider, _ = make_provider()
    assert provider.name == "cognee"
    assert provider.is_available() is True


def test_config_schema_keys():
    provider, _ = make_provider()
    keys = {f["key"] for f in provider.get_config_schema()}
    assert {"base_url", "dataset", "node_set", "prefetch_search_type",
            "tool_search_type", "cognify_every_n_turns", "add_buffer_size"} <= keys


def test_system_prompt_mentions_tools_and_does_not_disable_builtin():
    provider, _ = make_provider()
    block = provider.system_prompt_block()
    assert "cognee_recall" in block and "cognee_remember" in block and "cognee_forget" in block
    assert "DISABLED" not in block.upper()
    assert "alongside" in block.lower()
