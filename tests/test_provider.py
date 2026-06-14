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


def _join(provider):
    for t in list(provider._threads):
        t.join(timeout=2)


def test_sync_turn_buffers_and_flushes_at_threshold():
    provider, fake = make_provider(add_buffer_size=2)
    provider.sync_turn("hello there", "general kenobi", session_id="sess-1")
    assert fake.added == []  # 1 record buffered, threshold 2
    provider.sync_turn("second user msg", "second assistant msg", session_id="sess-1")
    _join(provider)
    assert len(fake.added) == 1
    assert any("general kenobi" in r for r in fake.added[0])


def test_on_memory_write_mirrors_into_buffer():
    provider, fake = make_provider(add_buffer_size=1)
    provider.on_memory_write("add", "user", "lives in Brisbane")
    _join(provider)
    assert fake.added and any("Brisbane" in r for r in fake.added[0])


def test_writes_skipped_for_non_primary_context():
    provider, fake = make_provider(add_buffer_size=1)
    provider._agent_context = "cron"
    provider.sync_turn("u long enough content", "a long enough content", session_id="s")
    provider.on_memory_write("add", "memory", "should not persist")
    _join(provider)
    assert fake.added == []


def test_remove_action_not_buffered():
    provider, fake = make_provider(add_buffer_size=1)
    provider.on_memory_write("remove", "memory", "deleting this")
    _join(provider)
    assert fake.added == []


def test_cognify_fires_every_n_turns():
    provider, fake = make_provider(cognify_every_n_turns=3)
    for i in range(2):
        provider.on_turn_start(i + 1, "msg")
    _join(provider)
    assert fake.cognified == 0
    provider.on_turn_start(3, "msg")  # 3rd turn → cognify
    _join(provider)
    assert fake.cognified == 1


def test_session_end_flushes_then_cognifies():
    provider, fake = make_provider(add_buffer_size=10)
    provider.sync_turn("a meaningful user message", "a meaningful assistant reply", session_id="s")
    assert fake.added == []  # buffered, below threshold
    provider.on_session_end([{"role": "user", "content": "x"}])
    _join(provider)
    assert len(fake.added) == 1   # buffer flushed
    assert fake.cognified == 1    # then cognify


def test_session_end_skipped_for_non_primary():
    provider, fake = make_provider()
    provider._agent_context = "subagent"
    provider.on_session_end([{"role": "user", "content": "x"}])
    _join(provider)
    assert fake.cognified == 0


def test_queue_prefetch_populates_cache_consumed_by_prefetch():
    provider, fake = make_provider()
    fake.search_return = [{"text": "user is a tech lead"}]
    provider.queue_prefetch("background", session_id="sess-1")
    _join(provider)
    out = provider.prefetch("background", session_id="sess-1")
    assert "tech lead" in out
    assert "<cognee-memory>" in out
    # prefetch uses the configured fast search type
    assert fake.searched and fake.searched[0][1] == "CHUNKS"
    # cache is consumed (one-shot)
    assert provider.prefetch("background", session_id="sess-1") == ""


def test_prefetch_returns_empty_when_nothing_cached():
    provider, _ = make_provider()
    assert provider.prefetch("q", session_id="sess-1") == ""


def test_session_switch_resets_session_and_cache_on_reset():
    provider, fake = make_provider()
    fake.search_return = [{"text": "ctx"}]
    provider.queue_prefetch("q", session_id="sess-1")
    _join(provider)
    provider.on_session_switch("sess-2", reset=True)
    assert provider._session_id == "sess-2"
    assert provider.prefetch("q", session_id="sess-1") == ""  # cache cleared
