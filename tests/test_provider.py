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
        self.remembered = []
        self.cognified = 0
        self.searched = []
        self.recalled = []
        self.forgotten = []
        self.search_return = [{"text": "recalled context"}]

    def add(self, texts):
        self.added.append(list(texts))

    def remember(self, texts, *, session_id=None):
        self.remembered.append((list(texts), session_id))

    def cognify(self):
        self.cognified += 1

    def search(self, query, *, search_type, top_k=10, only_context=False, node_name=None):
        self.searched.append((query, search_type, top_k, only_context, node_name))
        return list(self.search_return)

    def recall(self, query, *, search_type, top_k=10, only_context=False, node_name=None, session_id=None):
        self.recalled.append((query, search_type, top_k, only_context, node_name, session_id))
        return list(self.search_return)

    def forget(self, dataset, *, memory_only=False):
        self.forgotten.append((dataset, memory_only))

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


def test_sync_turn_uses_colon_labels_not_bracket_role_markers():
    # Bracketed role markers trip the upstream prompt-injection classifier, which
    # fails cognify for every such document while /add still reports success.
    provider, fake = make_provider(add_buffer_size=1)
    provider.sync_turn("a meaningful user message", "a meaningful assistant reply")
    _join(provider)
    record = fake.added[0][0]
    assert "[assistant]" not in record and "[user]" not in record
    assert "User:" in record and "Assistant:" in record


def test_role_markers_quoted_inside_turn_content_are_neutralized():
    # Colon-style framing is not enough on its own: a marker anywhere in the
    # payload blocks the request, including one quoted by the user.
    provider, fake = make_provider(add_buffer_size=1)
    provider.sync_turn("why does [assistant] appear here", "because <SYSTEM> was quoted")
    _join(provider)
    record = fake.added[0][0]
    assert "[assistant]" not in record and "<SYSTEM>" not in record
    assert "Assistant:" in record and "System:" in record


def test_memory_write_content_is_sanitized():
    provider, fake = make_provider(add_buffer_size=1)
    provider.on_memory_write("add", "prefs", "he pasted [assistant] into the log")
    _join(provider)
    record = fake.added[0][0]
    assert "[assistant]" not in record
    assert "[memory:prefs]" in record  # non-role bracket tags are left alone


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
    # prefetch uses recall with the configured fast search type
    assert fake.recalled and fake.recalled[0][1] == "CHUNKS"
    # prefetch is scoped to the agent's own node_set (no cross-tenant noise)
    assert fake.recalled[0][4] == ["hermes_agent"]
    # and to the current session, so same-session remembers surface promptly
    assert fake.recalled[0][5] == "sess-1"
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


def test_tool_schemas_expose_three_tools():
    provider, _ = make_provider()
    names = {s["name"] for s in provider.get_tool_schemas()}
    assert names == {"cognee_remember", "cognee_recall", "cognee_forget"}


def test_remember_tool_uses_remember_endpoint_with_session():
    provider, fake = make_provider(add_buffer_size=99)
    out = json.loads(provider.handle_tool_call("cognee_remember", {"content": "I prefer dark mode"}))
    assert out["ok"] is True
    texts, session_id = fake.remembered[0]
    assert any("dark mode" in t for t in texts)
    # session-attributed so the fact is recallable before the graph build finishes
    assert session_id == "sess-1"
    # server-side add+cognify — no client-side cognify hack anymore
    _join(provider)
    assert fake.cognified == 0


def test_recall_tool_uses_tool_search_type():
    provider, fake = make_provider()
    fake.search_return = [{"text": "found it"}]
    out = json.loads(provider.handle_tool_call("cognee_recall", {"query": "prefs", "top_k": 4}))
    assert out["ok"] is True
    assert fake.searched[0][1] == "GRAPH_COMPLETION"
    assert fake.searched[0][2] == 4
    # explicit recall is NOT node-scoped — reaches the whole dataset
    assert fake.searched[0][4] is None


def test_forget_requires_confirm():
    provider, fake = make_provider()
    out = json.loads(provider.handle_tool_call("cognee_forget", {"confirm": False}))
    assert "error" in out
    assert fake.forgotten == []


def test_forget_requires_explicit_dataset_name():
    # The dataset is shared across agents — no defaulting allowed.
    provider, fake = make_provider()
    out = json.loads(provider.handle_tool_call("cognee_forget", {"confirm": True}))
    assert "error" in out
    assert fake.forgotten == []


def test_forget_with_confirm_and_dataset_deletes():
    provider, fake = make_provider()
    out = json.loads(
        provider.handle_tool_call("cognee_forget", {"confirm": True, "dataset": "main_dataset"})
    )
    assert out["ok"] is True
    assert fake.forgotten == [("main_dataset", False)]


def test_forget_memory_only_preserves_raw_data():
    provider, fake = make_provider()
    out = json.loads(
        provider.handle_tool_call(
            "cognee_forget", {"confirm": True, "dataset": "main_dataset", "memory_only": True}
        )
    )
    assert out["ok"] is True and out["memory_only"] is True
    assert fake.forgotten == [("main_dataset", True)]


def test_flush_requeues_buffer_on_add_failure():
    provider, fake = make_provider(add_buffer_size=1)

    def failing_add(texts):
        raise RuntimeError("server down")

    fake.add = failing_add
    provider.sync_turn("a meaningful user message", "a meaningful assistant reply")
    _join(provider)
    assert provider._buffer  # records requeued, not dropped

    # Restore the client; the next flush delivers the requeued records.
    delivered = []
    fake.add = lambda texts: delivered.append(list(texts))
    provider._flush()
    assert delivered and "meaningful user message" in delivered[0][0]


def test_unknown_tool_returns_error():
    provider, _ = make_provider()
    out = json.loads(provider.handle_tool_call("cognee_bogus", {}))
    assert "error" in out


def test_remember_tool_degrades_on_client_error():
    provider, fake = make_provider(add_buffer_size=99)

    def boom(texts, *, session_id=None):
        raise RuntimeError("server down")

    fake.remember = boom
    out = json.loads(provider.handle_tool_call("cognee_remember", {"content": "x meaningful"}))
    assert "error" in out  # error surfaced to model, no exception raised


def test_queue_prefetch_noop_before_initialize():
    cfg = CogneeConfig()
    provider = CogneeMemoryProvider(config=cfg, client=FakeClient(cfg))
    # not initialize()'d → _initialized is False
    provider.queue_prefetch("q", session_id="s")
    _join(provider)
    assert provider.prefetch("q", session_id="s") == ""


def test_register_registers_provider():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cognee_pkg_init", str(Path(__file__).resolve().parent.parent / "__init__.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Ctx:
        def __init__(self):
            self.provider = None

        def register_memory_provider(self, provider):
            self.provider = provider

    ctx = Ctx()
    mod.register(ctx)
    assert ctx.provider is not None
    assert ctx.provider.name == "cognee"
