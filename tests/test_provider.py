import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("agent.memory_provider")  # needs a hermes-agent checkout on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for flat import

from provider import (  # noqa: E402
    PINNED_SOURCE,
    TURN_SOURCE,
    CogneeMemoryProvider,
)
from client import CogneeConfig  # noqa: E402
from graphiti_client import GraphitiConfig  # noqa: E402


class FakeClient:
    """Stand-in for the cognee REST client — the ARCHIVE tier."""

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


class FakeGraphiti:
    """Stand-in for the Graphiti MCP client — the WORKING tier."""

    def __init__(self, config):
        self.config = config
        self.episodes = []
        self.searched = []
        self.facts_return = []

    def add_episode(self, *, name, body, source_description, source="text",
                    reference_time=None, group_id=None):
        self.episodes.append({
            "name": name,
            "body": body,
            "source_description": source_description,
            "source": source,
            "reference_time": reference_time,
            "group_id": group_id,
        })
        return {"message": "queued"}

    def search_facts(self, query, *, max_facts=10, group_ids=None):
        self.searched.append((query, max_facts, group_ids))
        return list(self.facts_return)

    def close(self):
        pass


def make_provider(graphiti_over=None, **cfg_over):
    cfg = CogneeConfig(**cfg_over)
    gcfg = GraphitiConfig(**(graphiti_over or {}))
    fake = FakeClient(cfg)
    fake_graphiti = FakeGraphiti(gcfg)
    provider = CogneeMemoryProvider(
        config=cfg, client=fake, graphiti_config=gcfg, graphiti_client=fake_graphiti
    )
    provider.initialize("sess-1", agent_context="primary", hermes_home="/tmp/h")
    return provider, fake, fake_graphiti


def _join(provider):
    for t in list(provider._threads):
        t.join(timeout=2)


def test_name_and_availability():
    provider, _, _ = make_provider()
    assert provider.name == "cognee"
    assert provider.is_available() is True


def test_config_schema_covers_both_tiers():
    provider, _, _ = make_provider()
    keys = {f["key"] for f in provider.get_config_schema()}
    assert {"base_url", "dataset", "node_set", "prefetch_search_type", "tool_search_type"} <= keys
    assert {"graphiti_base_url", "graphiti_group_id", "graphiti_episode_buffer_size"} <= keys
    # Turn capture no longer reaches cognee, so these knobs would mislead an
    # operator running `hermes memory setup`.
    assert "cognify_every_n_turns" not in keys
    assert "add_buffer_size" not in keys


def test_system_prompt_mentions_tools_and_does_not_disable_builtin():
    provider, _, _ = make_provider()
    block = provider.system_prompt_block()
    assert "cognee_recall" in block and "cognee_remember" in block and "cognee_forget" in block
    assert "DISABLED" not in block.upper()
    assert "alongside" in block.lower()


# -- turn capture → Graphiti episodes ---------------------------------------


def test_sync_turn_writes_a_graphiti_episode_and_never_touches_cognee():
    provider, fake, graphiti = make_provider()
    provider.sync_turn("hello there friend", "general kenobi", session_id="sess-1")
    _join(provider)
    assert len(graphiti.episodes) == 1
    assert "general kenobi" in graphiti.episodes[0]["body"]
    # The acceptance criterion, asserted directly.
    assert fake.added == [] and fake.cognified == 0


def test_turn_episode_is_tagged_as_a_session_turn_not_pinned():
    provider, _, graphiti = make_provider()
    provider.sync_turn("a meaningful user message", "a meaningful assistant reply")
    _join(provider)
    episode = graphiti.episodes[0]
    assert episode["source_description"] == TURN_SOURCE
    assert episode["source_description"] != PINNED_SOURCE
    # `message` is the episode type graphiti pairs with "actor: content" framing.
    assert episode["source"] == "message"


def test_episodes_land_in_the_mapped_group_id():
    # node_set `hermes_agent` maps onto group_id `hermes` — see docs/truenas-graphiti.md.
    provider, _, graphiti = make_provider()
    provider.sync_turn("a meaningful user message", "a meaningful assistant reply")
    _join(provider)
    assert graphiti.episodes[0]["group_id"] == "hermes"


def test_explicit_group_id_config_overrides_the_node_set_mapping():
    provider, _, graphiti = make_provider(graphiti_over={"group_id": "hermes-staging"})
    provider.sync_turn("a meaningful user message", "a meaningful assistant reply")
    _join(provider)
    assert graphiti.episodes[0]["group_id"] == "hermes-staging"


def test_a_renamed_node_set_carries_through_to_the_group_id():
    provider, _, graphiti = make_provider(node_set="homelab")
    provider.sync_turn("a meaningful user message", "a meaningful assistant reply")
    _join(provider)
    assert graphiti.episodes[0]["group_id"] == "homelab"


def test_sync_turn_uses_colon_labels_not_bracket_role_markers():
    # Colon framing was chosen deliberately and is what graphiti's `message`
    # extraction prompt expects; it also keeps the archive path safe.
    provider, _, graphiti = make_provider()
    provider.sync_turn("a meaningful user message", "a meaningful assistant reply")
    _join(provider)
    body = graphiti.episodes[0]["body"]
    assert "[assistant]" not in body and "[user]" not in body
    assert "User:" in body and "Assistant:" in body


def test_role_markers_quoted_inside_turn_content_are_still_neutralized():
    # Defence in depth: the working tier's LLM is local and does not block
    # markers, but the sanitizer stays at the choke point.
    provider, _, graphiti = make_provider()
    provider.sync_turn("why does [assistant] appear here", "because <SYSTEM> was quoted")
    _join(provider)
    body = graphiti.episodes[0]["body"]
    assert "[assistant]" not in body and "<SYSTEM>" not in body
    assert "Assistant:" in body and "System:" in body


def test_reference_time_is_stamped_at_enqueue_not_at_flush():
    """Bi-temporal `valid_at` must reflect when the turn happened.

    After an outage the flush can run much later, so a server-side default
    would date every requeued turn to the recovery moment.
    """
    provider, _, graphiti = make_provider(graphiti_over={"episode_buffer_size": 99})
    provider.sync_turn("a meaningful user message", "a meaningful assistant reply")
    stamped = provider._buffer[0].reference_time
    assert stamped and stamped.endswith("+00:00")
    provider._flush()
    assert graphiti.episodes[0]["reference_time"] == stamped


def test_short_turns_are_skipped():
    provider, _, graphiti = make_provider()
    provider.sync_turn("hi", "ok")
    _join(provider)
    assert graphiti.episodes == []


def test_turn_episodes_are_named_with_an_increasing_sequence():
    provider, _, graphiti = make_provider()
    provider.sync_turn("a meaningful user message", "a meaningful assistant reply")
    provider.sync_turn("another meaningful message", "another meaningful reply")
    _join(provider)
    assert [e["name"] for e in graphiti.episodes] == ["hermes turn 1", "hermes turn 2"]


# -- explicit [memory:…] writes → pinned episodes ----------------------------


def test_memory_write_lands_as_a_pinned_episode():
    provider, _, graphiti = make_provider()
    provider.on_memory_write("add", "user", "lives in Brisbane")
    _join(provider)
    episode = graphiti.episodes[0]
    assert "Brisbane" in episode["body"]
    # #57's retention job has no other context — this marker is the contract.
    assert episode["source_description"] == PINNED_SOURCE
    assert episode["source"] == "text"


def test_pinned_and_turn_episodes_are_distinguishable_from_each_other():
    provider, _, graphiti = make_provider()
    provider.on_memory_write("add", "prefs", "prefers dark mode always")
    provider.sync_turn("a meaningful user message", "a meaningful assistant reply")
    _join(provider)
    markers = {e["source_description"] for e in graphiti.episodes}
    assert markers == {PINNED_SOURCE, TURN_SOURCE}


def test_memory_write_content_is_sanitized():
    provider, _, graphiti = make_provider()
    provider.on_memory_write("add", "prefs", "he pasted [assistant] into the log")
    _join(provider)
    body = graphiti.episodes[0]["body"]
    assert "[assistant]" not in body
    assert "[memory:prefs]" in body  # non-role bracket tags are left alone


def test_remove_action_is_not_written():
    provider, _, graphiti = make_provider()
    provider.on_memory_write("remove", "memory", "deleting this")
    _join(provider)
    assert graphiti.episodes == []


def test_writes_skipped_for_non_primary_context():
    provider, fake, graphiti = make_provider()
    provider._agent_context = "cron"
    provider.sync_turn("u long enough content", "a long enough content", session_id="s")
    provider.on_memory_write("add", "memory", "should not persist")
    _join(provider)
    assert graphiti.episodes == [] and fake.added == []


# -- the write queue ---------------------------------------------------------


def test_episodes_are_buffered_up_to_the_configured_depth():
    provider, _, graphiti = make_provider(graphiti_over={"episode_buffer_size": 2})
    provider.sync_turn("hello there friend", "general kenobi")
    assert graphiti.episodes == []  # 1 buffered, threshold 2
    provider.sync_turn("second user message", "second assistant message")
    _join(provider)
    assert len(graphiti.episodes) == 2


def test_flush_requeues_only_the_undelivered_tail():
    """A mid-batch failure must not resend delivered episodes as duplicates."""
    provider, _, graphiti = make_provider(graphiti_over={"episode_buffer_size": 99})
    provider.sync_turn("first meaningful message", "first meaningful reply")
    provider.sync_turn("second meaningful message", "second meaningful reply")
    provider.sync_turn("third meaningful message", "third meaningful reply")

    delivered = []

    def flaky(*, name, body, source_description, source="text", reference_time=None, group_id=None):
        if len(delivered) == 1:
            raise RuntimeError("server down")
        delivered.append(body)
        return {}

    graphiti.add_episode = flaky
    provider._flush()

    assert len(delivered) == 1 and "first meaningful message" in delivered[0]
    assert len(provider._buffer) == 2  # only the two that never landed
    assert "second meaningful message" in provider._buffer[0].body


def test_requeued_episodes_are_delivered_on_the_next_flush():
    provider, _, graphiti = make_provider(graphiti_over={"episode_buffer_size": 1})

    def boom(**kwargs):
        raise RuntimeError("server down")

    original = graphiti.add_episode
    graphiti.add_episode = boom
    provider.sync_turn("a meaningful user message", "a meaningful assistant reply")
    _join(provider)
    assert provider._buffer  # requeued, not dropped

    graphiti.add_episode = original
    provider._flush()
    assert graphiti.episodes and "meaningful user message" in graphiti.episodes[0]["body"]


def test_buffer_is_bounded_during_a_sustained_outage():
    provider, _, graphiti = make_provider(graphiti_over={"episode_buffer_size": 1})

    def boom(**kwargs):
        raise RuntimeError("server down")

    graphiti.add_episode = boom
    for i in range(provider._MAX_BUFFERED + 20):
        provider.sync_turn(f"user message number {i}", f"assistant reply number {i}")
    _join(provider)
    assert len(provider._buffer) <= provider._MAX_BUFFERED
    # The newest turns survive; the oldest are the ones dropped.
    assert f"number {provider._MAX_BUFFERED + 19}" in provider._buffer[-1].body


def test_session_end_flushes_the_queue():
    provider, fake, graphiti = make_provider(graphiti_over={"episode_buffer_size": 10})
    provider.sync_turn("a meaningful user message", "a meaningful assistant reply")
    assert graphiti.episodes == []  # buffered, below threshold
    provider.on_session_end([{"role": "user", "content": "x"}])
    _join(provider)
    assert len(graphiti.episodes) == 1
    assert fake.cognified == 0  # no periodic cognify survives the retarget


def test_session_end_skipped_for_non_primary():
    provider, _, graphiti = make_provider(graphiti_over={"episode_buffer_size": 10})
    provider._agent_context = "subagent"
    provider.on_session_end([{"role": "user", "content": "x"}])
    _join(provider)
    assert graphiti.episodes == []


# -- two-tier recall ---------------------------------------------------------


def test_recall_merges_both_tiers_with_provenance():
    provider, fake, graphiti = make_provider()
    graphiti.facts_return = [{"fact": "hermes runs on bruk", "valid_at": "2026-08-01"}]
    fake.search_return = [{"text": "an archived document chunk"}]
    out = json.loads(provider.handle_tool_call("cognee_recall", {"query": "hermes", "top_k": 4}))
    assert out["ok"] is True
    assert out["working"] == [{"fact": "hermes runs on bruk", "valid_at": "2026-08-01"}]
    assert out["archive"] == [{"text": "an archived document chunk"}]
    assert "degraded" not in out


def test_recall_scopes_the_working_tier_to_this_agents_group():
    provider, _, graphiti = make_provider()
    provider.handle_tool_call("cognee_recall", {"query": "prefs", "top_k": 4})
    query, max_facts, group_ids = graphiti.searched[0]
    assert group_ids == ["hermes"]
    assert max_facts == 4


def test_recall_leaves_the_archive_query_dataset_wide():
    provider, fake, _ = make_provider()
    provider.handle_tool_call("cognee_recall", {"query": "prefs", "top_k": 4})
    assert fake.searched[0][1] == "GRAPH_COMPLETION"
    assert fake.searched[0][2] == 4
    assert fake.searched[0][4] is None  # not node-scoped — reaches the whole dataset


def test_graphiti_outage_degrades_recall_to_archive_only():
    provider, fake, graphiti = make_provider()
    fake.search_return = [{"text": "an archived document chunk"}]

    def boom(*a, **kw):
        raise RuntimeError("graphiti down")

    graphiti.search_facts = boom
    out = json.loads(provider.handle_tool_call("cognee_recall", {"query": "hermes"}))
    assert out["ok"] is True
    # The archive half is exactly what it would have been without the working tier.
    assert out["archive"] == [{"text": "an archived document chunk"}]
    assert out["working"] == []
    assert out["degraded"] == ["working"]


def test_cognee_outage_degrades_recall_to_working_only():
    provider, fake, graphiti = make_provider()
    graphiti.facts_return = [{"fact": "hermes runs on bruk"}]

    def boom(*a, **kw):
        raise RuntimeError("cognee down")

    fake.search = boom
    out = json.loads(provider.handle_tool_call("cognee_recall", {"query": "hermes"}))
    assert out["ok"] is True
    assert out["working"] == [{"fact": "hermes runs on bruk"}]
    assert out["degraded"] == ["archive"]


def test_recall_errors_only_when_both_tiers_are_down():
    provider, fake, graphiti = make_provider()

    def boom(*a, **kw):
        raise RuntimeError("everything down")

    graphiti.search_facts = boom
    fake.search = boom
    out = json.loads(provider.handle_tool_call("cognee_recall", {"query": "hermes"}))
    assert "error" in out


# -- prefetch ----------------------------------------------------------------


def test_prefetch_returns_both_tiers_in_labelled_blocks():
    provider, fake, graphiti = make_provider()
    graphiti.facts_return = [{"fact": "user is a tech lead"}]
    fake.search_return = [{"text": "an archived chunk"}]
    provider.queue_prefetch("background", session_id="sess-1")
    _join(provider)
    out = provider.prefetch("background", session_id="sess-1")
    assert "<working-memory>" in out and "user is a tech lead" in out
    assert "<cognee-memory>" in out and "an archived chunk" in out
    assert provider.prefetch("background", session_id="sess-1") == ""  # one-shot


def test_prefetch_working_tier_is_group_scoped():
    provider, _, graphiti = make_provider()
    graphiti.facts_return = [{"fact": "x"}]
    provider.queue_prefetch("background", session_id="sess-1")
    _join(provider)
    assert graphiti.searched[0][2] == ["hermes"]


def test_prefetch_archive_stays_node_and_session_scoped():
    provider, fake, _ = make_provider()
    fake.search_return = [{"text": "ctx"}]
    provider.queue_prefetch("background", session_id="sess-1")
    _join(provider)
    assert fake.recalled[0][1] == "CHUNKS"
    assert fake.recalled[0][4] == ["hermes_agent"]
    assert fake.recalled[0][5] == "sess-1"


def test_prefetch_omits_a_tier_that_returned_nothing():
    provider, fake, graphiti = make_provider()
    graphiti.facts_return = []
    fake.search_return = [{"text": "an archived chunk"}]
    provider.queue_prefetch("q", session_id="sess-1")
    _join(provider)
    out = provider.prefetch("q", session_id="sess-1")
    assert "<working-memory>" not in out
    assert "<cognee-memory>" in out


def test_prefetch_survives_a_working_tier_outage():
    provider, fake, graphiti = make_provider()
    fake.search_return = [{"text": "an archived chunk"}]

    def boom(*a, **kw):
        raise RuntimeError("graphiti down")

    graphiti.search_facts = boom
    provider.queue_prefetch("q", session_id="sess-1")
    _join(provider)
    assert "an archived chunk" in provider.prefetch("q", session_id="sess-1")


def test_prefetch_returns_empty_when_nothing_cached():
    provider, _, _ = make_provider()
    assert provider.prefetch("q", session_id="sess-1") == ""


def test_queue_prefetch_noop_before_initialize():
    cfg = CogneeConfig()
    gcfg = GraphitiConfig()
    provider = CogneeMemoryProvider(
        config=cfg, client=FakeClient(cfg), graphiti_config=gcfg, graphiti_client=FakeGraphiti(gcfg)
    )
    provider.queue_prefetch("q", session_id="s")
    _join(provider)
    assert provider.prefetch("q", session_id="s") == ""


def test_session_switch_resets_session_cache_and_turn_sequence():
    provider, fake, graphiti = make_provider()
    graphiti.facts_return = [{"fact": "ctx"}]
    provider.queue_prefetch("q", session_id="sess-1")
    _join(provider)
    provider.sync_turn("a meaningful user message", "a meaningful assistant reply")
    _join(provider)
    provider.on_session_switch("sess-2", reset=True)
    assert provider._session_id == "sess-2"
    assert provider.prefetch("q", session_id="sess-1") == ""  # cache cleared
    provider.sync_turn("a fresh meaningful message", "a fresh meaningful reply")
    _join(provider)
    assert graphiti.episodes[-1]["name"] == "hermes turn 1"  # sequence restarted


# -- tools -------------------------------------------------------------------


def test_tool_schemas_expose_three_tools():
    provider, _, _ = make_provider()
    names = {s["name"] for s in provider.get_tool_schemas()}
    assert names == {"cognee_remember", "cognee_recall", "cognee_forget"}


def test_remember_tool_writes_to_the_archive_tier():
    provider, fake, graphiti = make_provider()
    out = json.loads(provider.handle_tool_call("cognee_remember", {"content": "I prefer dark mode"}))
    assert out["ok"] is True
    texts, session_id = fake.remembered[0]
    assert any("dark mode" in t for t in texts)
    assert session_id == "sess-1"
    # Explicit durable facts are the one write path still aimed at cognee — it
    # is what keeps the archive tier writable during a Graphiti outage.
    assert graphiti.episodes == []


def test_remember_tool_sanitizes_role_markers_on_the_archive_path():
    """The archive path still reaches OpenRouter, whose classifier blocks these.

    Previously unsanitized: a marker in an explicitly remembered fact was
    accepted by /remember and then failed every extraction call.
    """
    provider, fake, _ = make_provider()
    provider.handle_tool_call(
        "cognee_remember", {"content": "he pasted [assistant] and <system> into the log"}
    )
    texts, _ = fake.remembered[0]
    assert "[assistant]" not in texts[0] and "<system>" not in texts[0]
    assert "Assistant:" in texts[0] and "System:" in texts[0]


def test_remember_tool_rejects_empty_content():
    provider, fake, _ = make_provider()
    out = json.loads(provider.handle_tool_call("cognee_remember", {"content": "   "}))
    assert "error" in out
    assert fake.remembered == []


def test_remember_tool_degrades_on_client_error():
    provider, fake, _ = make_provider()

    def boom(texts, *, session_id=None):
        raise RuntimeError("server down")

    fake.remember = boom
    out = json.loads(provider.handle_tool_call("cognee_remember", {"content": "x meaningful"}))
    assert "error" in out  # surfaced to the model, no exception raised


def test_forget_requires_confirm():
    provider, fake, _ = make_provider()
    out = json.loads(provider.handle_tool_call("cognee_forget", {"confirm": False}))
    assert "error" in out
    assert fake.forgotten == []


def test_forget_requires_explicit_dataset_name():
    provider, fake, _ = make_provider()
    out = json.loads(provider.handle_tool_call("cognee_forget", {"confirm": True}))
    assert "error" in out
    assert fake.forgotten == []


def test_forget_with_confirm_and_dataset_deletes():
    provider, fake, _ = make_provider()
    out = json.loads(
        provider.handle_tool_call("cognee_forget", {"confirm": True, "dataset": "main_dataset"})
    )
    assert out["ok"] is True
    assert fake.forgotten == [("main_dataset", False)]


def test_forget_memory_only_preserves_raw_data():
    provider, fake, _ = make_provider()
    out = json.loads(
        provider.handle_tool_call(
            "cognee_forget", {"confirm": True, "dataset": "main_dataset", "memory_only": True}
        )
    )
    assert out["ok"] is True and out["memory_only"] is True
    assert fake.forgotten == [("main_dataset", True)]


def test_forget_never_reaches_the_working_tier():
    # cognee_forget names a cognee *dataset*; the working tier is partitioned by
    # group_id and has no dataset concept. Dropping a group is #57's job.
    provider, _, graphiti = make_provider()
    provider.handle_tool_call("cognee_forget", {"confirm": True, "dataset": "main_dataset"})
    assert graphiti.episodes == []


def test_unknown_tool_returns_error():
    provider, _, _ = make_provider()
    out = json.loads(provider.handle_tool_call("cognee_bogus", {}))
    assert "error" in out


def test_shutdown_flushes_and_closes_both_clients():
    provider, fake, graphiti = make_provider(graphiti_over={"episode_buffer_size": 99})
    closed = []
    fake.close = lambda: closed.append("cognee")
    graphiti.close = lambda: closed.append("graphiti")
    provider.sync_turn("a meaningful user message", "a meaningful assistant reply")
    provider.shutdown()
    assert len(graphiti.episodes) == 1
    assert set(closed) == {"cognee", "graphiti"}


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
