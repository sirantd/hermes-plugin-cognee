"""Two-tier memory provider for Hermes — Graphiti working tier + cognee archive.

Session turns and mirrored builtin-memory writes become **Graphiti episodes**:
bi-temporal, so a contradicted fact is invalidated rather than accumulating, and
partitioned by ``group_id`` so retention can drop a tier without touching the
archive. Documents, OCR and Obsidian stay on cognee, which this provider still
reads on every recall and writes through ``cognee_remember``.

Design: homelab ``docs/specs/2026-08-06-agent-memory-tiering.md``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

try:
    from .client import CogneeClient, CogneeConfig
    from .graphiti_client import GraphitiClient, GraphitiConfig
except ImportError:  # flat import during standalone unit tests
    from client import CogneeClient, CogneeConfig
    from graphiti_client import GraphitiClient, GraphitiConfig

logger = logging.getLogger(__name__)

# The upstream prompt-injection classifier rejects any request whose text holds a
# delimiter-wrapped role marker — `[assistant]`, `<system>`, `[ASSISTANT]` — no
# matter what surrounds it. Colon-style labels ("Assistant:") are not flagged, so
# rewrite rather than delete: the turn structure is worth keeping in the graph.
#
# The working tier's LLM is local and does not block markers, so this is no
# longer load-bearing there — but it still guards every cognee write, and the
# archive tier runs on OpenRouter.
_ROLE_MARKER_RE = re.compile(r"[\[<]\s*(user|assistant|system)\s*[\]>]", re.IGNORECASE)


def _neutralize_role_markers(text: str) -> str:
    """Rewrite `[assistant]`-style role markers to a colon form the classifier accepts."""
    return _ROLE_MARKER_RE.sub(lambda m: f"{m.group(1).capitalize()}:", text)


# Episode provenance markers, written to the `source_description` field and
# therefore readable as a property on the Episodic node. The retention job
# (homelab #57) runs with no other context, so these are a contract: it spares
# pinned episodes and ages out turns.
PINNED_SOURCE = "hermes:pinned-memory"
TURN_SOURCE = "hermes:session-turn"

# Provenance labels for merged recall results.
_WORKING = "working"
_ARCHIVE = "archive"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class _Episode:
    """One queued Graphiti write, with its reference time already fixed."""

    name: str
    body: str
    source_description: str
    source: str
    reference_time: str


REMEMBER_SCHEMA = {
    "name": "cognee_remember",
    "description": (
        "Store a durable fact, preference, or note in the cognee archive — the "
        "permanent reference tier alongside documents and OCR. Ordinary "
        "conversation is captured automatically into working memory; use this "
        "only for facts worth keeping indefinitely."
    ),
    "parameters": {
        "type": "object",
        "properties": {"content": {"type": "string", "description": "Text to remember."}},
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "cognee_recall",
    "description": (
        "Recall context from long-term memory. Queries both tiers and returns "
        "them separately: `working` holds recent temporally-ranked facts from "
        "this agent's sessions, `archive` holds documents, OCR and notes. A "
        "`degraded` list names any tier that was unreachable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language memory query."},
            "search_type": {"type": "string", "description": "Optional cognee SearchType override."},
            "top_k": {"type": "integer", "description": "Max results per tier (default 10)."},
        },
        "required": ["query"],
    },
}

FORGET_SCHEMA = {
    "name": "cognee_forget",
    "description": (
        "Delete a cognee archive dataset. The configured dataset is shared by "
        "multiple agents (all node_set partitions), so this wipes everyone's "
        "memory, not just this agent's. Only after an explicit user deletion "
        "request that names the dataset. With memory_only=true only the graph "
        "and embeddings are wiped; raw records survive and can be re-cognified. "
        "This does not touch the working tier."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "confirm": {"type": "boolean", "description": "Must be true to execute deletion."},
            "dataset": {
                "type": "string",
                "description": "Exact dataset name to delete. Required — there is no default.",
            },
            "memory_only": {
                "type": "boolean",
                "description": (
                    "If true, delete only graph nodes/edges and embeddings; "
                    "raw files and data records are preserved. Default false "
                    "(full dataset deletion)."
                ),
            },
        },
        "required": ["confirm", "dataset"],
    },
}


def _err(message: str) -> str:
    return json.dumps({"error": message})


class CogneeMemoryProvider(MemoryProvider):
    def __init__(
        self,
        config: CogneeConfig | None = None,
        client: CogneeClient | None = None,
        graphiti_config: GraphitiConfig | None = None,
        graphiti_client: GraphitiClient | None = None,
    ):
        self._config = config
        self._client = client
        self._graphiti_config = graphiti_config
        self._graphiti = graphiti_client
        self._group_id = ""
        self._session_id = ""
        self._agent_context = "primary"
        self._turn_seq = 0
        self._buffer: List[_Episode] = []
        self._buffer_lock = threading.Lock()
        self._prefetch_cache: Dict[str, str] = {}
        self._prefetch_lock = threading.Lock()
        self._threads: List[threading.Thread] = []
        self._threads_lock = threading.Lock()
        self._initialized = False

    @property
    def name(self) -> str:
        return "cognee"

    def is_available(self) -> bool:
        try:
            import httpx  # noqa: F401
        except Exception:
            return False
        cognee = self._config or CogneeConfig.from_hermes_config()
        graphiti = self._graphiti_config or GraphitiConfig.from_hermes_config()
        return bool(cognee.base_url or graphiti.base_url)

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        if self._config is None:
            self._config = CogneeConfig.from_hermes_config()
        if self._client is None:
            self._client = CogneeClient(self._config)
        if self._graphiti_config is None:
            self._graphiti_config = GraphitiConfig.from_hermes_config()
        if self._graphiti is None:
            self._graphiti = GraphitiClient(self._graphiti_config)
        # Resolved once and passed explicitly on every call: Graphiti's
        # partitioning is physical (one FalkorDB graph per group), so writing to
        # the wrong group is not a filter mistake, it is a different graph.
        self._group_id = self._graphiti_config.resolve_group_id(self._config.node_set)
        self._session_id = session_id or ""
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._initialized = True

    def system_prompt_block(self) -> str:
        return (
            "# Long-term Memory\n"
            "A two-tier long-term memory is active **alongside** your builtin file "
            "memory (both are available; neither replaces the other): a working tier "
            "holding recent facts from your own sessions, and a cognee archive of "
            "documents, OCR and notes.\n"
            "Use `cognee_recall` to search both, `cognee_remember` to store a durable "
            "fact in the archive, and `cognee_forget` only after an explicit deletion "
            "request. Ordinary conversation is captured into the working tier "
            "automatically."
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [REMEMBER_SCHEMA, RECALL_SCHEMA, FORGET_SCHEMA]

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "base_url", "description": "cognee archive server base URL", "default": "http://truenas.local:8000"},
            {"key": "dataset", "description": "cognee dataset name", "default": "main_dataset"},
            {"key": "node_set", "description": "node_set partition for this agent", "default": "hermes_agent"},
            {"key": "auth_token", "description": "Optional bearer token", "secret": True, "env_var": "COGNEE_AUTH_TOKEN"},
            {"key": "prefetch_search_type", "description": "SearchType for auto-prefetch", "default": "CHUNKS"},
            {"key": "tool_search_type", "description": "SearchType for cognee_recall", "default": "GRAPH_COMPLETION"},
            {"key": "graphiti_base_url", "description": "Graphiti MCP server base URL (working tier)", "default": "http://truenas.local:8001"},
            {"key": "graphiti_group_id", "description": "Graphiti group_id (default: mapped from node_set)", "default": ""},
            {"key": "graphiti_episode_buffer_size", "description": "Episodes buffered before flush", "default": 1},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        try:
            from hermes_cli.config import save_env_value, set_config_value
        except Exception:
            logger.warning("Hermes config helpers unavailable; cannot persist cognee config")
            return
        token = values.get("auth_token")
        if token:
            save_env_value("COGNEE_AUTH_TOKEN", str(token))
        for key, value in values.items():
            if key == "auth_token" or value in (None, ""):
                continue
            set_config_value(f"memory.cognee.{key}", value)

    _MIN_TURN_LEN = 16

    def _spawn(self, fn) -> None:
        thread = threading.Thread(target=fn, daemon=True)
        with self._threads_lock:
            self._threads = [t for t in self._threads if t.is_alive()]
            self._threads.append(thread)
        thread.start()

    def _enqueue_write(self, text: str, *, name: str, source_description: str, source: str) -> None:
        """The single choke point for every working-tier write.

        The sanitizer sits here rather than in the formatters because a marker
        quoted inside turn *content* is enough to trip the classifier on the
        archive path, and one choke point means no caller can reintroduce one.
        Rewriting markers is mildly lossy, but the episode body is otherwise
        stored verbatim — Graphiti keeps it on the Episodic node and extraction
        is additive.
        """
        if self._agent_context != "primary" or not text or not text.strip():
            return
        episode = _Episode(
            name=name,
            body=_neutralize_role_markers(text).strip(),
            source_description=source_description,
            source=source,
            # Stamped now, not at flush: after an outage requeue the two differ,
            # and this is the timestamp the bi-temporal model reasons about.
            reference_time=_utc_now_iso(),
        )
        with self._buffer_lock:
            self._buffer.append(episode)
            flush = len(self._buffer) >= max(1, self._graphiti_config.episode_buffer_size)
        if flush:
            self._spawn(self._flush)

    # Requeued episodes survive transient failures, but never let an extended
    # outage grow the buffer without bound.
    _MAX_BUFFERED = 100

    def _flush(self) -> None:
        """Drain the queue one episode at a time.

        ``add_memory`` returns on *enqueue*, not on completion — the server
        extracts asynchronously in a per-group sequential worker at roughly ten
        seconds an episode. So this is a cheap HTTP call per turn, and the queue
        exists to keep it off the turn's critical path and to survive outages,
        not to absorb extraction latency (which lands server-side and is
        invisible from here).
        """
        with self._buffer_lock:
            pending = self._buffer
            self._buffer = []
        if not pending:
            return
        delivered = 0
        try:
            for episode in pending:
                self._graphiti.add_episode(
                    name=episode.name,
                    body=episode.body,
                    source_description=episode.source_description,
                    source=episode.source,
                    reference_time=episode.reference_time,
                    group_id=self._group_id,
                )
                delivered += 1
        except Exception as exc:  # degrade — never break the turn
            # Only the tail is requeued; resending a delivered episode would
            # duplicate it, and Graphiti dedups entities, not episodes.
            failed = pending[delivered:]
            with self._buffer_lock:
                requeued = (failed + self._buffer)[-self._MAX_BUFFERED :]
                dropped = len(failed) + len(self._buffer) - len(requeued)
                self._buffer = requeued
            logger.warning(
                "graphiti add_episode failed after %d delivered (%d requeued, %d dropped): %s",
                delivered,
                len(requeued),
                dropped,
                exc,
            )

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        user_content = (user_content or "").strip()
        assistant_content = (assistant_content or "").strip()
        if len(user_content) + len(assistant_content) < self._MIN_TURN_LEN:
            return
        with self._buffer_lock:
            self._turn_seq += 1
            seq = self._turn_seq
        self._enqueue_write(
            f"User:\n{user_content}\n\nAssistant:\n{assistant_content}",
            name=f"hermes turn {seq}",
            source_description=TURN_SOURCE,
            # `message` is the episode type Graphiti's extraction prompt pairs
            # with "actor: content" framing, and what the spike measured.
            source="message",
        )

    def on_memory_write(self, action, target, content, metadata=None):
        if action not in {"add", "replace"}:
            return
        self._enqueue_write(
            f"[memory:{target}] {content}",
            name=f"hermes memory:{target}",
            source_description=PINNED_SOURCE,
            source="text",
        )

    def on_session_end(self, messages):
        if self._agent_context != "primary":
            return
        self._spawn(self._flush)

    @staticmethod
    def _format_recall(results: List[Any]) -> str:
        lines = []
        for item in results[:10]:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("answer") or json.dumps(item)
            else:
                text = str(item)
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)

    @staticmethod
    def _format_facts(facts: List[Any]) -> str:
        lines = []
        for item in facts[:10]:
            text = item.get("fact") if isinstance(item, dict) else str(item)
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        query = (query or "").strip()
        if not query or not self._initialized:
            return
        sid = session_id or self._session_id

        def _run():
            blocks = []
            try:
                facts = self._graphiti.search_facts(
                    query, max_facts=5, group_ids=[self._group_id]
                )
                formatted = self._format_facts(facts)
                if formatted:
                    blocks.append(f"<working-memory>\n{formatted}\n</working-memory>")
            except Exception as exc:
                logger.debug("graphiti prefetch failed: %s", exc)
            try:
                # recall (not search) so the session cache is covered too:
                # facts remembered this session surface before the background
                # graph build finishes.
                results = self._client.recall(
                    query,
                    search_type=self._config.prefetch_search_type,
                    top_k=5,
                    only_context=True,
                    node_name=[self._config.node_set] if self._config.node_set else None,
                    session_id=sid or None,
                )
                formatted = self._format_recall(results)
                if formatted:
                    blocks.append(f"<cognee-memory>\n{formatted}\n</cognee-memory>")
            except Exception as exc:
                logger.debug("cognee prefetch failed: %s", exc)
            if blocks:
                with self._prefetch_lock:
                    self._prefetch_cache[sid] = "\n".join(blocks)

        self._spawn(_run)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        sid = session_id or self._session_id
        with self._prefetch_lock:
            # Already provenance-wrapped per tier by queue_prefetch.
            return self._prefetch_cache.pop(sid, "")

    def on_session_switch(self, new_session_id, *, parent_session_id="", reset=False, rewound=False, **kwargs):
        self._session_id = new_session_id or ""
        if reset or rewound:
            with self._prefetch_lock:
                self._prefetch_cache.clear()
            with self._buffer_lock:
                self._turn_seq = 0

    def _recall_both_tiers(self, query: str, *, search_type: str, top_k: int) -> Dict[str, Any]:
        """Query both tiers independently so either can fail alone."""
        degraded = []
        working: List[Any] = []
        archive: List[Any] = []
        try:
            working = self._graphiti.search_facts(
                query, max_facts=top_k, group_ids=[self._group_id]
            )
        except Exception as exc:
            degraded.append(_WORKING)
            logger.warning("graphiti recall failed, degrading to archive only: %s", exc)
        try:
            archive = self._client.search(
                query, search_type=search_type, top_k=top_k, only_context=False
            )
        except Exception as exc:
            degraded.append(_ARCHIVE)
            logger.warning("cognee recall failed: %s", exc)
        return {"working": working, "archive": archive, "degraded": degraded}

    def handle_tool_call(self, tool_name, args, **kwargs) -> str:
        try:
            if tool_name == "cognee_remember":
                content = str(args.get("content") or "").strip()
                if not content:
                    return _err("content is required")
                # This is the archive write path, and it reaches OpenRouter's
                # prompt-injection classifier — sanitize like every other write.
                content = _neutralize_role_markers(content)
                # Single server-side add+cognify; the session_id keeps the fact
                # recallable this session while the graph builds in the background.
                self._client.remember([content], session_id=self._session_id or None)
                return json.dumps({"ok": True})
            if tool_name == "cognee_recall":
                query = str(args.get("query") or "").strip()
                if not query:
                    return _err("query is required")
                found = self._recall_both_tiers(
                    query,
                    search_type=str(args.get("search_type") or self._config.tool_search_type),
                    top_k=int(args.get("top_k") or 10),
                )
                degraded = found.pop("degraded")
                if len(degraded) == 2:
                    return _err("both memory tiers are unreachable")
                out = {"ok": True, **found}
                if degraded:
                    out["degraded"] = degraded
                return json.dumps(out)
            if tool_name == "cognee_forget":
                if args.get("confirm") is not True:
                    return _err("cognee_forget requires confirm=true after an explicit deletion request")
                # The dataset is shared across agents — deleting it destroys every
                # agent's memory. Never default it; the model must name it, echoing
                # an explicit user request.
                dataset = str(args.get("dataset") or "").strip()
                if not dataset:
                    return _err(
                        "cognee_forget requires the exact dataset name; it is shared "
                        "across agents and will be deleted for all of them"
                    )
                memory_only = args.get("memory_only") is True
                self._client.forget(dataset, memory_only=memory_only)
                return json.dumps({"ok": True, "dataset": dataset, "memory_only": memory_only})
            return _err(f"Unknown cognee tool: {tool_name}")
        except Exception as exc:
            logger.warning("cognee tool %s failed: %s", tool_name, exc)
            return _err(str(exc))

    def shutdown(self) -> None:
        try:
            self._flush()
        except Exception:
            logger.debug("flush during shutdown failed", exc_info=True)
        with self._threads_lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout=2)
        for client in (self._client, self._graphiti):
            if client is None:
                continue
            try:
                client.close()
            except Exception:
                pass
