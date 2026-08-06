"""Cognee memory provider for Hermes — remote-REST, layered alongside builtin memory."""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

try:
    from .client import CogneeClient, CogneeConfig
except ImportError:  # flat import during standalone unit tests
    from client import CogneeClient, CogneeConfig

logger = logging.getLogger(__name__)

# The upstream prompt-injection classifier rejects any request whose text holds a
# delimiter-wrapped role marker — `[assistant]`, `<system>`, `[ASSISTANT]` — no
# matter what surrounds it. cognee's cognify step is such a request, so an
# unsanitized transcript is accepted by /add and then fails every extraction
# call, stalling the pipeline with no error on the write path. Colon-style
# labels ("Assistant:") are not flagged, so rewrite rather than delete: the
# turn structure is worth keeping in the graph.
_ROLE_MARKER_RE = re.compile(r"[\[<]\s*(user|assistant|system)\s*[\]>]", re.IGNORECASE)


def _neutralize_role_markers(text: str) -> str:
    """Rewrite `[assistant]`-style role markers to a colon form the classifier accepts."""
    return _ROLE_MARKER_RE.sub(lambda m: f"{m.group(1).capitalize()}:", text)


REMEMBER_SCHEMA = {
    "name": "cognee_remember",
    "description": "Store a durable fact, preference, or note in the cognee knowledge-graph memory.",
    "parameters": {
        "type": "object",
        "properties": {"content": {"type": "string", "description": "Text to remember."}},
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "cognee_recall",
    "description": "Recall relevant context from cognee long-term memory via semantic/graph search.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language memory query."},
            "search_type": {"type": "string", "description": "Optional cognee SearchType override."},
            "top_k": {"type": "integer", "description": "Max results (default 10)."},
        },
        "required": ["query"],
    },
}

FORGET_SCHEMA = {
    "name": "cognee_forget",
    "description": (
        "Delete a cognee dataset. The configured dataset is shared by "
        "multiple agents (all node_set partitions), so this wipes everyone's "
        "memory, not just this agent's. Only after an explicit user deletion "
        "request that names the dataset. With memory_only=true only the graph "
        "and embeddings are wiped; raw records survive and can be re-cognified."
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
    def __init__(self, config: CogneeConfig | None = None, client: CogneeClient | None = None):
        self._config = config
        self._client = client
        self._session_id = ""
        self._agent_context = "primary"
        self._turn_counter = 0
        self._buffer: List[str] = []
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
        cfg = self._config or CogneeConfig.from_hermes_config()
        return bool(cfg.base_url)

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        if self._config is None:
            self._config = CogneeConfig.from_hermes_config()
        if self._client is None:
            self._client = CogneeClient(self._config)
        self._session_id = session_id or ""
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._initialized = True

    def system_prompt_block(self) -> str:
        return (
            "# cognee Memory\n"
            "A shared cognee knowledge-graph memory is active **alongside** your builtin "
            "file memory (both are available; neither replaces the other).\n"
            "Use `cognee_remember` to store durable facts, `cognee_recall` to search "
            "long-term memory, and `cognee_forget` only after an explicit deletion request."
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [REMEMBER_SCHEMA, RECALL_SCHEMA, FORGET_SCHEMA]

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "base_url", "description": "cognee server base URL", "default": "http://truenas.local:8000"},
            {"key": "dataset", "description": "cognee dataset name", "default": "main_dataset"},
            {"key": "node_set", "description": "node_set partition for this agent", "default": "hermes_agent"},
            {"key": "auth_token", "description": "Optional bearer token", "secret": True, "env_var": "COGNEE_AUTH_TOKEN"},
            {"key": "prefetch_search_type", "description": "SearchType for auto-prefetch", "default": "CHUNKS"},
            {"key": "tool_search_type", "description": "SearchType for cognee_recall", "default": "GRAPH_COMPLETION"},
            {"key": "cognify_every_n_turns", "description": "Background cognify cadence", "default": 10},
            {"key": "add_buffer_size", "description": "Records buffered before flush", "default": 5},
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

    def _enqueue_write(self, text: str) -> None:
        if self._agent_context != "primary" or not text or not text.strip():
            return
        # Single choke point for every write, so no caller can reintroduce a
        # marker — including markers quoted inside the turn content itself.
        text = _neutralize_role_markers(text)
        flush = False
        with self._buffer_lock:
            self._buffer.append(text.strip())
            if len(self._buffer) >= self._config.add_buffer_size:
                flush = True
        if flush:
            self._spawn(self._flush)

    # Requeued records survive transient failures, but never let an extended
    # outage grow the buffer without bound.
    _MAX_BUFFERED = 100

    def _flush(self) -> None:
        with self._buffer_lock:
            pending = self._buffer
            self._buffer = []
        if not pending:
            return
        try:
            self._client.add(pending)
        except Exception as exc:  # degrade — never break the turn
            with self._buffer_lock:
                requeued = (pending + self._buffer)[-self._MAX_BUFFERED :]
                dropped = len(pending) + len(self._buffer) - len(requeued)
                self._buffer = requeued
            logger.warning(
                "cognee add failed (%d records requeued, %d dropped): %s",
                len(requeued),
                dropped,
                exc,
            )

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        user_content = (user_content or "").strip()
        assistant_content = (assistant_content or "").strip()
        if len(user_content) + len(assistant_content) < self._MIN_TURN_LEN:
            return
        self._enqueue_write(f"User:\n{user_content}\n\nAssistant:\n{assistant_content}")

    def on_memory_write(self, action, target, content, metadata=None):
        if action not in {"add", "replace"}:
            return
        self._enqueue_write(f"[memory:{target}] {content}")

    def _cognify(self) -> None:
        try:
            self._client.cognify()
        except Exception as exc:
            logger.warning("cognee cognify failed: %s", exc)

    def on_turn_start(self, turn_number, message, **kwargs):
        if self._agent_context != "primary":
            return
        with self._threads_lock:
            self._turn_counter += 1
            due = self._turn_counter % max(1, self._config.cognify_every_n_turns) == 0
        if due:
            self._spawn(self._cognify)

    def on_session_end(self, messages):
        if self._agent_context != "primary":
            return

        def _finalize():
            self._flush()
            self._cognify()

        self._spawn(_finalize)

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

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        query = (query or "").strip()
        if not query or not self._initialized:
            return
        sid = session_id or self._session_id

        def _run():
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
                    with self._prefetch_lock:
                        self._prefetch_cache[sid] = formatted
            except Exception as exc:
                logger.debug("cognee prefetch failed: %s", exc)

        self._spawn(_run)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        sid = session_id or self._session_id
        with self._prefetch_lock:
            result = self._prefetch_cache.pop(sid, "")
        if not result:
            return ""
        return f"<cognee-memory>\n{result}\n</cognee-memory>"

    def on_session_switch(self, new_session_id, *, parent_session_id="", reset=False, rewound=False, **kwargs):
        self._session_id = new_session_id or ""
        if reset or rewound:
            with self._prefetch_lock:
                self._prefetch_cache.clear()

    def handle_tool_call(self, tool_name, args, **kwargs) -> str:
        try:
            if tool_name == "cognee_remember":
                content = str(args.get("content") or "").strip()
                if not content:
                    return _err("content is required")
                # Single server-side add+cognify; the session_id keeps the fact
                # recallable this session while the graph builds in the background.
                self._client.remember([content], session_id=self._session_id or None)
                return json.dumps({"ok": True})
            if tool_name == "cognee_recall":
                query = str(args.get("query") or "").strip()
                if not query:
                    return _err("query is required")
                results = self._client.search(
                    query,
                    search_type=str(args.get("search_type") or self._config.tool_search_type),
                    top_k=int(args.get("top_k") or 10),
                    only_context=False,
                )
                return json.dumps({"ok": True, "results": results})
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
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
