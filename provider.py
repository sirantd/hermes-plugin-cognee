"""Cognee memory provider for Hermes — remote-REST, layered alongside builtin memory."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

try:
    from .client import CogneeClient, CogneeConfig
except ImportError:  # flat import during standalone unit tests
    from client import CogneeClient, CogneeConfig

logger = logging.getLogger(__name__)

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
    "description": "Delete/reset a cognee dataset. Only after an explicit user deletion request.",
    "parameters": {
        "type": "object",
        "properties": {
            "confirm": {"type": "boolean", "description": "Must be true to execute deletion."},
            "dataset": {"type": "string", "description": "Dataset name (defaults to the configured one)."},
        },
        "required": ["confirm"],
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
            {"key": "node_set", "description": "node_set partition for this agent", "default": "hermes"},
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
        self._threads = [t for t in self._threads if t.is_alive()]
        self._threads.append(thread)
        thread.start()

    def _enqueue_write(self, text: str) -> None:
        if self._agent_context != "primary" or not text or not text.strip():
            return
        flush = False
        with self._buffer_lock:
            self._buffer.append(text.strip())
            if len(self._buffer) >= self._config.add_buffer_size:
                flush = True
        if flush:
            self._spawn(self._flush)

    def _flush(self) -> None:
        with self._buffer_lock:
            pending = self._buffer
            self._buffer = []
        if not pending:
            return
        try:
            self._client.add(pending)
        except Exception as exc:  # degrade — never break the turn
            logger.warning("cognee add failed (%d records dropped): %s", len(pending), exc)

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        user_content = (user_content or "").strip()
        assistant_content = (assistant_content or "").strip()
        if len(user_content) + len(assistant_content) < self._MIN_TURN_LEN:
            return
        self._enqueue_write(f"[user]\n{user_content}\n\n[assistant]\n{assistant_content}")

    def on_memory_write(self, action, target, content, metadata=None):
        if action not in {"add", "replace"}:
            return
        self._enqueue_write(f"[memory:{target}] {content}")
