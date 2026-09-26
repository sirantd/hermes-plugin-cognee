"""Remote cognee REST client + config for the Hermes cognee memory provider."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class CogneeConfig:
    base_url: str = "http://truenas.lan:8000"
    dataset: str = "main_dataset"
    node_set: str = "hermes_agent"
    auth_token: str = ""
    prefetch_search_type: str = "CHUNKS"
    tool_search_type: str = "GRAPH_COMPLETION"
    request_timeout: float = 30.0

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> "CogneeConfig":
        data = data or {}
        known = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key in known and value not in (None, ""):
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_hermes_config(cls) -> "CogneeConfig":
        """Read non-secret config from Hermes config.yaml ``memory.cognee``,
        and the auth token from the ``COGNEE_AUTH_TOKEN`` env var."""
        import os

        data: Dict[str, Any] = {}
        try:
            from hermes_cli.config import cfg_get, load_config

            cfg = load_config()
            data = cfg_get(cfg, "memory", "cognee", default={}) or {}
        except Exception:
            logger.debug("Could not load Hermes config for cognee", exc_info=True)
        cfg = cls.from_mapping(data)
        cfg.auth_token = os.environ.get("COGNEE_AUTH_TOKEN", cfg.auth_token)
        return cfg


class CogneeClient:
    """Thin synchronous client for the remote cognee REST API."""

    def __init__(self, config: CogneeConfig, *, transport: Optional[httpx.BaseTransport] = None):
        self._config = config
        headers = {}
        if config.auth_token:
            headers["Authorization"] = f"Bearer {config.auth_token}"
        self._http = httpx.Client(
            base_url=config.base_url,
            timeout=config.request_timeout,
            headers=headers,
            transport=transport,
        )

    @property
    def config(self) -> CogneeConfig:
        return self._config

    def close(self) -> None:
        self._http.close()

    def remember(self, texts: List[str], *, session_id: Optional[str] = None) -> None:
        """Ingest and build the graph in a single server-side call (``/api/v1/remember``).

        The only write this client makes. Session turns go to the working tier
        now, so the client-side ``/api/v1/add`` + ``/api/v1/cognify`` pair this
        replaced has no remaining caller and is gone.

        With a ``session_id`` the fact also lands in the server's session cache,
        so a session-scoped ``recall`` can see it before the background graph
        build finishes.
        """
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return
        files = [
            ("data", (f"memory_{i}.md", io.BytesIO(t.encode("utf-8")), "text/markdown"))
            for i, t in enumerate(texts)
        ]
        data = {
            "datasetName": self._config.dataset,
            "node_set": self._config.node_set,
            "run_in_background": "true",  # multipart form value is a string, not a JSON bool
        }
        if session_id:
            data["session_id"] = session_id
        resp = self._http.post("/api/v1/remember", data=data, files=files)
        resp.raise_for_status()

    def search(
        self,
        query: str,
        *,
        search_type: str,
        top_k: int = 10,
        only_context: bool = False,
        node_name: Optional[List[str]] = None,
    ) -> List[Any]:
        payload: Dict[str, Any] = {
            "searchType": search_type,
            "datasets": [self._config.dataset],
            "query": query,
            "topK": top_k,
            "onlyContext": only_context,
        }
        # Scope results to specific node_set tags when requested. The cognee REST
        # API exposes this as `nodeName` (array). Used for agent-scoped prefetch;
        # omitted for explicit cognee_recall so it can reach the whole dataset.
        if node_name:
            payload["nodeName"] = node_name
        resp = self._http.post("/api/v1/search", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else [data]

    def recall(
        self,
        query: str,
        *,
        search_type: str,
        top_k: int = 10,
        only_context: bool = False,
        node_name: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> List[Any]:
        """Memory-aware search via ``/api/v1/recall``.

        Unlike ``/api/v1/search`` this also covers the server's session cache,
        so facts remembered with a ``session_id`` are recallable in the same
        session before the background graph build completes.
        """
        payload: Dict[str, Any] = {
            "searchType": search_type,
            "datasets": [self._config.dataset],
            "query": query,
            "topK": top_k,
            "onlyContext": only_context,
        }
        if node_name:
            payload["nodeName"] = node_name
        if session_id:
            payload["sessionId"] = session_id
        resp = self._http.post("/api/v1/recall", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else [data]

    def forget(self, dataset: str, *, memory_only: bool = False) -> None:
        """Delete a dataset via ``/api/v1/forget``.

        With ``memory_only`` the graph nodes/edges and vector embeddings are
        wiped but raw files and data records survive, so the dataset can be
        re-cognified from scratch.
        """
        payload = {"dataset": dataset, "memoryOnly": memory_only}
        resp = self._http.post("/api/v1/forget", json=payload)
        resp.raise_for_status()

    def list_datasets(self) -> List[Dict[str, Any]]:
        resp = self._http.get("/api/v1/datasets")
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
