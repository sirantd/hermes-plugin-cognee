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
    base_url: str = "http://truenas.local:8000"
    dataset: str = "main_dataset"
    node_set: str = "hermes"
    auth_token: str = ""
    prefetch_search_type: str = "CHUNKS"
    tool_search_type: str = "GRAPH_COMPLETION"
    cognify_every_n_turns: int = 10
    add_buffer_size: int = 5
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

    def add(self, texts: List[str]) -> None:
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
            "run_in_background": "false",
        }
        resp = self._http.post("/api/v1/add", data=data, files=files)
        resp.raise_for_status()

    def search(
        self,
        query: str,
        *,
        search_type: str,
        top_k: int = 10,
        only_context: bool = False,
    ) -> List[Any]:
        payload = {
            "searchType": search_type,
            "datasets": [self._config.dataset],
            "query": query,
            "topK": top_k,
            "onlyContext": only_context,
        }
        resp = self._http.post("/api/v1/search", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else [data]

    def cognify(self) -> None:
        payload = {"datasets": [self._config.dataset], "runInBackground": True}
        resp = self._http.post("/api/v1/cognify", json=payload)
        resp.raise_for_status()

    def list_datasets(self) -> List[Dict[str, Any]]:
        resp = self._http.get("/api/v1/datasets")
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    def delete_dataset_by_name(self, name: str) -> bool:
        dataset_id = None
        for ds in self.list_datasets():
            if isinstance(ds, dict) and ds.get("name") == name:
                dataset_id = ds.get("id")
                break
        if not dataset_id:
            return False
        resp = self._http.delete(f"/api/v1/datasets/{dataset_id}")
        resp.raise_for_status()
        return True
