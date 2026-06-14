# cognee Memory Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `hermes-plugin-cognee` — a standalone, user-installed Hermes memory provider that layers a remote cognee knowledge graph (`truenas.local:8000`, `main_dataset`, node_set `hermes`) alongside builtin file memory.

**Architecture:** A thin synchronous `httpx` REST client (`client.py`) talks to the remote cognee server. `CogneeMemoryProvider` (`provider.py`) implements the Hermes `MemoryProvider` ABC: cheap async `/add` buffering on writes, background `/cognify` on a turn/session cadence, `CHUNKS` prefetch + `GRAPH_COMPLETION` recall tool, and `on_memory_write` mirroring of builtin writes. The repo root *is* the plugin directory — `hermes plugins install sirantd/hermes-plugin-cognee` clones it into `$HERMES_HOME/plugins/cognee/`.

**Tech Stack:** Python 3.11, `httpx`, `pytest` (+ `httpx.MockTransport`). The Hermes `MemoryProvider` ABC comes from a sibling `NousResearch/hermes-agent` checkout on `sys.path` (public, imports cleanly standalone).

---

## Verified cognee REST API (live, `truenas.local:8000`, no auth on LAN)

- `POST /api/v1/add` — **multipart**: `data` (file array; one markdown blob per record), `datasetName`, `node_set`, `run_in_background`.
- `POST /api/v1/cognify` — **JSON**: `{datasets: [name], runInBackground: true}` (camelCase).
- `POST /api/v1/search` — **JSON**: `{searchType, datasets: [name], query, topK, onlyContext}` (camelCase). `searchType` ∈ `{SUMMARIES, CHUNKS, RAG_COMPLETION, GRAPH_COMPLETION, GRAPH_COMPLETION_COT, …}`.
- `GET /api/v1/datasets` — array of `{id, name, …}`; used to resolve a dataset name → id.
- `DELETE /api/v1/datasets/{dataset_id}` — delete/reset a dataset (used by `cognee_forget`).

## File Structure (repo root = plugin dir)

```
hermes-plugin-cognee/
├── plugin.yaml          # manifest: name/version/description/pip_dependencies/hooks
├── __init__.py          # register(ctx) + re-export CogneeMemoryProvider
├── provider.py          # CogneeMemoryProvider(MemoryProvider) — lifecycle, tools, threading
├── client.py            # CogneeConfig + CogneeClient (httpx REST)
├── cli.py               # register_cli → `hermes cognee status|recall`
├── README.md            # install + config
├── conftest.py          # put a hermes-agent checkout on sys.path for tests
├── requirements-dev.txt # pytest, httpx, jiter
├── .gitignore
├── tests/
│   ├── test_client.py   # CogneeClient against httpx.MockTransport (no hermes dep)
│   └── test_provider.py # CogneeMemoryProvider with an injected FakeClient (importorskip ABC)
└── .github/workflows/ci.yml
```

**Import shim:** `__init__.py` and `provider.py` must import siblings with a fallback so the module works both packaged (`_hermes_user_memory.cognee.client`) and flat (test `import client`):

```python
try:
    from .client import CogneeConfig, CogneeClient
except ImportError:  # flat import during standalone unit tests
    from client import CogneeConfig, CogneeClient
```

---

## Task 1: Repo scaffold + manifest + dev harness

**Files:**
- Create: `plugin.yaml`, `.gitignore`, `requirements-dev.txt`, `conftest.py`
- Test: `tests/test_client.py` (manifest smoke test for now)

- [ ] **Step 1: Write the failing test**

`tests/test_client.py`:
```python
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parent.parent


def test_plugin_manifest_valid():
    manifest = yaml.safe_load((REPO / "plugin.yaml").read_text())
    assert manifest["name"] == "cognee"
    assert "httpx" in " ".join(manifest["pip_dependencies"])
    assert "on_session_end" in manifest["hooks"]
    assert "on_memory_write" in manifest["hooks"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_client.py::test_plugin_manifest_valid -v`
Expected: FAIL — `FileNotFoundError: plugin.yaml`.

- [ ] **Step 3: Create the scaffold files**

`plugin.yaml`:
```yaml
name: cognee
version: 0.1.0
description: "cognee — shared knowledge-graph long-term memory (remote cognee server) for Hermes."
pip_dependencies:
  - "httpx>=0.27"
hooks:
  - on_session_end
  - on_memory_write
  - on_turn_start
```

`.gitignore`:
```
__pycache__/
*.pyc
.pytest_cache/
.venv/
hermes-agent/
```

`requirements-dev.txt`:
```
pytest>=8
httpx>=0.27
PyYAML>=6
jiter>=0.8
```

`conftest.py`:
```python
"""Make the Hermes MemoryProvider ABC importable for provider tests.

Looks for a NousResearch/hermes-agent checkout via $HERMES_AGENT_PATH or a
sibling ../hermes-agent directory and puts it on sys.path. Provider tests
`pytest.importorskip("agent.memory_provider")`, so they skip cleanly if no
checkout is present (client tests still run).
"""
import os
import sys
from pathlib import Path

_candidates = [
    os.environ.get("HERMES_AGENT_PATH"),
    str(Path(__file__).resolve().parent / "hermes-agent"),
    str(Path(__file__).resolve().parent.parent / "hermes-agent"),
]
for _c in _candidates:
    if _c and Path(_c).is_dir() and _c not in sys.path:
        sys.path.insert(0, _c)
        break
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_client.py::test_plugin_manifest_valid -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugin.yaml .gitignore requirements-dev.txt conftest.py tests/test_client.py
git commit -m "chore: scaffold cognee plugin repo + manifest smoke test"
```

---

## Task 2: `CogneeConfig` dataclass + config resolution

**Files:**
- Create: `client.py`
- Test: `tests/test_client.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_client.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_client.py -k config -v`
Expected: FAIL — `ImportError: cannot import name 'CogneeConfig'`.

- [ ] **Step 3: Write minimal implementation**

`client.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_client.py -k config -v`
Expected: PASS (both config tests).

- [ ] **Step 5: Commit**

```bash
git add client.py tests/test_client.py
git commit -m "feat: add CogneeConfig with defaults and config resolution"
```

---

## Task 3: `CogneeClient.add` (multipart ingest)

**Files:**
- Modify: `client.py`
- Test: `tests/test_client.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_client.py`:
```python
import httpx
from client import CogneeClient


def _client(handler):
    transport = httpx.MockTransport(handler)
    cfg = CogneeConfig(base_url="http://test")
    return CogneeClient(cfg, transport=transport)


def test_add_posts_multipart_with_dataset_and_node_set():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["content_type"] = request.headers.get("content-type", "")
        seen["body"] = request.content
        return httpx.Response(200, json={"status": "ok"})

    client = _client(handler)
    client.add(["fact one", "fact two"])

    assert seen["url"] == "http://test/api/v1/add"
    assert "multipart/form-data" in seen["content_type"]
    body = seen["body"]
    assert b"main_dataset" in body          # datasetName field
    assert b'name="node_set"' in body
    assert b"hermes" in body
    assert b"fact one" in body and b"fact two" in body


def test_add_empty_list_is_noop():
    def handler(request):
        raise AssertionError("should not call server for empty add")

    _client(handler).add([])  # no exception, no request
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_client.py -k add -v`
Expected: FAIL — `ImportError: cannot import name 'CogneeClient'`.

- [ ] **Step 3: Write minimal implementation**

Append to `client.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_client.py -k add -v`
Expected: PASS (both add tests).

- [ ] **Step 5: Commit**

```bash
git add client.py tests/test_client.py
git commit -m "feat: add CogneeClient.add multipart ingest"
```

---

## Task 4: `CogneeClient.search` (+ response normalization)

**Files:**
- Modify: `client.py`
- Test: `tests/test_client.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_client.py`:
```python
import json


def test_search_posts_camelcase_json_and_returns_list():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=[{"text": "a"}, {"text": "b"}])

    client = _client(handler)
    out = client.search("who am i", search_type="GRAPH_COMPLETION", top_k=3, only_context=True)

    assert seen["url"] == "http://test/api/v1/search"
    assert seen["payload"] == {
        "searchType": "GRAPH_COMPLETION",
        "datasets": ["main_dataset"],
        "query": "who am i",
        "topK": 3,
        "onlyContext": True,
    }
    assert out == [{"text": "a"}, {"text": "b"}]


def test_search_wraps_non_list_response_in_list():
    def handler(request):
        return httpx.Response(200, json={"answer": "42"})

    out = _client(handler).search("q", search_type="CHUNKS")
    assert out == [{"answer": "42"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_client.py -k search -v`
Expected: FAIL — `AttributeError: 'CogneeClient' object has no attribute 'search'`.

- [ ] **Step 3: Write minimal implementation**

Append to `CogneeClient` in `client.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_client.py -k search -v`
Expected: PASS (both search tests).

- [ ] **Step 5: Commit**

```bash
git add client.py tests/test_client.py
git commit -m "feat: add CogneeClient.search with response normalization"
```

---

## Task 5: `CogneeClient.cognify` (background graph build)

**Files:**
- Modify: `client.py`
- Test: `tests/test_client.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_client.py`:
```python
def test_cognify_posts_camelcase_background_json():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "started"})

    _client(handler).cognify()

    assert seen["url"] == "http://test/api/v1/cognify"
    assert seen["payload"] == {"datasets": ["main_dataset"], "runInBackground": True}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_client.py -k cognify -v`
Expected: FAIL — no attribute `cognify`.

- [ ] **Step 3: Write minimal implementation**

Append to `CogneeClient` in `client.py`:
```python
    def cognify(self) -> None:
        payload = {"datasets": [self._config.dataset], "runInBackground": True}
        resp = self._http.post("/api/v1/cognify", json=payload)
        resp.raise_for_status()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_client.py -k cognify -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add client.py tests/test_client.py
git commit -m "feat: add CogneeClient.cognify background graph build"
```

---

## Task 6: `CogneeClient` dataset resolve + delete (for forget)

**Files:**
- Modify: `client.py`
- Test: `tests/test_client.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_client.py`:
```python
def test_delete_dataset_resolves_name_to_id():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/api/v1/datasets":
            return httpx.Response(200, json=[
                {"id": "ds-1", "name": "other"},
                {"id": "ds-2", "name": "main_dataset"},
            ])
        if request.method == "DELETE" and request.url.path == "/api/v1/datasets/ds-2":
            return httpx.Response(200, json={"deleted": True})
        return httpx.Response(404)

    deleted = _client(handler).delete_dataset_by_name("main_dataset")
    assert deleted is True
    assert ("GET", "/api/v1/datasets") in calls
    assert ("DELETE", "/api/v1/datasets/ds-2") in calls


def test_delete_dataset_missing_name_returns_false():
    def handler(request):
        return httpx.Response(200, json=[{"id": "x", "name": "nope"}])

    assert _client(handler).delete_dataset_by_name("absent") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_client.py -k delete_dataset -v`
Expected: FAIL — no attribute `delete_dataset_by_name`.

- [ ] **Step 3: Write minimal implementation**

Append to `CogneeClient` in `client.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_client.py -k delete_dataset -v`
Expected: PASS (both tests). Then full client suite: `python -m pytest tests/test_client.py -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add client.py tests/test_client.py
git commit -m "feat: add dataset resolve + delete for cognee_forget"
```

---

## Task 7: Provider skeleton — identity, availability, config, system prompt

**Files:**
- Create: `provider.py`
- Test: `tests/test_provider.py`

> Provider tests inject a `FakeClient` (records calls) — no httpx, deterministic. They `importorskip` the real ABC. Ensure a hermes-agent checkout is on `sys.path` (see Task 12 dev-setup / conftest) or these tests skip.

- [ ] **Step 1: Write the failing test**

`tests/test_provider.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_provider.py -k "name_and_availability or config_schema or system_prompt" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'provider'` (or skip if no hermes-agent checkout — set one up per Task 12 first).

- [ ] **Step 3: Write minimal implementation**

`provider.py`:
```python
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
```

> NOTE: `set_config_value` is the assumed Hermes config writer. During implementation, confirm the exact non-secret writer in `hermes_cli/config.py` (grep for `def save_` / `def set_config`); if the name differs, adjust this method only. Unit tests stub these helpers, so the suite is unaffected.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_provider.py -k "name_and_availability or config_schema or system_prompt" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add provider.py tests/test_provider.py
git commit -m "feat: add CogneeMemoryProvider skeleton (identity, config, prompt)"
```

---

## Task 8: Write path — buffered add + agent_context gating

**Files:**
- Modify: `provider.py`
- Test: `tests/test_provider.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_provider.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_provider.py -k "buffers or mirrors or non_primary or remove_action" -v`
Expected: FAIL — `sync_turn` is the ABC no-op, so nothing buffers.

- [ ] **Step 3: Write minimal implementation**

Append to `CogneeMemoryProvider` in `provider.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_provider.py -k "buffers or mirrors or non_primary or remove_action" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add provider.py tests/test_provider.py
git commit -m "feat: buffered write path with agent_context gating"
```

---

## Task 9: Cognify cadence — per-N-turns + session end

**Files:**
- Modify: `provider.py`
- Test: `tests/test_provider.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_provider.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_provider.py -k "every_n_turns or session_end" -v`
Expected: FAIL — `on_turn_start`/`on_session_end` are ABC no-ops.

- [ ] **Step 3: Write minimal implementation**

Append to `CogneeMemoryProvider` in `provider.py`:
```python
    def _cognify(self) -> None:
        try:
            self._client.cognify()
        except Exception as exc:
            logger.warning("cognee cognify failed: %s", exc)

    def on_turn_start(self, turn_number, message, **kwargs):
        if self._agent_context != "primary":
            return
        self._turn_counter += 1
        if self._turn_counter % max(1, self._config.cognify_every_n_turns) == 0:
            self._spawn(self._cognify)

    def on_session_end(self, messages):
        if self._agent_context != "primary":
            return

        def _finalize():
            self._flush()
            self._cognify()

        self._spawn(_finalize)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_provider.py -k "every_n_turns or session_end" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add provider.py tests/test_provider.py
git commit -m "feat: cognify cadence (per-N-turns + session end)"
```

---

## Task 10: Recall — background prefetch cache + session switch

**Files:**
- Modify: `provider.py`
- Test: `tests/test_provider.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_provider.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_provider.py -k "prefetch or session_switch" -v`
Expected: FAIL — `queue_prefetch`/`prefetch` are ABC no-ops returning "".

- [ ] **Step 3: Write minimal implementation**

Append to `CogneeMemoryProvider` in `provider.py`:
```python
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
                results = self._client.search(
                    query,
                    search_type=self._config.prefetch_search_type,
                    top_k=5,
                    only_context=True,
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_provider.py -k "prefetch or session_switch" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add provider.py tests/test_provider.py
git commit -m "feat: background prefetch cache + session-switch handling"
```

---

## Task 11: Tools — recall / remember / forget dispatch + graceful degradation

**Files:**
- Modify: `provider.py`
- Test: `tests/test_provider.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_provider.py`:
```python
def test_tool_schemas_expose_three_tools():
    provider, _ = make_provider()
    names = {s["name"] for s in provider.get_tool_schemas()}
    assert names == {"cognee_remember", "cognee_recall", "cognee_forget"}


def test_remember_tool_adds_and_flushes_immediately():
    provider, fake = make_provider(add_buffer_size=99)
    out = json.loads(provider.handle_tool_call("cognee_remember", {"content": "I prefer dark mode"}))
    assert out["ok"] is True
    assert fake.added and any("dark mode" in r for r in fake.added[0])


def test_recall_tool_uses_tool_search_type():
    provider, fake = make_provider()
    fake.search_return = [{"text": "found it"}]
    out = json.loads(provider.handle_tool_call("cognee_recall", {"query": "prefs", "top_k": 4}))
    assert out["ok"] is True
    assert fake.searched[0][1] == "GRAPH_COMPLETION"
    assert fake.searched[0][2] == 4


def test_forget_requires_confirm():
    provider, fake = make_provider()
    out = json.loads(provider.handle_tool_call("cognee_forget", {"confirm": False}))
    assert "error" in out
    assert fake.deleted == []


def test_forget_with_confirm_deletes_dataset():
    provider, fake = make_provider()
    out = json.loads(provider.handle_tool_call("cognee_forget", {"confirm": True}))
    assert out["ok"] is True
    assert fake.deleted == ["main_dataset"]


def test_unknown_tool_returns_error():
    provider, _ = make_provider()
    out = json.loads(provider.handle_tool_call("cognee_bogus", {}))
    assert "error" in out


def test_remember_tool_degrades_on_client_error():
    provider, fake = make_provider(add_buffer_size=99)

    def boom(texts):
        raise RuntimeError("server down")

    fake.add = boom
    out = json.loads(provider.handle_tool_call("cognee_remember", {"content": "x meaningful"}))
    assert "error" in out  # error surfaced to model, no exception raised
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_provider.py -k "tool" -v`
Expected: FAIL — `get_tool_schemas` returns `[]`, `handle_tool_call` raises `NotImplementedError`.

- [ ] **Step 3: Write minimal implementation**

Append to `CogneeMemoryProvider` in `provider.py`:
```python
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [REMEMBER_SCHEMA, RECALL_SCHEMA, FORGET_SCHEMA]

    def handle_tool_call(self, tool_name, args, **kwargs) -> str:
        try:
            if tool_name == "cognee_remember":
                content = str(args.get("content") or "").strip()
                if not content:
                    return _err("content is required")
                self._client.add([content])
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
                dataset = str(args.get("dataset") or self._config.dataset)
                deleted = self._client.delete_dataset_by_name(dataset)
                return json.dumps({"ok": True, "deleted": deleted, "dataset": dataset})
            return _err(f"Unknown cognee tool: {tool_name}")
        except Exception as exc:
            logger.warning("cognee tool %s failed: %s", tool_name, exc)
            return _err(str(exc))

    def shutdown(self) -> None:
        try:
            self._flush()
        except Exception:
            logger.debug("flush during shutdown failed", exc_info=True)
        for thread in list(self._threads):
            thread.join(timeout=2)
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_provider.py -v`
Expected: PASS (all provider tests).

- [ ] **Step 5: Commit**

```bash
git add provider.py tests/test_provider.py
git commit -m "feat: cognee tools (recall/remember/forget) + graceful degradation"
```

---

## Task 12: Registration, CLI, README, CI, dev-setup

**Files:**
- Create: `__init__.py`, `cli.py`, `README.md`, `.github/workflows/ci.yml`
- Test: `tests/test_provider.py` (registration test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_provider.py`:
```python
def test_register_registers_provider():
    import importlib
    pkg = importlib.import_module("__init__") if "__init__" in sys.modules else None
    # import the package __init__ as a flat module
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_provider.py::test_register_registers_provider -v`
Expected: FAIL — `__init__.py` has no `register`.

- [ ] **Step 3: Write the implementation files**

`__init__.py`:
```python
"""cognee memory provider plugin for Hermes (standalone, user-installed)."""

try:
    from .provider import CogneeMemoryProvider
except ImportError:  # flat import during standalone unit tests
    from provider import CogneeMemoryProvider

__all__ = ["CogneeMemoryProvider", "register"]


def register(ctx) -> None:
    """Hermes plugin entry point — register the cognee memory provider."""
    ctx.register_memory_provider(CogneeMemoryProvider())
```

`cli.py`:
```python
"""CLI helpers for the cognee memory provider: `hermes cognee ...`."""

from __future__ import annotations


def register_cli(subparser) -> None:
    sub = subparser.add_parser("cognee", help="cognee memory provider utilities")
    actions = sub.add_subparsers(dest="cognee_action")

    status = actions.add_parser("status", help="Show cognee server reachability + config")
    status.set_defaults(func=_cmd_status)

    recall = actions.add_parser("recall", help="Run a one-off cognee recall query")
    recall.add_argument("query")
    recall.set_defaults(func=_cmd_recall)


def _build_client():
    try:
        from .client import CogneeClient, CogneeConfig
    except ImportError:
        from client import CogneeClient, CogneeConfig
    cfg = CogneeConfig.from_hermes_config()
    return CogneeClient(cfg), cfg


def _cmd_status(args=None) -> None:
    client, cfg = _build_client()
    try:
        datasets = client.list_datasets()
        print(f"cognee OK @ {cfg.base_url} — dataset={cfg.dataset} node_set={cfg.node_set} "
              f"({len(datasets)} datasets visible)")
    except Exception as exc:
        print(f"cognee UNREACHABLE @ {cfg.base_url}: {exc}")
    finally:
        client.close()


def _cmd_recall(args) -> None:
    client, cfg = _build_client()
    try:
        results = client.search(args.query, search_type=cfg.tool_search_type, top_k=10)
        for r in results:
            print("-", r)
    except Exception as exc:
        print(f"recall failed: {exc}")
    finally:
        client.close()
```

`README.md`:
```markdown
# hermes-plugin-cognee

A standalone [Hermes](https://github.com/NousResearch/hermes-agent) memory provider
that layers a **remote cognee knowledge graph** alongside Hermes' builtin file memory.

In-tree memory providers are closed upstream (`CONTRIBUTING.md`), so cognee ships as a
user-installed plugin.

## Install (on the Hermes host)

```bash
hermes plugins install sirantd/hermes-plugin-cognee
hermes config set memory.provider cognee
# optional guided config:
hermes memory setup
```

Update later with `hermes plugins update cognee`.

## Configuration

Non-secrets live under `memory.cognee` in `config.yaml`; the optional bearer token is
`COGNEE_AUTH_TOKEN` in `.env`.

| key | default |
|---|---|
| `base_url` | `http://truenas.local:8000` |
| `dataset` | `main_dataset` |
| `node_set` | `hermes` |
| `prefetch_search_type` | `CHUNKS` |
| `tool_search_type` | `GRAPH_COMPLETION` |
| `cognify_every_n_turns` | `10` |
| `add_buffer_size` | `5` |

## Behaviour

- Writes (turns + mirrored builtin `memory` writes) are buffered and flushed to
  `/api/v1/add`; the graph is rebuilt via background `/api/v1/cognify` every
  `cognify_every_n_turns` and at session end.
- `prefetch` injects fast `CHUNKS` recall each turn; tools `cognee_recall`,
  `cognee_remember`, `cognee_forget` are exposed to the model.
- cognee is best-effort: a down server never breaks a turn.

## Development

```bash
git clone https://github.com/NousResearch/hermes-agent  # provides the MemoryProvider ABC
pip install -r requirements-dev.txt
HERMES_AGENT_PATH=$PWD/hermes-agent python -m pytest -v
```
```

`.github/workflows/ci.yml`:
```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Clone hermes-agent (provides MemoryProvider ABC)
        run: git clone --depth 1 https://github.com/NousResearch/hermes-agent hermes-agent
      - run: pip install -r requirements-dev.txt
      - run: HERMES_AGENT_PATH=$PWD/hermes-agent python -m pytest -v
```

- [ ] **Step 4: Run test to verify it passes**

Set up the dev dependency once, then run the full suite:
```bash
git clone --depth 1 https://github.com/NousResearch/hermes-agent hermes-agent
pip install -r requirements-dev.txt
HERMES_AGENT_PATH=$PWD/hermes-agent python -m pytest -v
```
Expected: ALL tests PASS (client + provider + registration).

- [ ] **Step 5: Commit**

```bash
git add __init__.py cli.py README.md .github/workflows/ci.yml tests/test_provider.py
git commit -m "feat: registration entry point, CLI, README, CI"
```

---

## Task 13: Publish + deploy to the Hermes VM

**Files:** none (ops task)

- [ ] **Step 1: Create the public GitHub repo and push**

```bash
cd ~/Projects/personal/hermes-plugin-cognee
gh repo create sirantd/hermes-plugin-cognee --public --source . --remote origin --push
```

- [ ] **Step 2: Confirm CI is green**

Run: `gh run watch` (or `gh run list -L 1`). Expected: the `ci` workflow passes.

- [ ] **Step 3: Install on the Hermes VM**

```bash
ssh hermes '~/.local/bin/hermes plugins install sirantd/hermes-plugin-cognee && \
            ~/.local/bin/hermes config set memory.provider cognee'
```
Expected: plugin cloned into `~/.hermes/plugins/cognee/`; `hermes plugins list` shows `cognee`.

- [ ] **Step 4: Verify reachability + live recall from the VM**

```bash
ssh hermes '~/.local/bin/hermes cognee status'
```
Expected: `cognee OK @ http://truenas.local:8000 — dataset=main_dataset node_set=hermes (N datasets visible)`.

- [ ] **Step 5: Smoke-test a real session, then retire the old hooks**

Run a short gateway/CLI session, confirm via `hermes cognee recall "<something you just told it>"` that the turn was captured, then remove the legacy SessionEnd cognee flush hook + cognee-note buffer (out of scope to script here — do it once parity is confirmed). Commit any config cleanup.

---

## Self-Review Notes

- **Spec coverage:** packaging/deploy (T1, T12, T13), config + `hermes memory setup` (T2, T7), REST client add/cognify/search/delete (T3–T6), write path + `on_memory_write` mirroring + agent_context gating (T8), cognify cadence (T9), prefetch + tools (T10, T11), error degradation (T8/T9/T11 via try/except + dedicated tool-degrade test), node_set static `hermes` (T2 default + T3 assertion), testing harness via discovery-path ABC + fake client (T7+). Divergences from PR #26179 (remote REST, alongside-builtin, on_memory_write present) are realised in T3–T6 (httpx client) and T7–T11 (provider).
- **Out-of-scope items** (RLIMIT/WAL/instructor, per-user node_set, historical migration) are intentionally absent.
- **Type consistency:** `CogneeClient.add(texts)`, `.search(query, *, search_type, top_k, only_context)`, `.cognify()`, `.delete_dataset_by_name(name)`, `.list_datasets()`, `.close()` are used identically in `FakeClient` and `provider.py`. Config keys match between `CogneeConfig` fields, `get_config_schema`, and `save_config`.
- **Implementation-time confirmations** (flagged inline, do not block the plan): exact non-secret config writer name in `hermes_cli/config.py` (Task 7 note); `cognee_forget` data-id-level delete endpoint if per-item forget is later wanted (currently dataset-level only).
