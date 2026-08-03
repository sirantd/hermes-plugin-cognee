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
| `node_set` | `hermes_agent` |
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
- **Node- and session-scoped prefetch:** the per-turn auto-prefetch uses
  `/api/v1/recall` scoped to this agent's own `node_set` (the cognee `nodeName`
  filter) and the current `sessionId`, so it only surfaces this agent's memories —
  not unrelated documents that share the dataset — and covers the server's
  session cache. Explicit `cognee_recall` is deliberately left **dataset-wide**
  so a direct "check memory" request can reach everything in the dataset.
- **Remember via `/api/v1/remember`:** `cognee_remember` is a single server-side
  add + cognify call (graph build runs in the background on the server). The fact
  is attributed to the current session, so session-scoped prefetch can surface it
  immediately — before the background graph build finishes.
- **Forget modes:** `cognee_forget` uses `/api/v1/forget`. Default is full dataset
  deletion; with `memory_only=true` only graph nodes/edges and embeddings are
  wiped — raw records survive and the dataset can be re-cognified from scratch.
  Both modes require `confirm=true` and the exact dataset name (shared dataset).
- cognee is best-effort: a down server never breaks a turn.

## Development

```bash
git clone https://github.com/NousResearch/hermes-agent  # provides the MemoryProvider ABC
pip install -r requirements-dev.txt
HERMES_AGENT_PATH=$PWD/hermes-agent python -m pytest -v
```
