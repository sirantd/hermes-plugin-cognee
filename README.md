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
