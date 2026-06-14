# cognee Memory Provider for Hermes — Design

**Date:** 2026-06-14
**Status:** Approved design, pre-implementation
**Repo:** `sirantd/hermes-plugin-cognee` (public) — local dev at `~/Projects/personal/hermes-plugin-cognee`
**Target:** Hermes Agent on `hermes.local` VM (`~/.hermes/hermes-agent`)

## Goal

Add cognee as a first-class Hermes memory provider so it is selectable via
`memory.provider: cognee` and configured through `hermes memory setup` — replacing
the current ad-hoc cognee integration (custom SessionEnd flush hooks + cognee-note
buffer). cognee runs as a remote server on `truenas.local:8000`, backing the shared
`main_dataset` knowledge graph, partitioned by `node_set`.

## Packaging & deployment

A **user-installed plugin** (not bundled — no push access to the upstream
`hermes-agent` repo). Shipped from a new public repo and installed via Hermes' own
plugin manager, which clones into `$HERMES_HOME/plugins/cognee/`.

- Develop locally at `~/Projects/personal/hermes-plugin-cognee`, push to
  `github.com/sirantd/hermes-plugin-cognee` (public).
- Deploy on the VM: `hermes plugins install sirantd/hermes-plugin-cognee`.
- Update: `hermes plugins update cognee`.
- No VM GitHub auth or deploy key needed (public repo).
- Activate: set `memory.provider: cognee` in the agent's `config.yaml`
  (or via `hermes memory setup`).

Hermes' loader (`plugins/memory/__init__.py`) discovers user-installed providers in
`$HERMES_HOME/plugins/<name>/` via the `_hermes_user_memory.<name>` synthetic
namespace, and detects them by scanning `__init__.py` for `register_memory_provider`
or `MemoryProvider`.

## Architecture

```
$HERMES_HOME/plugins/cognee/
├── plugin.yaml          # metadata + pip_dependencies + hooks
├── __init__.py          # register(ctx) + CogneeMemoryProvider(MemoryProvider)
├── client.py            # thin cognee REST client (add/cognify/search) + worker
├── cli.py               # optional: `hermes cognee status|search` convenience cmds
├── README.md
└── tests/               # pytest unit tests against a fake HTTP transport
```

Nothing in the Hermes core changes. The provider implements the
`agent.memory_provider.MemoryProvider` ABC and is registered through the
`register(ctx)` → `ctx.register_memory_provider(instance)` pattern used by bundled
providers.

### plugin.yaml

```yaml
name: cognee
version: 0.1.0
description: "cognee — shared knowledge-graph long-term memory (remote cognee server)."
pip_dependencies:
  - "httpx>=0.27"
requires_env: []
hooks:
  - on_session_end
  - on_memory_write
  - on_turn_start
```

## Component responsibilities

### `client.py` — `CogneeClient`
Thin async-capable HTTP wrapper around the cognee REST API, ported from the proven
`paperclip-plugin-cognee/src/cognee-client.ts` contract:

| Method | Endpoint | Payload |
|---|---|---|
| `add(text)` | `POST /api/v1/add` | multipart: `data` (markdown blob), `datasets`, `node_set` |
| `cognify()` | `POST /api/v1/cognify` | JSON `{datasets: [dataset], run_in_background: true}` |
| `search(query, search_type)` | `POST /api/v1/search` | JSON `{searchType, datasets: [dataset], query}` |

- Optional `Authorization: Bearer <token>` header (config; default none — LAN-open server).
- All non-2xx responses raise; the provider layer catches and degrades.
- Bounded request timeout (config, default 30s).

### `__init__.py` — `CogneeMemoryProvider(MemoryProvider)`

Implements the ABC. A single daemon **worker thread** drains two queues so the agent
turn never blocks on cognee:
- **write buffer** — accumulated markdown records pending `/add`
- **search queue** — prefetch search requests; results cached per `session_id`

| ABC method | Behaviour |
|---|---|
| `name` | `"cognee"` |
| `is_available()` | `httpx` importable **and** `base_url` configured. No network call. |
| `initialize(session_id, **kwargs)` | Build client (`base_url`, `dataset=main_dataset`, `node_set=hermes`); start worker thread. Capture `agent_context`; record platform/identity. |
| `system_prompt_block()` | One line: cognee long-term graph memory active; `cognee_search`/`cognee_remember` tools available. |
| `prefetch(query, session_id)` | Return **cached** search result for `session_id` instantly (non-blocking). |
| `queue_prefetch(query, session_id)` | Enqueue a background search for the next turn (uses `prefetch_search_type`, default `CHUNKS`). |
| `sync_turn(user, asst, ...)` | Append formatted turn record to write buffer (primary context only). |
| `on_memory_write(action, target, content, metadata)` | Mirror builtin memory-tool writes into the write buffer (primary context only). |
| `on_turn_start(turn_number, ...)` | Increment turn counter; every `cognify_every_n_turns` (default 10) enqueue a background cognify. |
| `on_session_end(messages)` | Flush write buffer → `/add`, then trigger `/cognify` (background). |
| `get_tool_schemas()` | `cognee_search`, `cognee_remember`. |
| `handle_tool_call(name, args)` | Dispatch; return JSON string. |
| `shutdown()` | Flush buffer with bounded timeout; stop worker. |
| `get_config_schema()` | Fields for `hermes memory setup` (below). |
| `save_config(values, hermes_home)` | Write non-secret fields to `config.yaml` under `memory.cognee`. |

## Data flow

**Write path (cheap, async):** `sync_turn` and `on_memory_write` append markdown
records to the write buffer. The worker flushes via `/add` when the buffer reaches
`add_buffer_size` (default 5) or at session end. Writes are gated to
`agent_context == "primary"` — subagent/cron/flush contexts are recall-only to avoid
corrupting the shared graph.

**Cognify cadence (expensive, background):** `/cognify {run_in_background: true}`
fires at `on_session_end` and every `cognify_every_n_turns` turns. Never per-turn.

**Recall path:**
- End of turn → `queue_prefetch(query)` enqueues a search; worker runs it and caches
  the result keyed by `session_id`.
- Start of next turn → `prefetch(query)` returns the cached result instantly and it
  is injected as turn context. Default search type `CHUNKS` (fast, non-LLM).
- `cognee_search(query, search_type?)` tool → on-demand search, default
  `GRAPH_COMPLETION` (rich, LLM-synthesised answer).
- `cognee_remember(text)` tool → immediate `/add` + buffer flush.

## Configuration (`get_config_schema` → `hermes memory setup`)

| key | default | secret | notes |
|---|---|---|---|
| `base_url` | `http://truenas.local:8000` | no | cognee server |
| `dataset` | `main_dataset` | no | shared graph dataset |
| `node_set` | `hermes` | no | static partition for this agent |
| `auth_token` | *(empty)* | yes → `.env` | optional bearer; LAN-open server needs none |
| `prefetch_search_type` | `CHUNKS` | no | fast recall for auto-prefetch |
| `tool_search_type` | `GRAPH_COMPLETION` | no | rich recall for `cognee_search` |
| `cognify_every_n_turns` | `10` | no | periodic graph build cadence |
| `add_buffer_size` | `5` | no | records buffered before flush |
| `request_timeout` | `30` | no | seconds |

`save_config` writes non-secrets under `memory.cognee` in `config.yaml`; `auth_token`
(if set) goes to `.env`.

## Error handling

cognee is an enhancement, never a hard dependency:
- Every client call is wrapped; failures log at warning and degrade — empty recall on
  search failure, dropped/retried-next-flush on write failure.
- A down/unreachable cognee server must never break or delay a turn (prefetch returns
  cached-or-empty; writes are async).
- The worker thread swallows and logs exceptions; it never propagates into the turn loop.
- `shutdown()` attempts a final flush with a bounded timeout, then exits cleanly.

## node_set scoping

Static `node_set: hermes` for all memory, matching the existing shared-graph
convention (one pooled memory across users/platforms). Not per-user — accepted
trade-off for a unified graph and simpler cross-context recall.

## Testing

Follow the Hermes `tests/plugins/memory/` style; pytest with a **fake httpx
transport** (mirrors `paperclip-plugin-cognee/test/cognee-client.test.ts`). Cases:
- `client`: add multipart shape (datasets + node_set), cognify payload, search payload
  + response normalisation, auth header when token set, error raises on non-2xx.
- buffering: flush at `add_buffer_size`, flush at session end.
- cognify cadence: fires every N turns and at session end; never per-turn.
- prefetch: `queue_prefetch` populates cache; `prefetch` returns cached result and is
  non-blocking; cache is per `session_id`.
- tools: `cognee_search`/`cognee_remember` dispatch and JSON return shape.
- write-gating: no writes when `agent_context != "primary"`.
- `is_available`: true with config, false without; no network.
- config: `get_config_schema` fields; `save_config` round-trip (non-secrets to yaml,
  token to env).
- graceful degradation: server errors never raise into turn loop; recall returns empty.

## Out of scope

- Per-user / per-platform node_set isolation.
- Migrating historical data already cognified by the current hook-based flow.
- Changes to the upstream `hermes-agent` repo.
- Replacing builtin file memory (it stays active; cognee layers alongside).

## Open follow-ups (post-MVP)

- Retire the existing custom SessionEnd cognee hooks once the provider is verified live.
- Optional: expose cognify status via `cli.py` (`hermes cognee status`).
