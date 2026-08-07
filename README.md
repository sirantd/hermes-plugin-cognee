# hermes-plugin-cognee

A standalone [Hermes](https://github.com/NousResearch/hermes-agent) memory provider
that layers **two-tier long-term memory** alongside Hermes' builtin file memory.

In-tree memory providers are closed upstream (`CONTRIBUTING.md`), so this ships as a
user-installed plugin.

| tier | backend | holds | lifecycle |
|---|---|---|---|
| **working** | Graphiti (MCP) on FalkorDB | session turns + mirrored builtin `memory` writes, as bi-temporal **episodes** | contradicted facts self-invalidate; retention ages episodes out |
| **archive** | remote cognee knowledge graph | documents, OCR, notes, explicitly remembered facts | permanent |

Recall queries both and labels results by provenance. A working-tier outage degrades
recall to archive-only rather than failing.

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
`COGNEE_AUTH_TOKEN` in `.env`. Working-tier keys carry a `graphiti_` prefix so the two
tiers' `base_url`s don't collide.

| key | default |
|---|---|
| `base_url` | `http://truenas.local:8000` |
| `dataset` | `main_dataset` |
| `node_set` | `hermes_agent` |
| `prefetch_search_type` | `CHUNKS` |
| `tool_search_type` | `GRAPH_COMPLETION` |
| `graphiti_base_url` | `http://truenas.local:8001` |
| `graphiti_group_id` | *(derived from `node_set`)* |
| `graphiti_episode_buffer_size` | `1` |

**Partitioning.** cognee's `node_set` maps onto Graphiti's `group_id`
(`hermes_agent` → `hermes`; `homelab`/`paperclip`/`meta` unchanged). The mapping is
physical — one FalkorDB graph per group — so a wrong `group_id` is a *different graph*,
not a filter mistake. An unmapped `node_set` passes through unchanged rather than
falling back to a default.

⚠️ **The Graphiti server validates the `Host` header** and rejects anything outside its
allowlist with a bare `421 Misdirected Request`. Whatever hostname `graphiti_base_url`
dials must be in the server's `graphiti_mcp_allowed_hosts`.

## Behaviour

### Writes

- **Turn capture and mirrored builtin `memory` writes become Graphiti episodes.**
  Nothing written on a turn reaches cognee. Turns keep their `User:` / `Assistant:`
  colon framing — deliberate, and what Graphiti's `message` extraction prompt expects.
- **`[memory:…]` writes are pinned** via `source_description: hermes:pinned-memory`
  (turns get `hermes:session-turn`). That marker is the contract a retention job reads;
  it lands as a property on the `Episodic` node.
- **`reference_time` is stamped when the turn happens, not when the flush runs.** After
  an outage requeue those differ, and it is the timestamp the bi-temporal model reasons
  about.
- **The write queue stays.** `add_memory` returns on *enqueue* (~11 ms client-side), so
  the queue is not absorbing extraction latency — extraction runs server-side in a
  per-group sequential worker at roughly ten seconds an episode. The queue exists to
  keep the HTTP call off the turn's critical path and to survive outages: a failure
  requeues only the *undelivered tail*, since resending a delivered episode would
  duplicate it.
- **`cognee_remember` is the one write path still aimed at cognee** — explicit durable
  facts join the permanent archive, and it is what keeps that tier writable during a
  Graphiti outage. It uses `/api/v1/remember` (server-side add + background cognify),
  attributed to the current session so session-scoped prefetch surfaces it immediately.
- **Role-marker sanitizing runs on every write.** The classifier that blocks
  `[assistant]`-style markers sits in front of the archive tier's LLM, so the sanitizer
  guards `cognee_remember` too. The working tier's LLM is local and does not block them,
  so there it is defence in depth.

### Reads

- **`cognee_recall` queries both tiers** and returns `working` (RRF-ranked facts from
  this agent's group) and `archive` (dataset-wide cognee search) separately, plus a
  `degraded` list naming any tier that was unreachable. Only a *both*-tier outage is an
  error.
- **Per-turn prefetch** emits one labelled block per tier that returned anything —
  `<working-memory>` and `<cognee-memory>`. The archive half stays scoped to this
  agent's `node_set` and the current `sessionId`; explicit `cognee_recall` is
  deliberately dataset-wide so a direct "check memory" request can reach everything.
- **`cognee_forget` deletes a cognee *dataset* only.** The working tier has no dataset
  concept; dropping a group is a separate operation. Both modes require `confirm=true`
  and the exact dataset name (it is shared across agents). With `memory_only=true` only
  graph nodes/edges and embeddings are wiped — raw records survive and can be
  re-cognified.
- Both tiers are best-effort: a down server never breaks a turn.

## CLI

```bash
hermes cognee status              # reachability + config for both tiers
hermes cognee recall "<query>"    # one-off recall, printed per tier
```

## Development

```bash
git clone https://github.com/NousResearch/hermes-agent  # provides the MemoryProvider ABC
pip install -r requirements-dev.txt
HERMES_AGENT_PATH=$PWD/hermes-agent python -m pytest -v
```

Tests are fully offline — the cognee REST surface and the Graphiti MCP surface are both
driven through `httpx.MockTransport`. The Graphiti tests assert exact JSON-RPC argument
dicts on purpose: **the server silently drops unknown arguments and returns an empty
success**, so a misspelt `group_ids` reads as "the tier is empty" rather than failing.
