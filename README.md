# Stateful AI on Kubernetes

A fully self-contained, local demo of a **stateful LLM agent** running on
**minikube**, using **MongoDB Community Edition 8** as its persistent memory
layer. The agent remembers past turns, retrieves relevant long-term context
with **hybrid search** (`$vectorSearch` + `$search` fused by `$rankFusion`),
sharpens results with the **Voyage AI reranker**, and answers via an LLM served
through **Azure Grove**.

The MongoDB replica set and `mongot` search node run **locally on minikube** —
no MongoDB Atlas cluster is required for the database. Two hosted APIs are
called over the network:

* **Voyage AI embeddings + rerank**, accessed through the **MongoDB Atlas
  Embedding and Reranking API** (`https://ai.mongodb.com/v1`). `VOYAGE_API_KEY`
  is therefore a **MongoDB Atlas model API key**, created in the Atlas UI.
* **Azure Grove**, an **OpenAI-compatible** LLM gateway.

---

## Architecture

```
                         minikube (namespace: mongodb)
 ┌──────────────────────────────────────────────────────────────────────┐
 │                                                                        │
 │   ┌─────────────────┐        ┌──────────────────────────────────────┐ │
 │   │ stateful-agent  │        │  MongoDBCommunity replica set (x3)    │ │
 │   │ (FastAPI, Py)   │  TLS   │  mdbc-rs-0 / -1 / -2  (mongod)        │ │
 │   │                 │───────▶│  + mongodb-agent sidecar             │ │
 │   │  /chat /search  │  SCRAM │  each backed by a PersistentVolume    │ │
 │   │  /rerank /memory│        └──────────────────────────────────────┘ │
 │   └───────┬─────────┘                       ▲                          │
 │           │                                 │ replication / mongot     │
 │           │                        ┌────────┴──────────┐               │
 │           │                        │  MongoDBSearch    │               │
 │           │                        │  (mongot node)    │               │
 │           │                        │  $search /        │               │
 │           │                        │  $vectorSearch    │               │
 │           │                        └───────────────────┘               │
 │           │   managed by: MongoDB Controllers for Kubernetes (MCK)     │
 │           │                + cert-manager (TLS)                        │
 └───────────┼────────────────────────────────────────────────────────────┘
             │
   Voyage AI via MongoDB Atlas API      Azure Grove (OpenAI-compatible LLM)
   (https://ai.mongodb.com/v1)          (embeddings + rerank)     (gpt-5.5)
```

* **MongoDB Community + MCK operator** runs a 3-member replica set. The
  open-source `MongoDBSearch` resource brings up a `mongot` process, which is
  what provides `$search`/`$vectorSearch` *without* Atlas.
* **The agent** is a small FastAPI service. Memory lives entirely in MongoDB,
  so the agent pod itself is stateless and disposable.

The replica set is pinned to the **`8.2` major (LTS) release** — see
[MongoDB version policy](#mongodb-version-policy) for why, and the safe upgrade
path if you ever need a newer version.

---

## Prerequisites

Install these on your machine first:

| Tool       | Version           | Notes                                  |
|------------|-------------------|----------------------------------------|
| minikube   | latest            | docker driver recommended              |
| kubectl    | 1.28+             | matches your minikube Kubernetes       |
| helm       | 3.12+             | installs cert-manager + MCK operator   |
| docker     | latest            | builds the agent image into minikube   |
| curl       | any               | used by the demo scripts               |
| jq         | any (optional)    | pretty-prints demo output if present   |

Recommended host resources: **>= 8 GB RAM free** (minikube is started with
6 GB / 4 CPUs) and ~10 GB free disk.

---

## Environment variables

The only secrets you must provide are for the two external APIs. Copy the
template and fill in your values:

```bash
cp .env.example .env
# edit .env
```

`.env` (gitignored) holds:

```bash
# Voyage embeddings + rerank, accessed through the MongoDB Atlas model API.
# This is a MongoDB Atlas model API key (created in the Atlas UI), NOT a
# native voyageai.com key.
VOYAGE_API_KEY="your-atlas-model-api-key"

# Azure Grove LLM gateway (OpenAI-compatible).
AZURE_GROVE_API_KEY="your-grove-key"        # sent as the "api-key" header
AZURE_GROVE_ENDPOINT="https://<grove-host>/openai/v1" # OpenAI-compatible base URL

# Optional (defaults shown):
LLM_PROVIDER="openai"        # openai | anthropic
LLM_MODEL_NAME="gpt-5.5"     # model name exposed by your Grove gateway
VOYAGE_API_BASE="https://ai.mongodb.com/v1" # Atlas Embedding & Reranking API
```

`setup.sh` auto-loads `.env` if present (you can also just `export` these
in your shell instead). Secret values are stored in a Kubernetes `Secret`
(`agent-secrets`) and are never baked into the image or committed to git.

> **Voyage key note:** the Voyage clients route through the MongoDB Atlas
> Embedding and Reranking API (`https://ai.mongodb.com/v1`) by default. Atlas
> model API keys are rejected by the native `api.voyageai.com` host, so a
> standard `pa-…` Voyage key will not work unless you also override
> `VOYAGE_API_BASE` back to `https://api.voyageai.com/v1`.

> **LLM gateway note:** Azure Grove is reached with LangChain's `ChatOpenAI`
> (not `AzureChatOpenAI`). The endpoint already ends in `/openai/v1`, the key
> is passed via the `api-key` header, and `temperature` is pinned to `1`
> because some newer models (e.g. `gpt-5.5`) reject any non-default value.

---

## Quick start (zero to demo in ~15 minutes)

```bash
# 1. Provide your API credentials (see the section above).
cp .env.example .env   # then edit .env with your keys

# 2. Provision everything: minikube, cert-manager, the MCK operator,
#    the MongoDB replica set + mongot, the agent image, and the indexes.
bash scripts/setup.sh

# 3. Run the guided live demo (port-forwards to the agent and walks the flow).
bash scripts/demo.sh

# 4. Show durability: kill a MongoDB pod and prove memory survives.
bash scripts/chaos.sh
```

The bulk of the time is pulling the MongoDB images and waiting for the replica
set + `mongot` to reach `Running`. `setup.sh` is **idempotent** — re-run it
safely if a step times out.

Presenting this at a talk? See **[`RUNBOOK.md`](RUNBOOK.md)** for a timed,
narrated walkthrough with recovery tips.

---

## Project structure

```
.
├── README.md              # this file
├── RUNBOOK.md             # presenter how-to for running the live demo
├── .env.example           # copy to .env and fill in your API keys
├── .gitignore             # keeps .env and Python caches out of git
├── .dockerignore          # keeps .env, .git, docs, caches out of the build context
├── requirements-dev.txt   # test-only deps (pytest, mongomock); not in the image
├── agent/
│   ├── main.py            # FastAPI + CLI entrypoint, Settings, LLM wiring
│   ├── memory.py          # MongoDB CRUD + recent/semantic/hybrid retrieval
│   ├── search.py          # $rankFusion pipeline + Voyage reranker
│   ├── embeddings.py      # Voyage AI embeddings wrapper
│   ├── requirements.txt   # pinned Python dependencies
│   └── Dockerfile         # multi-stage build (slim runtime image)
├── k8s/
│   ├── mongodb-operator.yaml     # Helm values for the MCK operator
│   ├── mongodb-community.yaml    # namespace, TLS (cert-manager), users, RS
│   ├── mongodb-search-index.yaml # MongoDBSearch (mongot) + its TLS cert
│   ├── agent-configmap.yaml      # non-secret agent configuration
│   └── agent-deployment.yaml     # agent Deployment + Service
├── scripts/
│   ├── setup.sh           # one-shot, idempotent provisioning
│   ├── create_indexes.py  # creates + waits for the search/vector indexes
│   ├── demo.sh            # guided, keypress-paced live walkthrough
│   └── chaos.sh           # pod-failure / persistence demonstration
└── tests/
    └── test_memory_turns.py  # hermetic memory-layer tests (mongomock default)
```

---

## How it works

### Memory document schema

Each conversational turn is one document in `agent_memory.messages`:

```json
{
  "_id": "uuid",
  "session_id": "meetup-demo",
  "role": "user | assistant",
  "content": "the message text",
  "embedding": [0.01, -0.02, ...],   // 1024-dim, voyage-4
  "timestamp": "2026-07-05T12:00:00Z",
  "turn": 0                            // monotonic per session
}
```

Turn numbers come from an atomic per-session counter kept in a sibling
`agent_memory.counters` collection (`findOneAndUpdate` + `$inc`), and
`(session_id, turn)` is a unique index — so turns stay collision-free even
under concurrent writes.

### Retrieval flow (per `/chat` turn)

1. **Recent turns** — last `RECENT_TURNS` messages fetched by `turn` order.
2. **Hybrid search** — `build_rankfusion_pipeline` runs a `$vectorSearch`
   sub-pipeline and a `$search` (BM25) sub-pipeline and fuses their rankings
   with MongoDB 8's `$rankFusion` (equal weights), scoped to the session.
3. **Rerank** — the fused candidates are re-scored by Voyage `rerank-2.5`; the
   top `RERANK_TOP_N` become the long-term context.
4. **Answer** — recent turns + reranked context are rendered into a prompt and
   sent to the LLM via Azure Grove. Both the user message and the reply are
   embedded and written back to MongoDB.

### HTTP API (agent)

| Method | Path                     | Purpose                                   |
|--------|--------------------------|-------------------------------------------|
| GET    | `/`                      | demo web UI (chat / search / rerank)      |
| GET    | `/health`                | liveness / readiness                      |
| GET    | `/metrics`               | Prometheus-format in-process counters     |
| POST   | `/chat`                  | full RAG turn, persists both messages     |
| POST   | `/search`                | `mode: hybrid` or `vector` retrieval only |
| POST   | `/rerank`                | show candidates before vs. after rerank   |
| GET    | `/memory/{session_id}`   | inspect recent persisted turns            |

### Talk to it manually

```bash
kubectl -n mongodb port-forward svc/stateful-agent 8080:80 &

# Web UI (MongoDB look-and-feel): open http://localhost:8080/ in a browser —
# tabs for chat, hybrid/vector search, and before/after rerank.

curl -s localhost:8080/chat -H 'content-type: application/json' \
  -d '{"session_id":"s1","message":"Hello, remember I like Rust."}' | jq .

# Or an interactive CLI inside the pod:
kubectl -n mongodb exec -it deploy/stateful-agent -- python main.py chat
```

### MongoDB version policy

The replica set is pinned in `k8s/mongodb-community.yaml`:

```yaml
version: "8.2.0"
featureCompatibilityVersion: "8.2"
```

**Stay on `8.2`.** From MongoDB 8.2 onward there are two release tracks:
*major* (LTS) releases like `8.0` and `8.2`, and *minor* releases like `8.3`
that ship incremental features for specific use cases (Search, Vector Search,
Queryable Encryption). This demo's search runs on `mongot` + the Voyage/Atlas
APIs and needs nothing from the minor track, so `8.2` is the stable target.

Two rules make ad-hoc version bumps risky, and caused a live upgrade to wedge
during development:

* **Every upgrade *and downgrade* step needs both a binary change and an FCV
  change**, and you cannot skip minor releases (e.g. `8.2 → 8.3`, never
  straight to a later minor).
* The StatefulSet uses the `OnDelete` update strategy, so a `spec.version`
  change only takes effect when each pod is manually deleted. Bumping the
  binary without also advancing FCV — then trying to roll back — can leave one
  member on newer on-disk data than its binary, which the operator cannot
  reconcile.

If you ever must move to a newer version, do it deliberately:

1. Start healthy on `8.2` with FCV `8.2`.
2. Raise `spec.version`; because of `OnDelete`, delete pods **one at a time**
   (secondaries first, primary last), waiting for each to rejoin `2/2`.
3. **Only after all members are on the new binary**, raise
   `featureCompatibilityVersion`.
4. To roll back: **lower FCV first**, *then* step the binary down — never the
   reverse.

If a member does get stuck in a version split, the recovery is to patch the
StatefulSet image back to the good version and wipe just that member's PVC so
it re-syncs cleanly from the healthy majority.

---

## Tests

The `tests/` suite covers the memory layer's turn counter, unique-index
enforcement, and recent-read clamp. It is **hermetic by default**: with no
`TEST_MONGODB_URI` set it runs against an in-process `mongomock` backend, so it
needs no MongoDB and no network.

```bash
python3 -m pip install -r requirements-dev.txt
pytest tests/ -v
```

To also run the concurrency test (which needs true server-side atomicity),
point `TEST_MONGODB_URI` at a real replica set — e.g. a port-forward to the
demo primary:

```bash
kubectl -n mongodb port-forward pod/mdbc-rs-0 27018:27017 &
export TEST_MONGODB_URI="mongodb://admin-user:<pw>@127.0.0.1:27018/admin?tls=true&tlsAllowInvalidCertificates=true&directConnection=true"
pytest tests/ -v
```

`mongomock` and `pytest` are test-only (in `requirements-dev.txt`) and are not
installed into the agent image.

---

## Scope & non-goals

This is a **local, runnable demo of a production-style architecture pattern**
for durable agent memory on Kubernetes — not a production deployment. It shows
the *shape* of a real system (operator-managed replica set, hybrid retrieval,
reranking, a stateless agent, secret/config separation) so you can see how the
pieces fit, but it deliberately omits the operational controls a production
system needs. Called out explicitly so nothing here is mistaken for hardened:

* **Scaling** — the agent runs `replicas: 1` with no HorizontalPodAutoscaler,
  PodDisruptionBudget, or multi-replica validation.
* **Write concurrency** — turn numbers are assigned by an atomic per-session
  counter (`MongoMemory._next_turn()` uses `findOneAndUpdate` with `$inc` on a
  sibling `counters` collection), and `(session_id, turn)` is a **unique**
  index, so concurrent writes to the same `session_id` cannot collide on a
  `turn` value. What the demo does *not* exercise is running the agent at
  `replicas > 1` under real concurrent load (see scaling above).
* **Secrets** — the demo MongoDB passwords are **hardcoded** in
  `k8s/mongodb-community.yaml`. Real deployments would source these from a
  secret manager and never commit them.
* **Backup / DR** — there is no backup, point-in-time recovery, or restore
  procedure. Replica-set replication (what `chaos.sh` demonstrates) provides
  failover durability, **not** a backup.
* **Security hardening** — no NetworkPolicy, dedicated ServiceAccount/RBAC
  scoping, pod security context, or ingress/edge-TLS pattern. The agent HTTP
  API is unauthenticated; `session_id` is a client-asserted string, so queries
  are session-scoped but there is no tenant security boundary.
* **Observability** — a `/metrics` endpoint exposes Prometheus-format
  in-process counters (requests by route/outcome, messages saved by role,
  retrieval skips, uptime); counters are process-local and reset on restart.
  There is still no tracing, no dashboards, and no shared multi-replica
  registry.
* **Testing** — a hermetic memory-layer test suite exists (`tests/`, runs
  offline via `mongomock`; see [Tests](#tests)), but there is no wired-up CI
  pipeline, load testing, or RAG evaluation.
* **Memory lifecycle** — turns carry a BSON `created_at` and the agent can
  auto-expire them via an optional TTL index (`MEMORY_TTL_SECONDS`) and cap
  per-session history (`MAX_TURNS_PER_SESSION`); both are **disabled by
  default** so the demo keeps all history. There is still no summarization,
  compaction, or tenant isolation for stored turns.

In a talk, frame it as: *"a local demo of the architecture pattern; a
production deployment would add backup, observability, security hardening,
scaling, and operational controls."*

---

## Chaos demo (what it proves)

`scripts/chaos.sh`:

1. Stores a memorable fact through the agent and reads it back.
2. Resolves the current replica-set **primary** and deletes that pod with no
   grace period (falls back to `mdbc-rs-0` if the primary can't be resolved).
3. Watches the StatefulSet recreate the pod and waits for the replica set to
   report `Running` again.
4. Replays the query — the fact is still there because it is persisted on a
   `PersistentVolumeClaim` and replicated across members.
5. Restarts the (stateless) agent pod to show the memory lives in MongoDB, not
   in the agent.

---

## Troubleshooting

* **`setup.sh` times out waiting for `mdbc/mdbc-rs`** — image pulls can be
  slow the first time. Re-run `bash scripts/setup.sh` (it is idempotent) and
  watch progress with `kubectl -n mongodb get pods -w`.
* **Search indexes never reach `READY`** — confirm the `mongot` pod is up:
  `kubectl -n mongodb get pods -l app=mdbc-rs-search-svc`. Re-run the index
  step with `kubectl -n mongodb exec deploy/stateful-agent -- python create_indexes.py`.
* **Agent `CrashLoopBackOff`** — check env/secrets:
  `kubectl -n mongodb logs deploy/stateful-agent`. Verify `agent-secrets` and
  the `agent-mongodb-connection` secret exist.
* **cert-manager webhook errors** — wait for its deployments to be ready
  (`kubectl -n cert-manager get pods`) before re-running setup.
* **Not enough resources** — free host RAM or lower the replica set to 1
  member in `k8s/mongodb-community.yaml` (`spec.members: 1`).
* **One member stuck after a version change / CR stays `Pending`** — likely a
  version split from an unsafe upgrade (see
  [MongoDB version policy](#mongodb-version-policy)). Patch the StatefulSet
  `mongod` image back to the good version and delete that member's pod **and**
  its PVC so it re-syncs from the healthy majority; the other members and all
  data are untouched.
* **Voyage `401` / `APIError` on embed or rerank** — the `VOYAGE_API_KEY` must
  be a **MongoDB Atlas model API key** used against `https://ai.mongodb.com/v1`
  (the default). A native `pa-…` voyageai.com key requires setting
  `VOYAGE_API_BASE=https://api.voyageai.com/v1` instead.
* **LLM `404` from the gateway** — `AZURE_GROVE_ENDPOINT` must point at the
  OpenAI-compatible base URL (ending in `/openai/v1`); the agent POSTs to
  `{endpoint}/chat/completions`.
* **LLM `400 unsupported temperature`** — expected for models that only accept
  the default; the client already pins `temperature=1`.

---

## Cleanup

```bash
# Remove just the demo workloads:
kubectl delete namespace mongodb

# Or tear the whole cluster down:
minikube delete
```

