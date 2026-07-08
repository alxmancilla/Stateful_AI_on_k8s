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
└── scripts/
    ├── setup.sh           # one-shot, idempotent provisioning
    ├── create_indexes.py  # creates + waits for the search/vector indexes
    ├── demo.sh            # guided, keypress-paced live walkthrough
    └── chaos.sh           # pod-failure / persistence demonstration
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
  "embedding": [0.01, -0.02, ...],   // 1024-dim, voyage-3
  "timestamp": "2026-07-05T12:00:00Z",
  "turn": 0                            // monotonic per session
}
```

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
| GET    | `/health`                | liveness / readiness                      |
| POST   | `/chat`                  | full RAG turn, persists both messages     |
| POST   | `/search`                | `mode: hybrid` or `vector` retrieval only |
| POST   | `/rerank`                | show candidates before vs. after rerank   |
| GET    | `/memory/{session_id}`   | inspect recent persisted turns            |

### Talk to it manually

```bash
kubectl -n mongodb port-forward svc/stateful-agent 8080:80 &
curl -s localhost:8080/chat -H 'content-type: application/json' \
  -d '{"session_id":"s1","message":"Hello, remember I like Rust."}' | jq .

# Or an interactive CLI inside the pod:
kubectl -n mongodb exec -it deploy/stateful-agent -- python main.py chat
```

---

## Chaos demo (what it proves)

`scripts/chaos.sh`:

1. Stores a memorable fact through the agent and reads it back.
2. Deletes a MongoDB replica-set pod (`mdbc-rs-0`) with no grace period.
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

