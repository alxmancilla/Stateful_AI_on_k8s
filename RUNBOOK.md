# Presenter Runbook — Stateful AI on Kubernetes

A step-by-step guide for running this demo live during a tech talk. It assumes
you have already read `README.md`. Total on-stage time is ~8–10 minutes once
the cluster is up.

---

## The three scripts

| Script                | What it does on stage                                        |
|-----------------------|--------------------------------------------------------------|
| `scripts/setup.sh`    | Provisions everything. **Run this BEFORE the talk.**         |
| `scripts/demo.sh`     | Guided, keypress-paced walkthrough of the agent's abilities. |
| `scripts/chaos.sh`    | Kills a MongoDB pod to prove memory survives.                |

`demo.sh` pauses for **Enter** between every step, so you narrate first, then
press Enter to run the command in front of the audience.

---

## T-minus 30 min: provision (do this off-stage)

```bash
cp .env.example .env        # fill in the three required keys
bash scripts/setup.sh       # ~10–15 min, mostly image pulls
```

Required in `.env` (see README "Environment variables"):

* `VOYAGE_API_KEY` — a **MongoDB Atlas model API key** (not a `pa-…` key).
* `AZURE_GROVE_API_KEY` and `AZURE_GROVE_ENDPOINT` (base URL ending `/openai/v1`).

Confirm it is healthy before you present:

```bash
kubectl -n mongodb get pods                 # all Running/Ready
kubectl -n mongodb port-forward svc/stateful-agent 8080:80 &
curl -s localhost:8080/health               # {"status":"ok"}
kill %1                                      # demo.sh opens its own forward
```

> Leave the cluster running. Do **not** re-run `setup.sh` during the talk.

> **Version note:** the replica set is pinned to MongoDB **`8.2`** (major/LTS)
> in `k8s/mongodb-community.yaml`. Do **not** bump `spec.version` to a minor
> release (e.g. `8.3`) before or during the talk — the operator uses the
> `OnDelete` update strategy, and an FCV-unaware bump can wedge a member in a
> version split. See README "MongoDB version policy" for the safe procedure.

---

## On stage, part 1 — the guided demo

```bash
bash scripts/demo.sh
```

Narration beats (each ends with an Enter press):

1. **Connect** — a `port-forward` to the in-cluster agent Service. Point out the
   agent pod is stateless; all state lives in MongoDB.
2. **Teach three facts** — name/company, tech stack, language preference. Each
   turn is embedded and written to MongoDB as it happens.
3. **Inspect memory** — `/memory/<session>` shows the raw persisted turns.
   "This is the durable state — rows in a local MongoDB replica set."
4. **Hybrid search** — `$rankFusion` blends `$vectorSearch` (semantic) and
   `$search` (BM25). Then the same query in **pure vector** mode — call out how
   the ordering differs.
5. **Rerank** — Voyage `rerank-2.5` (via the Atlas API) re-scores the fused
   candidates. Show the `before` vs `after` ordering flip.
6. **Memory-grounded answer** — ask "what's my name and preferred language?".
   The agent answers from retrieved long-term memory, not the prompt.

---

## On stage, part 2 — chaos / durability

```bash
bash scripts/chaos.sh
```

Talking points as it runs:

1. Stores a fact and reads it back.
2. **Resolves and deletes the current PRIMARY** with no grace period — "pull
   the plug on the DB node that's serving writes." (Falls back to `mdbc-rs-0`
   if the primary can't be resolved.)
3. The StatefulSet recreates the pod; wait for the replica set to report
   `Running`. Data is on a `PersistentVolumeClaim` and replicated.
4. Replays the query — **the fact is still there.**
5. Restarts the agent pod to reinforce: memory is in MongoDB, not the agent.

---

## Custom / off-script moments

Talk to the agent directly (great for audience questions):

```bash
kubectl -n mongodb port-forward svc/stateful-agent 8080:80 &
curl -s localhost:8080/chat -H 'content-type: application/json' \
  -d '{"session_id":"live","message":"Remember I like Rust."}' | jq .reply
```

Interactive CLI inside the pod:

```bash
kubectl -n mongodb exec -it deploy/stateful-agent -- python main.py chat
```

Tuning knobs (via env / `.env`):

* `SESSION_ID` — use a fresh id for a clean slate per run.
* `LOCAL_PORT` — change if `8080` is taken.

---

## If something breaks mid-talk

| Symptom                                   | Fast recovery                                                        |
|-------------------------------------------|----------------------------------------------------------------------|
| `demo.sh` can't reach the agent           | Re-run it; it re-establishes the port-forward. Check `kubectl get pods`. |
| Agent replies error / LLM 4xx             | `kubectl -n mongodb logs deploy/stateful-agent --tail=30`.           |
| Search returns nothing                    | Confirm `mongot`: `kubectl -n mongodb get pods -l app=mdbc-rs-search-svc`. |
| Voyage 401 / APIError                     | Wrong key type — must be an Atlas model key. See README troubleshooting. |
| Port already in use                       | `LOCAL_PORT=8090 bash scripts/demo.sh`.                              |
| One MongoDB member stuck / CR `Pending`   | Version split from an unsafe bump. Patch the StatefulSet image back to `8.2.0` and delete that member's pod **+** PVC to re-sync. See README "MongoDB version policy". |

Have a browser tab with the README "Troubleshooting" section open as backup.

---

## After the talk — cleanup

```bash
kubectl delete namespace mongodb   # remove just the demo
# or
minikube delete                    # tear the whole cluster down
```
