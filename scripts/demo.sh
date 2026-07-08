#!/usr/bin/env bash
# Guided walkthrough for the live tech talk.
#
# Each stage prints a section header and pauses for a keypress so you can
# narrate before the command runs. The agent is reached over a local
# port-forward to the in-cluster Service, so nothing here needs cloud access.
set -euo pipefail

NS="mongodb"
LOCAL_PORT="${LOCAL_PORT:-8080}"
BASE="http://127.0.0.1:${LOCAL_PORT}"
SESSION="${SESSION_ID:-meetup-demo}"

C_HDR='\033[1;36m'; C_CMD='\033[1;33m'; C_OK='\033[1;32m'; C_OFF='\033[0m'
PF_PID=""

log()  { printf "\n${C_HDR}=== %s ===${C_OFF}\n" "$*"; }
note() { printf "${C_OK}%s${C_OFF}\n" "$*"; }
run()  { printf "${C_CMD}\$ %s${C_OFF}\n" "$*"; eval "$*"; }
pause(){ printf "\n${C_OK}[press Enter to continue]${C_OFF}"; read -r _; }

cleanup() { [ -n "$PF_PID" ] && kill "$PF_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# jq is optional; fall back to raw output when it is absent.
if command -v jq >/dev/null 2>&1; then PP() { jq "$@"; }; else PP() { cat; }; fi

chat() {
  curl -s "${BASE}/chat" -H 'content-type: application/json' \
    -d "{\"session_id\":\"${SESSION}\",\"message\":$1}"
}

# --- Start a background port-forward -----------------------------------------
log "Connecting to the agent (kubectl port-forward)"
kubectl -n "$NS" port-forward svc/stateful-agent "${LOCAL_PORT}:80" \
  >/tmp/agent-pf.log 2>&1 &
PF_PID=$!
for _ in $(seq 1 30); do
  curl -sf "${BASE}/health" >/dev/null 2>&1 && break
  sleep 1
done
run "curl -s ${BASE}/health | PP ."
note "Agent reachable. Session id: ${SESSION}"
pause

# --- 1. Teach the agent some facts -------------------------------------------
log "1. Teach the agent (these turns are written to MongoDB)"
run "chat '\"Hi! My name is Dana and I run the platform team at Globex.\"' | PP -r .reply"
pause
run "chat '\"We standardized on Kubernetes and MongoDB for all new services.\"' | PP -r .reply"
pause
run "chat '\"My favorite programming language is Rust, but our services are in Go.\"' | PP -r .reply"
pause

# --- 2. Show the persisted memory --------------------------------------------
log "2. Inspect the recent turns persisted in MongoDB"
run "curl -s '${BASE}/memory/${SESSION}?n=6' | PP ."
pause

# --- 3. Hybrid search vs. pure vector ----------------------------------------
log "3. Hybrid search (\$rankFusion: \$vectorSearch + \$search)"
QUERY='"Which programming languages does the team use?"'
run "curl -s ${BASE}/search -H 'content-type: application/json' \
  -d '{\"session_id\":\"${SESSION}\",\"query\":${QUERY},\"mode\":\"hybrid\"}' | PP '.results[] | {role, content, score}'"
pause

log "3b. Pure vector search for the same query (compare the ordering)"
run "curl -s ${BASE}/search -H 'content-type: application/json' \
  -d '{\"session_id\":\"${SESSION}\",\"query\":${QUERY},\"mode\":\"vector\"}' | PP '.results[] | {role, content, score}'"
pause

# --- 4. Reranking ------------------------------------------------------------
log "4. Voyage AI reranking (before vs. after cross-encoder rescoring)"
run "curl -s ${BASE}/rerank -H 'content-type: application/json' \
  -d '{\"session_id\":\"${SESSION}\",\"query\":${QUERY},\"top_k\":10}' | PP ."
pause

# --- 5. Memory-grounded answer -----------------------------------------------
log "5. Ask a question that requires long-term memory"
run "chat '\"Remind me: what is my name and what language do I personally prefer?\"' | PP -r .reply"
note "The agent recalls facts from earlier turns retrieved out of MongoDB."
pause

log "Demo complete. Run scripts/chaos.sh to show persistence across pod failure."
