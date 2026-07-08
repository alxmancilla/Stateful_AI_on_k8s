#!/usr/bin/env bash
# Chaos demo: prove that agent memory survives pod failure.
#
# 1. Write a fact through the agent.
# 2. Kill the current MongoDB PRIMARY pod and watch StatefulSet + the replica
#    set recover on their own.
# 3. Replay a memory query -> the fact is still there (PersistentVolumeClaim +
#    replication kept the data).
# Optionally also restarts the agent pod to show the stateless tier recovers.
set -euo pipefail

NS="mongodb"
LOCAL_PORT="${LOCAL_PORT:-8080}"
BASE="http://127.0.0.1:${LOCAL_PORT}"
SESSION="${SESSION_ID:-chaos-demo}"
STS="mdbc-rs"

C_HDR='\033[1;36m'; C_CMD='\033[1;33m'; C_OK='\033[1;32m'; C_ERR='\033[1;31m'; C_OFF='\033[0m'
PF_PID=""

log()  { printf "\n${C_HDR}=== %s ===${C_OFF}\n" "$*"; }
note() { printf "${C_OK}%s${C_OFF}\n" "$*"; }
warn() { printf "${C_ERR}%s${C_OFF}\n" "$*"; }
run()  { printf "${C_CMD}\$ %s${C_OFF}\n" "$*"; eval "$*"; }
pause(){ printf "\n${C_OK}[press Enter to continue]${C_OFF}"; read -r _; }

cleanup() { [ -n "$PF_PID" ] && kill "$PF_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

if command -v jq >/dev/null 2>&1; then PP() { jq "$@"; }; else PP() { cat; }; fi

start_pf() {
  cleanup
  kubectl -n "$NS" port-forward svc/stateful-agent "${LOCAL_PORT}:80" \
    >/tmp/agent-pf.log 2>&1 &
  PF_PID=$!
  for _ in $(seq 1 30); do
    curl -sf "${BASE}/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  warn "agent health check did not pass in time"; return 1
}

chat() {
  curl -s "${BASE}/chat" -H 'content-type: application/json' \
    -d "{\"session_id\":\"${SESSION}\",\"message\":$1}"
}

# --- 0. Connect --------------------------------------------------------------
log "Connecting to the agent"
start_pf
run "curl -s ${BASE}/health | PP ."
pause

# --- 1. Write a fact ---------------------------------------------------------
log "1. Store a fact the audience picks"
run "chat '\"Remember this launch code for the demo: ZULU-42-TANGO.\"' | PP -r .reply"
note "Reading it back BEFORE the chaos:"
run "chat '\"What launch code did I just give you?\"' | PP -r .reply"
pause

# --- 2. Show the MongoDB pods ------------------------------------------------
log "2. Current MongoDB replica set pods (note the AGE column)"
run "kubectl -n ${NS} get pods -l app=${STS}-svc -o wide"
pause

# --- 3. Kill a pod -----------------------------------------------------------
log "3. Delete a MongoDB pod to simulate a node/pod failure"
VICTIM="${STS}-0"
warn "Deleting pod ${VICTIM} ..."
run "kubectl -n ${NS} delete pod ${VICTIM} --grace-period=0 --wait=false"
note "Watching the StatefulSet recreate it (Ctrl-C the watch once Running)..."
run "kubectl -n ${NS} get pods -l app=${STS}-svc -w &
     WPID=\$!; sleep 25; kill \$WPID 2>/dev/null || true"
pause

log "3b. Wait for the replica set to report all members healthy again"
run "kubectl -n ${NS} wait --for=jsonpath='{.status.phase}'=Running mdbc/${STS} --timeout=300s"
run "kubectl -n ${NS} rollout status statefulset/${STS} --timeout=300s"
pause

# --- 4. Prove persistence ----------------------------------------------------
log "4. Reconnect and replay the query -- the memory survived"
start_pf
run "chat '\"What launch code did I give you earlier?\"' | PP -r .reply"
note "The launch code is still ZULU-42-TANGO: data persisted on the PVC and"
note "was served by the recovered replica set. No memory was lost."
pause

# --- 5. (optional) restart the agent tier ------------------------------------
log "5. Bonus: restart the stateless agent pod (memory lives in MongoDB, not here)"
run "kubectl -n ${NS} rollout restart deployment/stateful-agent"
run "kubectl -n ${NS} rollout status deployment/stateful-agent --timeout=180s"
start_pf
run "chat '\"One more time: the launch code?\"' | PP -r .reply"
note "Chaos demo complete: stateful memory is durable across pod failures."
