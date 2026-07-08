#!/usr/bin/env bash
# Idempotent bootstrap for the Stateful-AI-on-Kubernetes demo.
# Safe to re-run: every step uses `apply`, `helm upgrade --install`, or a
# dry-run pipe so a second run reconciles rather than breaks the cluster.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"
K8S="$DEMO_DIR/k8s"
NS="mongodb"

log() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- 0. Load .env if present -------------------------------------------------
# Variables defined in .env are exported for this run (see .env.example).
if [ -f "$DEMO_DIR/.env" ]; then
  log "Loading environment from $DEMO_DIR/.env"
  set -a
  # shellcheck disable=SC1091
  . "$DEMO_DIR/.env"
  set +a
fi

# --- 1. Preconditions --------------------------------------------------------
for bin in minikube kubectl helm docker; do
  command -v "$bin" >/dev/null 2>&1 || die "'$bin' is required but not installed."
done
: "${VOYAGE_API_KEY:?Set VOYAGE_API_KEY}"
: "${AZURE_GROVE_API_KEY:?Set AZURE_GROVE_API_KEY}"
: "${AZURE_GROVE_ENDPOINT:?Set AZURE_GROVE_ENDPOINT}"
LLM_PROVIDER="${LLM_PROVIDER:-openai}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-gpt-5.5}"
VOYAGE_API_BASE="${VOYAGE_API_BASE:-https://ai.mongodb.com/v1}"

# --- 2. minikube -------------------------------------------------------------
if ! minikube status >/dev/null 2>&1; then
  log "Starting minikube (docker driver, 6GB / 4 CPU)"
  minikube start --driver=docker --memory=6144 --cpus=4
else
  log "minikube already running"
fi
minikube addons enable default-storageclass >/dev/null 2>&1 || true

# --- 3. Helm repositories ----------------------------------------------------
log "Configuring Helm repositories"
helm repo add mongodb https://mongodb.github.io/helm-charts >/dev/null 2>&1 || true
helm repo add jetstack https://charts.jetstack.io >/dev/null 2>&1 || true
helm repo update >/dev/null

# --- 4. cert-manager (required for mongod <-> mongot TLS) --------------------
log "Installing cert-manager"
helm upgrade --install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace \
  --set crds.enabled=true --wait
for d in cert-manager cert-manager-cainjector cert-manager-webhook; do
  kubectl -n cert-manager rollout status "deployment/$d" --timeout=180s
done

# --- 5. MCK operator ---------------------------------------------------------
log "Installing MongoDB Controllers for Kubernetes operator"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install mongodb-kubernetes mongodb/mongodb-kubernetes \
  --namespace "$NS" -f "$K8S/mongodb-operator.yaml" --wait
kubectl -n "$NS" rollout status deployment/mongodb-kubernetes-operator --timeout=180s

# --- 6. MongoDB replica set + TLS materials ----------------------------------
log "Applying issuers, certificates, users, and MongoDBCommunity CR"
kubectl apply -f "$K8S/mongodb-community.yaml"

log "Waiting for the local CA, then publishing the CA bundle ConfigMap"
kubectl -n "$NS" wait --for=condition=Ready certificate/mongodb-ca --timeout=180s
CA_CRT="$(kubectl -n "$NS" get secret mongodb-ca -o jsonpath='{.data.ca\.crt}' | base64 -d)"
kubectl -n "$NS" create configmap mongodb-ca-cm \
  --from-literal=ca.crt="$CA_CRT" --dry-run=client -o yaml | kubectl apply -f -

log "Waiting for the MongoDBCommunity replica set to reach Running (a few minutes)"
kubectl -n "$NS" wait --for=jsonpath='{.status.phase}'=Running \
  mdbc/mdbc-rs --timeout=600s

# --- 7. MongoDB Search (mongot) ----------------------------------------------
log "Applying MongoDBSearch (mongot) resource"
kubectl apply -f "$K8S/mongodb-search-index.yaml"
kubectl -n "$NS" wait --for=jsonpath='{.status.phase}'=Running \
  mdbs/mdbc-rs --timeout=600s

# Applying MongoDBSearch makes the operator inject search config into mongod,
# which triggers a rolling restart of the replica set. create_indexes.py will
# fail with SearchNotEnabled / code 125 if it runs before that restart lands,
# so give the operator time to begin reconciling and then wait for the replica
# set to settle back into Running with search enabled.
log "Waiting for the replica set to finish the search-enabling rolling restart"
sleep 30
kubectl -n "$NS" wait --for=jsonpath='{.status.phase}'=Running \
  mdbc/mdbc-rs --timeout=600s
kubectl -n "$NS" rollout status statefulset/mdbc-rs --timeout=600s

# --- 8. Build the agent image into minikube ----------------------------------
log "Building the agent image inside the minikube Docker daemon"
eval "$(minikube docker-env)"
docker build -t stateful-agent:latest -f "$DEMO_DIR/agent/Dockerfile" "$DEMO_DIR"
eval "$(minikube docker-env -u)"

# --- 9. Agent secrets + config + deployment ----------------------------------
log "Creating agent secrets from your environment"
kubectl -n "$NS" create secret generic agent-secrets \
  --from-literal=VOYAGE_API_KEY="$VOYAGE_API_KEY" \
  --from-literal=AZURE_GROVE_API_KEY="$AZURE_GROVE_API_KEY" \
  --from-literal=AZURE_GROVE_ENDPOINT="$AZURE_GROVE_ENDPOINT" \
  --dry-run=client -o yaml | kubectl apply -f -

log "Applying agent ConfigMap and Deployment"
kubectl apply -f "$K8S/agent-configmap.yaml"
# Override the LLM provider/model and Voyage endpoint from the environment.
kubectl -n "$NS" patch configmap agent-config --type merge \
  -p "{\"data\":{\"LLM_PROVIDER\":\"$LLM_PROVIDER\",\"LLM_MODEL_NAME\":\"$LLM_MODEL_NAME\",\"VOYAGE_API_BASE\":\"$VOYAGE_API_BASE\"}}"
kubectl apply -f "$K8S/agent-deployment.yaml"
kubectl -n "$NS" rollout status deployment/stateful-agent --timeout=180s

# --- 10. Search + vector indexes --------------------------------------------
log "Creating search and vector indexes (waits until READY)"
kubectl -n "$NS" exec deploy/stateful-agent -- python create_indexes.py

log "Setup complete. Try: bash scripts/demo.sh"
