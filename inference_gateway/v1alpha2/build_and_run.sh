#!/bin/bash
set -euo pipefail

# --- Configuration ---
export PROJECT_ID="gemle-gke-dev"
export CLUSTER_NAME="chavoshi-inference-gateway-ext"
export ZONE="us-west2-b"
export REGION="us-west2"
export MACHINE_TYPE="n1-standard-8"
export TPU_NODE_POOL_NAME="tpu-v6e-pool"
# Find available TPU types with: gcloud compute accelerator-types list --filter="name~ct6e"
export TPU_TYPE="ct6e-hightpu-4t"
# Find available topologies with: gcloud compute accelerator-types describe ct6e-hightpu-4t --zone us-central2-b
export TPU_TOPOLOGY="2x2x1"
export HF_TOKEN_PLACEHOLDER="HF_TOKEN" # Replace with your actual Hugging Face token

# --- Helper Functions ---
info() {
  echo "✅ [INFO] $1"
}

warn() {
  echo "⚠️ [WARN] $1"
}

error() {
  echo "❌ [ERROR] $1" >&2
  exit 1
}

# --- Script ---

info "Setting active project to $PROJECT_ID"
gcloud config set project "$PROJECT_ID"

info "Checking if GKE cluster '$CLUSTER_NAME' exists in zone '$ZONE'..."
if ! gcloud container clusters describe "$CLUSTER_NAME" --zone "$ZONE" --project "$PROJECT_ID" &>/dev/null; then
  info "Cluster not found. Creating cluster '$CLUSTER_NAME'..."
  gcloud beta container --project "$PROJECT_ID" clusters create "$CLUSTER_NAME" --zone "$ZONE" --no-enable-basic-auth --cluster-version "1.33.3-gke.1136000" --release-channel "regular" --machine-type "$MACHINE_TYPE" --image-type "COS_CONTAINERD" --disk-type "pd-balanced" --disk-size "100" --metadata disable-legacy-endpoints=true --scopes "https://www.googleapis.com/auth/devstorage.full_control","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --max-pods-per-node "110" --num-nodes "3" --logging=SYSTEM,WORKLOAD --monitoring=SYSTEM --enable-ip-alias --network "projects/$PROJECT_ID/global/networks/default" --subnetwork "projects/$PROJECT_ID/regions/$REGION/subnetworks/default" --no-enable-intra-node-visibility --default-max-pods-per-node "110" --security-posture=standard --workload-vulnerability-scanning=disabled --addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver,GcsFuseCsiDriver --enable-autoupgrade --enable-autorepair --max-surge-upgrade 1 --max-unavailable-upgrade 0 --enable-managed-prometheus --workload-pool "$PROJECT_ID.svc.id.goog" --enable-shielded-nodes
else
  info "Cluster '$CLUSTER_NAME' already exists."
fi

info "Getting credentials for cluster '$CLUSTER_NAME'"
gcloud container clusters get-credentials "$CLUSTER_NAME" --zone "$ZONE" --project "$PROJECT_ID"

info "Checking if TPU node pool '$TPU_NODE_POOL_NAME' exists..."
if ! gcloud container node-pools describe "$TPU_NODE_POOL_NAME" --cluster "$CLUSTER_NAME" --zone "$ZONE" --project "$PROJECT_ID" &>/dev/null; then
    info "TPU node pool not found. Creating '$TPU_NODE_POOL_NAME'..."
    gcloud container node-pools create "$TPU_NODE_POOL_NAME" \
        --cluster="$CLUSTER_NAME" \
        --machine-type="$TPU_TYPE" \
        --num-nodes=1 \
        --tpu-topology="$TPU_TOPOLOGY" \
        --zone="$ZONE" \
        --project="$PROJECT_ID"
else
    info "TPU node pool '$TPU_NODE_POOL_NAME' already exists."
fi

info "Checking if Gateway API is enabled..."
GATEWAY_STATUS=$(gcloud container clusters describe "$CLUSTER_NAME" --zone "$ZONE" --project "$PROJECT_ID" --format='value(gatewayApiConfig.channel)')
if [ "$GATEWAY_STATUS" != "standard" ]; then
    info "Gateway API not enabled or not standard. Enabling it now..."
    gcloud container clusters update "$CLUSTER_NAME" --zone "$ZONE" --project "$PROJECT_ID" --gateway-api=standard
    info "Waiting for cluster update to complete. This may take a few minutes..."
    gcloud container operations list --project="$PROJECT_ID" --filter="targetLink~/$CLUSTER_NAME AND status=RUNNING" --format='value(name)' | xargs -r gcloud container operations wait --project="$PROJECT_ID" --zone="$ZONE"
else
    info "Gateway API is already enabled."
fi

info "Installing required CRDs if they don't exist..."
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.1.0/standard-install.yaml
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v0.3.0/manifests.yaml
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/gke-gateway-api/main/config/crd/networking.gke.io_gcpbackendpolicies.yaml
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/gke-gateway-api/main/config/crd/networking.gke.io_healthcheckpolicies.yaml
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/gke-gateway-api/refs/heads/main/config/crd/networking.gke.io_gcptrafficextensions.yaml
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/gke-gateway-api/refs/heads/main/config/crd/networking.gke.io_gcproutingextensions.yaml

info "Applying RBAC for metrics if not present..."
if ! kubectl get clusterrole inference-gateway-metrics-reader &>/dev/null; then
    kubectl apply -f ./metrics-rbac.yaml
else
    info "Metrics RBAC already exists."
fi

info "Creating Hugging Face token secret if not present..."
if ! kubectl get secret hf-token &>/dev/null; then
    if [ "$HF_TOKEN_PLACEHOLDER" == "HF_TOKEN" ]; then
        warn "Using a placeholder for Hugging Face token. The model deployment may fail if it needs to pull private resources."
    fi
    kubectl create secret generic hf-token --from-literal=token="$HF_TOKEN_PLACEHOLDER"
else
    info "Secret 'hf-token' already exists."
fi

info "Applying vLLM deployment and ConfigMap if not present..."
if ! kubectl get deployment vllm-llama3-8b-instruct &>/dev/null; then
    kubectl apply -f ./vllm-llama3-8b-instruct.yaml
else
    info "Deployment 'vllm-llama3-8b-instruct' already exists."
fi

info "Installing InferencePool via Helm if not present..."
if ! helm status vllm-llama3-8b-instruct &>/dev/null; then
    helm install vllm-llama3-8b-instruct \
      --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
      --set provider.name=gke \
      --version v0.3.0 \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
else
    info "Helm release 'vllm-llama3-8b-instruct' already exists."
fi

info "Applying InferenceModel resources if not present..."
if ! kubectl get inferencemodel food-review &>/dev/null; then
    kubectl apply -f ./inferencemodel.yaml
else
    info "InferenceModel 'food-review' already exists."
fi

info "Applying Gateway if not present..."
if ! kubectl get gateway inference-gateway &>/dev/null; then
    kubectl apply -f ./gateway.yaml
else
    info "Gateway 'inference-gateway' already exists."
fi

info "Applying HTTPRoute if not present..."
if ! kubectl get httproute my-route &>/dev/null; then
    kubectl apply -f ./httproute.yaml
else
    info "HTTPRoute 'my-route' already exists."
fi

info "Waiting for Gateway to get an external IP address..."
IP=""
for i in {1..30}; do
    IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}' 2>/dev/null || true)
    if [ -n "$IP" ]; then
        info "Gateway IP found: $IP"
        break
    fi
    info "Still waiting for IP... ($i/30)"
    sleep 10
done

if [ -z "$IP" ]; then
    error "Failed to get Gateway IP address after 5 minutes. Please check the Gateway status manually with 'kubectl describe gateway inference-gateway'."
fi

info "Setup complete!"
info "You can send a test request with the following command:"
echo
echo "curl -i -X POST http://${IP}:80/v1/completions -H 'Content-Type: application/json' -d '{\
    \"model\": \"food-review\",\
    \"prompt\": \"What is the best pizza in the world?\",\
    \"max_tokens\": 2048,\
    \"temperature\": \"0\"\
}'"
echo
