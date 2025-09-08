#!/bin/bash
set -euo pipefail

# --- Configuration ---
export PROJECT_ID="gemle-gke-dev"
export CLUSTER_NAME="chavoshi-inference-gateway-ext"
export ZONE="us-east4-b"
export REGION="us-east4"
export MACHINE_TYPE="n1-standard-8"
export TPU_NODE_POOL_NAME="tpu-8"
export TPU_MACHINE_TYPE="ct6e-standard-8t"
export GS_BUCKET="chavoshi-gkegmle"
export KSA_NAME="vllm-ksa"
export HF_SECRET_NAME="hf-secret"
# Replace with your actual Hugging Face token before running
export HF_TOKEN_PLACEHOLDER=""
export PROXY_SUBNET_NAME="proxy-only-subnet-${REGION}"
# Note: This range should not overlap with other subnets in the 'default' VPC
export PROXY_SUBNET_RANGE="192.168.253.0/24"


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

info "Checking for proxy-only subnet '$PROXY_SUBNET_NAME' in region '$REGION'..."
if ! gcloud compute networks subnets describe "$PROXY_SUBNET_NAME" --region "$REGION" --project "$PROJECT_ID" &>/dev/null; then
    info "Proxy-only subnet not found. Creating it now..."
    gcloud compute networks subnets create "$PROXY_SUBNET_NAME" \
        --purpose=REGIONAL_MANAGED_PROXY \
        --role=ACTIVE \
        --region="$REGION" \
        --network=default \
        --range="$PROXY_SUBNET_RANGE" \
        --project="$PROJECT_ID"
else
    info "Proxy-only subnet '$PROXY_SUBNET_NAME' already exists."
fi

info "Checking if GKE cluster '$CLUSTER_NAME' exists in zone '$ZONE'வதற்காக..."
if ! gcloud container clusters describe "$CLUSTER_NAME" --zone "$ZONE" --project "$PROJECT_ID" &>/dev/null; then
  info "Cluster not found. Creating cluster '$CLUSTER_NAME'வதற்காக..."
  gcloud beta container --project "$PROJECT_ID" clusters create "$CLUSTER_NAME" --zone "$ZONE" --no-enable-basic-auth --cluster-version "1.33.3-gke.1136000" --release-channel "regular" --machine-type "$MACHINE_TYPE" --image-type "COS_CONTAINERD" --disk-type "pd-balanced" --disk-size "100" --metadata disable-legacy-endpoints=true --scopes "https://www.googleapis.com/auth/devstorage.full_control","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --max-pods-per-node "110" --num-nodes "1" --logging=SYSTEM,WORKLOAD --monitoring=SYSTEM --enable-ip-alias --network "projects/$PROJECT_ID/global/networks/default" --subnetwork "projects/$PROJECT_ID/regions/$REGION/subnetworks/default" --no-enable-intra-node-visibility --default-max-pods-per-node "110" --security-posture=standard --workload-vulnerability-scanning=disabled --addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver,GcsFuseCsiDriver --enable-autoupgrade --enable-autorepair --max-surge-upgrade 1 --max-unavailable-upgrade 0 --enable-managed-prometheus --workload-pool "$PROJECT_ID.svc.id.goog" --enable-shielded-nodes
else
  info "Cluster '$CLUSTER_NAME' already exists."
fi

info "Getting credentials for cluster '$CLUSTER_NAME'"
gcloud container clusters get-credentials "$CLUSTER_NAME" --zone "$ZONE" --project "$PROJECT_ID"

info "Checking if TPU node pool '$TPU_NODE_POOL_NAME' exists..."
if ! gcloud container node-pools describe "$TPU_NODE_POOL_NAME" --cluster "$CLUSTER_NAME" --zone "$ZONE" --project "$PROJECT_ID" &>/dev/null; then
    info "TPU node pool not found. Creating '$TPU_NODE_POOL_NAME' with Spot VMs..."
    gcloud container node-pools create "$TPU_NODE_POOL_NAME" \
        --cluster="$CLUSTER_NAME" \
        --machine-type="$TPU_MACHINE_TYPE" \
        --spot \
        --zone="$ZONE" \
        --project="$PROJECT_ID"
else
    info "TPU node pool '$TPU_NODE_POOL_NAME' already exists."
fi

info "Creating Kubernetes Service Account '$KSA_NAME' if it doesn't exist..."
if ! kubectl get sa "$KSA_NAME" &>/dev/null; then
    kubectl create serviceaccount "$KSA_NAME"
else
    info "Service Account '$KSA_NAME' already exists."
fi

info "Granting Service Account IAM permissions for GCS bucket '$GS_BUCKET'..."
PROJECT_NUMBER_CMD="gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)'"
PROJECT_NUMBER=$($PROJECT_NUMBER_CMD)
if [ -z "$PROJECT_NUMBER" ]; then
    error "Failed to get project number for project '$PROJECT_ID'. Please check your gcloud configuration."
fi
gcloud storage buckets add-iam-policy-binding "gs://${GS_BUCKET}" \
  --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/default/sa/${KSA_NAME}" \
  --role "roles/storage.objectUser"

info "Creating Hugging Face token secret '$HF_SECRET_NAME' if it doesn't exist..."
if ! kubectl get secret "$HF_SECRET_NAME" &>/dev/null; then
    if [ "$HF_TOKEN_PLACEHOLDER" == "HF_TOKEN" ]; then
        warn "Using a placeholder for Hugging Face token. The model deployment will fail."
        warn "Please edit this script and replace 'HF_TOKEN' with your actual token."
    fi
    kubectl create secret generic "$HF_SECRET_NAME" --from-literal=hf_api_token="$HF_TOKEN_PLACEHOLDER"
else
    info "Secret '$HF_SECRET_NAME' already exists."
fi

info "Applying the vLLM TPU deployment and service..."
kubectl apply -f ./vllm-tpu-deployment.yaml

info "Waiting for the vLLM service to get an external IP address..."
IP=""
for i in {1..30}; do
    IP=$(kubectl get service/vllm-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
    if [ -n "$IP" ]; then
        info "LoadBalancer IP found: $IP"
        break
    fi
    info "Still waiting for IP... ($i/30)"
    sleep 10
fi

if [ -z "$IP" ]; then
    error "Failed to get LoadBalancer IP address after 5 minutes. Please check the service status manually with 'kubectl describe service vllm-service'."
fi

info "Setup complete!"
info "The vLLM pod is now being created. It may take a significant amount of time to download the model."
info "Monitor the pod status with: kubectl get pods -l app=vllm-tpu --watch"
info "Once the pod is 'Running', you can send a test request with the following command:"
echo

echo "curl http://${IP}:8000/v1/completions -H 'Content-Type: application/json' -d '{\"model\": \"meta-llama/Llama-3.1-70B\", \"prompt\": \"San Francisco is a\", \"max_tokens\": 7, \"temperature\": 0}'"
echo
