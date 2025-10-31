export PROJECT_ID=cloud-tpu-multipod-dev
export ZONE=us-central1-c

export RESOURCE_NAME=chavoshi-v7-inf
export NETWORK_NAME=${RESOURCE_NAME}-privatenetwork
export SUBNET_NAME=${RESOURCE_NAME}-privatesubnet
export NETWORK_FW_NAME=${RESOURCE_NAME}-privatefirewall
export ROUTER_NAME=${RESOURCE_NAME}-network
export NAT_CONFIG=${RESOURCE_NAME}-natconfig
export REGION=us-central1

# cluster 
export CLUSTER_NAME=chavoshi-v7-inf # Target cluster name
export GKE_VERSION=1.34.0-gke.2201000 # or later
export ACCELERATOR_TYPE=tpu7x-2x2x1 # Example:tpu7x-4x4x8, See topologies here 
export BASE_OUTPUT_DIR="gs://chavoshi-v7-inf" # Output directory for model training
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${SUBNET_NAME}"
export CPU_MACHINE_TYPE=n1-standard-8 # CPU machine type for system pods to land on


# Setup single NIC
gcloud compute networks create ${NETWORK_NAME} --mtu=8896 --project=${PROJECT_ID} --subnet-mode=custom --bgp-routing-mode=regional
gcloud compute networks subnets create "${SUBNET_NAME}" --network="${NETWORK_NAME}" --range=10.10.0.0/18 --region="${REGION}" --project=$PROJECT_ID
gcloud compute networks subnets create chavoshi-v7-inf-proxy-subnet \
    --purpose=REGIONAL_MANAGED_PROXY \
    --role=ACTIVE \
    --region=us-central1 \
    --network=chavoshi-v7-inf-privatenetwork \
    --range=10.12.0.0/24 \
    --project=cloud-tpu-multipod-dev
gcloud compute firewall-rules create ${NETWORK_FW_NAME} --network ${NETWORK_NAME} --allow tcp,icmp,udp --project=${PROJECT_ID}
gcloud compute routers create "${ROUTER_NAME}" \
  --project="${PROJECT_ID}" \
  --network="${NETWORK_NAME}" \
  --region="${REGION}"
gcloud compute routers nats create "${NAT_CONFIG}" \
  --router="${ROUTER_NAME}" \
  --region="${REGION}" \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges \
  --project="${PROJECT_ID}" \
  --enable-logging

# Set up uv 
sudo apt update
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Set up and Activate Python 3.12 virtual environment
uv venv --seed ./xpk_venv --python 3.12 --clear

source xpk_venv/bin/activate
pip install --upgrade pip
pip install xpk==0.14.2

## Download kueuectl - https://kueue.sigs.k8s.io/docs/reference/kubectl-kueue/installation/#installing-from-release-binaries
curl -Lo ./kubectl-kueue https://github.com/kubernetes-sigs/kueue/releases/download/v0.12.2/kubectl-kueue-linux-amd64
chmod +x ./kubectl-kueue
sudo mv ./kubectl-kueue /usr/local/bin/kubectl-kueue

## Download kjob - https://github.com/kubernetes-sigs/kjob/blob/main/docs/installation.md
curl -Lo ./kubectl-kjob https://github.com/kubernetes-sigs/kjob/releases/download/v0.1.0/kubectl-kjob-linux-amd64
chmod +x ./kubectl-kjob
sudo mv ./kubectl-kjob /usr/local/bin/kubectl-kjob

gcloud container clusters create ${CLUSTER_NAME} \
    --project=${PROJECT_ID} \
    --location=${REGION} \
    --cluster-version=${GKE_VERSION} \
    --machine-type=${CPU_MACHINE_TYPE} \
    --node-locations=${ZONE} \
    --enable-dataplane-v2 \
    --enable-ip-alias \
    --enable-multi-networking \
    --network=${NETWORK_NAME} \
    --subnetwork=${SUBNET_NAME} \
    --addons=HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver,GcsFuseCsiDriver \
    --enable-managed-prometheus \
    --workload-pool="cloud-tpu-multipod-dev.svc.id.goog" \
    --enable-image-streaming
gcloud container clusters update ${CLUSTER_NAME} \
    --region ${REGION} \
    --gateway-api=standard

# connect to cluster 
gcloud container clusters get-credentials chavoshi-v7-inf --region us-central1 --project cloud-tpu-multipod-dev

# Create the 1x1x1 node pool for vLLM
gcloud container node-pools create vllm-tpu-pool \
    --cluster=chavoshi-v7-inf \
    --project=cloud-tpu-multipod-dev \
    --region=us-central1 \
    --node-locations=us-central1-c \
    --machine-type "tpu7x-standard-1t" \
    --scopes "https://www.googleapis.com/auth/devstorage.full_control","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
    --reservation-affinity=specific \
    --reservation=cloudtpu-20251017124413-573252602 \
    --num-nodes "1"

# Deploy the vLLM workload
kubectl apply -f vllm-tpu.yaml

# --- Inference Gateway Setup ---

# 1. Install Inference Gateway CRDs
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.1.0/manifests.yaml

# 2. Apply RBAC for metrics
kubectl apply -f inference-gateway-rbac.yaml

# 3. Create the InferencePool
helm install vllm-llama3-1-8b-instruct \
  --set inferencePool.modelServers.matchLabels.app=vllm-tpu \
  --set provider.name=gke \
  --set inferenceExtension.monitoring.gke.enabled=true \
  --version v1.1.0 \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool

# 4. Create the Gateway
kubectl apply -f gateway.yaml

# 5. Create the HTTPRoute
kubectl apply -f httproute.yaml

# 6. Create the HealthCheckPolicy
kubectl apply -f health-check-policy.yaml


# --- Testing Commands ---

# Create a debug pod
kubectl apply -f debug-pod.yaml

# Install curl and hey in the debug pod
# kubectl exec -it debug-pod -- apt-get update && kubectl exec -it debug-pod -- apt-get install -y curl hey

# Command to send an inference request from inside the debug-pod
# Note: You may need to get the new gateway IP if it changes
IP=$(kubectl get gateway internal-gateway -o jsonpath='{.status.addresses[0].value}')
PORT=80

# check the tpu endpoint
kubectl exec -it debug-pod -- curl -X POST http://${IP}:${PORT}/tpu/v1/completions  \
-H "Content-Type: application/json" \
-H "X-Gateway-Model-Name: meta-llama/Llama-3.1-8B-Instruct" \
-d '{ 
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "What is the capital of France?",
    "max_tokens": 50,
    "temperature": "0.7"
}'

# check the cpu end point
kubectl exec -it debug-pod -- curl -X POST http://${IP}:${PORT}/cpu/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-Gateway-Model-Name: Qwen/Qwen2.5-1.5B-Instruct" \
  -d '{ 
      "model": "Qwen/Qwen2.5-1.5B-Instruct",
      "prompt": "What is the capital of France?",
      "max_tokens": 100,
      "temperature": "0.7"
  }'; 


# install infernece pool 
helm install vllm-tpu-inferencepool \
  --set inferencePool.modelServers.matchLabels.app=vllm-tpu \
  --set provider.name=gke \
  --set inferenceExtension.monitoring.gke.enabled=true \
  --version v1.0.1 \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool


kubectl patch inferencepool vllm-tpu-inferencepool --type='json' -p='[{"op": "replace", "path": "/spec/endpointPickerRef/failureMode", "value": "FailOpen"}]'
