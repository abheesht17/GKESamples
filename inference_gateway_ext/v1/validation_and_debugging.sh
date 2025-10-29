#!/bin/bash

# This script contains a series of commands to validate the setup of a GKE Inference Gateway,
# manage its network settings, and recreate the entire environment from scratch.

# --- Configuration ---
# Set these variables from the command line or edit them here.
PROJECT=${1:-"apigee-ai"}
REGION=${2:-"us-central1"}
CLUSTER=${3:-"infgw-cluster-01"}
NETWORK="$CLUSTER-net"
SUBNET="$CLUSTER-subnet"
PROXY_SUBNET="$CLUSTER-proxy-subnet"

echo "Using Project: $PROJECT, Region: $REGION, Cluster: $CLUSTER"

# --- Validation Commands ---

# Describe the GKE cluster to check its version and other configuration details.
echo "--- Describing GKE Cluster ---"
gcloud container clusters describe $CLUSTER --region $REGION --project $PROJECT

# Get cluster credentials and check for the presence of Inference Gateway CRDs.
echo "--- Checking for Inference Gateway CRDs ---"
gcloud container clusters get-credentials $CLUSTER --region $REGION --project $PROJECT
kubectl get crds | grep inference

# Check for the model server deployment using its label.
echo "--- Checking for model server deployment by label ---"
kubectl get deployment -l app=vllm-llama3-8b-instruct

# List all deployments in the cluster to find the model server if the label search fails.
echo "--- Listing all deployments in the cluster ---"
kubectl get deployments --all-namespaces

# Get the detailed YAML configuration of the model server deployment.
echo "--- Getting model server deployment details ---"
kubectl get deployment vllm-llama3-8b-instruct -o yaml

# Check the status of the model server pods.
echo "--- Checking model server pod status ---"
kubectl get pods -l app=vllm-llama3-8b-instruct -o wide

# Get the details of the InferencePool custom resource.
echo "--- Getting InferencePool details ---"
kubectl get inferencepool vllm-llama3-8b-instruct -o yaml

# List the InferenceObjective resources.
echo "--- Listing InferenceObjective resources ---"
kubectl get inferenceobjective

# Get the details of the Gateway resource.
echo "--- Getting Gateway details ---"
kubectl get gateway inference-gateway -o yaml

# List the HTTPRoute resources.
echo "--- Listing HTTPRoute resources ---"
kubectl get httproute

# Get the details of the HTTPRoute.
echo "--- Getting HTTPRoute details ---"
kubectl get httproute httproute -o yaml

# Inspect the LoRA adapter ConfigMap.
echo "--- Inspecting the LoRA adapter ConfigMap ---"
kubectl get configmap vllm-llama3-8b-instruct-adapters -o yaml

# Send a test inference request to the Gateway with the correct model ID.
echo "--- Sending test inference request ---"
IP=$(kubectl get gateway inference-gateway -o jsonpath='{.status.addresses[0].value}')
PORT="80"
echo "Gateway IP: $IP"
curl -i -X POST ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -H "Authorization: Bearer $(gcloud auth print-access-token)" -d '{
    "model": "food-review-1",
    "prompt": "What is the best pizza in the world?",
    "max_tokens": 2048,
    "temperature": "0"
}'

# --- Master Authorized Networks Management ---

# Disable master authorized networks (allows access from any IP).
echo "--- Disabling Master Authorized Networks ---"
gcloud container clusters update $CLUSTER --region $REGION --project $PROJECT --no-enable-master-authorized-networks

# Enable master authorized networks and restrict to your current IP.
echo "--- Enabling Master Authorized Networks (Current IP) ---"
MY_IP=$(curl -s ifconfig.me)
echo "Your current IP is: $MY_IP"
gcloud container clusters update $CLUSTER --region $REGION --project $PROJECT --enable-master-authorized-networks --master-authorized-networks "${MY_IP}/32"


# --- Cluster and Configuration Recreation (VALIDATED) ---
# Note: Run these commands from a clean environment. They are not designed to be idempotent.

echo "--- Starting Full Re-creation Process ---"

# 1. Create the VPC Network
echo "--- Creating VPC Network ---"
gcloud compute networks create $NETWORK --project=$PROJECT --subnet-mode=custom --mtu=8244 --bgp-routing-mode=global

# 2. Create the primary GKE Subnet
echo "--- Creating Primary GKE Subnet ---"
gcloud compute networks subnets create $SUBNET --project=$PROJECT --range=10.1.0.0/20 --network=$NETWORK --region=$REGION

# 3. Create the Proxy-Only Subnet for the Gateway
echo "--- Creating Proxy-Only Subnet ---"
gcloud compute networks subnets create $PROXY_SUBNET --project=$PROJECT --purpose=REGIONAL_MANAGED_PROXY --role=ACTIVE --region=$REGION --network=$NETWORK --range=10.1.16.0/24

# 4. Create the GKE Cluster
echo "--- Creating GKE Cluster (this may take several minutes) ---"
gcloud container clusters create $CLUSTER --project=$PROJECT --region $REGION --network "projects/$PROJECT/global/networks/$NETWORK" --subnetwork "projects/$PROJECT/regions/$REGION/subnetworks/$SUBNET" \
--cluster-version "1.32.9-gke.1072000" \
--machine-type "e2-standard-2" --num-nodes "1" --enable-autoscaling --min-nodes "1" --max-nodes "3" \
--enable-shielded-nodes --shielded-secure-boot --shielded-integrity-monitoring \
--enable-ip-alias --cluster-secondary-range-name "pods" --services-secondary-range-name "services" \
--cluster-ipv4-cidr "10.4.0.0/14" --services-ipv4-cidr "10.0.32.0/20" \
--enable-private-nodes --enable-private-endpoint --master-ipv4-cidr "172.16.0.32/28" \
--enable-master-authorized-networks --master-authorized-networks "0.0.0.0/0" \
--addons HttpLoadBalancing,GcePersistentDiskCsiDriver \
--gateway-api "standard" --datapath-provider "ADVANCED_DATAPATH" --enable-multi-networking \
--workload-pool "$PROJECT.svc.id.goog" \
--autoscaling-profile "OPTIMIZE_UTILIZATION" \
--logging "SYSTEM,WORKLOAD" --monitoring "SYSTEM,POD,DAEMONSET,DEPLOYMENT,STATEFULSET,STORAGE,HPA,CADVISOR,KUBELET" \
--enable-managed-prometheus \
--maintenance-window-start "09:00" --maintenance-window-end "13:00" --maintenance-window-recurrence "FREQ=DAILY" \
--no-enable-autoupgrade --no-enable-autorepair

# 5. Add the System Node Pool
echo "--- Adding System Node Pool ---"
gcloud container node-pools create system --project $PROJECT --region $REGION --cluster $CLUSTER \
--machine-type "e2-standard-4" --num-nodes "2" --enable-autoscaling --min-nodes "2" --max-nodes "10"

# 6. Add the A3 GPU Node Pool
echo "--- Adding A3 GPU Node Pool ---"
gcloud container node-pools create a3-highgpu-8g-a3highgpupool --project $PROJECT --region $REGION --cluster $CLUSTER \
--machine-type "a3-highgpu-8g" --accelerator "type=nvidia-h100-80gb,count=8,gpu-driver-version=default" \
--spot --num-nodes 2 --no-enable-autoscaling \
--node-taints "nvidia.com/gpu=present:NoSchedule"

# 7. Get Cluster Credentials
echo "--- Getting Credentials for New Cluster ---"
gcloud container clusters get-credentials $CLUSTER --region $REGION --project $PROJECT

# 8. Install Inference Gateway CRDs
echo "--- Installing Inference Gateway CRDs ---"
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.0.0/manifests.yaml

# 9. Create Hugging Face Token Secret
echo "--- Creating Hugging Face Secret (enter your token when prompted) ---"
read -sp "Enter your Hugging Face Token: " HF_TOKEN
kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN

# 10. Deploy the vLLM Model Server
echo "--- Deploying vLLM Model Server ---"
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/release-1.0/config/manifests/vllm/gpu-deployment.yaml

# 11. Create the LoRA Adapter ConfigMap
echo "--- Creating LoRA Adapter ConfigMap ---"
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-llama3-8b-instruct-adapters
data:
  configmap.yaml: |
    vLLMLoRAConfig:
      name: vllm-llama3-8b-instruct-adapters
      port: 8000
      defaultBaseModel: meta-llama/Llama-3.1-8B-Instruct
      ensureExist:
        models:
        - id: food-review-1
          source: Kawon/llama3.1-food-finetune_v14_r8
EOF

# 12. Install InferencePool via Helm
echo "--- Installing InferencePool and Endpoint Picker ---"
helm install vllm-llama3-8b-instruct \
  --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
  --set provider.name=gke \
  --set inferenceExtension.monitoring.gke.enabled=true \
  --version v1.0.1 \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool

# 13. Create InferenceObjectives
echo "--- Creating InferenceObjectives ---"
cat <<EOF | kubectl apply -f -
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceObjective
metadata:
  name: food-review
spec:
  priority: 10
  poolRef:
    name: vllm-llama3-8b-instruct
    group: "inference.networking.k8s.io"
---
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceObjective
metadata:
  name: llama3-base-model
spec:
  priority: 20 # Higher priority
  poolRef:
    name: vllm-llama3-8b-instruct
EOF

# 14. Create the Gateway
echo "--- Creating the Gateway ---"
cat <<EOF | kubectl apply -f -
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: inference-gateway
spec:
  gatewayClassName: gke-l7-regional-external-managed
  listeners:
    - protocol: HTTP
      port: 80
      name: http
EOF

# 15. Create the HTTPRoute
echo "--- Creating the HTTPRoute ---"
cat <<EOF | kubectl apply -f -
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: httproute
spec:
  parentRefs:
  - name: inference-gateway
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: "/"
    backendRefs:
    - name: vllm-llama3-8b-instruct
      group: "inference.networking.k8s.io"
      kind: InferencePool
EOF

echo "--- Recreation Script Finished ---"
echo "Note: It may take several minutes for the Gateway to get an IP address and become fully operational."
