#!/bin/bash
# E2E POC Validation Script



########## GPU demo section ###############
# Connect to the inference-gateway-2 cluster
gcloud container clusters get-credentials inference-gateway-2 --region us-central1 --project gemle-gke-dev


### CUJ 1 & 2
kubectl get gateway
kubectl get pods | grep instruct
kubectl get inferencepools | grep instruct

# Get the Internal Gateway IP
export GW_IP=$(kubectl get gateway internal-gateway -n default -o jsonpath='{.status.addresses[0].value}')
echo "Gateway Internal IP: $GW_IP"


# Connect to validation POD
kubectl exec -it load-test-pod -- /bin/bash

# In Validation POD run the following
apt-get update && apt-get install -y curl hey # Install curl and hey load testing tools

# Validate Gateway in cluster 2
export GW_IP=10.1.0.20
# Send a valid request to the model server
curl -X POST http://${GW_IP}/v1/completions \
-H "Content-Type: application/json" \
-d '{ 
"model": "meta-llama/Llama-3.1-8B-Instruct",
"prompt": "What is the Llama 3.1 model?",
"max_tokens": 50
}'

# Send an invalid request to the model server
curl -X POST http://${GW_IP}/v1/completions \
-H "Content-Type: application/json" \
-d '{ 
"model": "invalid-model-name",
"prompt": "This request should fail.",
"max_tokens": 50
}'

# Watch logs in other window 
kubectl logs -l app=vllm-llama3-8b-instruct -f  | grep -v -e "/health" -e "/metrics" -e "/v1/models" 
kubectl logs -l app=vllm-llama3-8b-canary -f  | grep -v -e "/health" -e "/metrics" -e "/v1/models"	

### CUJ 5
# Validate traffic splitting
for i in {1..100}; do \ 
printf "\n Request number: $i \n\n"; 
curl -X POST http://${GW_IP}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{ 
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "What is the Llama 3.1 model?",
  "max_tokens": 50
  }'; \ 
done


########## TPU demo section ###############
# Set environment variables for TPU deployment
export PROJECT_ID="gemle-gke-dev"
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
export CLUSTER_NAME=inference-gateway-1
export CONTROL_PLANE_LOCATION=us-east5-b
export ZONE=us-east5-b
export CLUSTER_VERSION=1.31.2-gke.1115000
export GSBUCKET=tpu-model-cache-1761674247
export KSA_NAME=vllm-tpu-ksa
export NAMESPACE=vllm-tpu-ns
export HF_TOKEN=

# Switch to the TPU cluster
gcloud container clusters get-credentials inference-gateway-1 --zone us-east5-b --project gemle-gke-dev


# generate load 
export GW_IP=34.144.191.173
hey -c 10 -z 1m -m POST \
  -H "Content-Type: application/json" \
  -d '{ \ 
  "model": "/data/lama3model-bucket/Llama-3.1-70B-Instruct", \ 
  "prompt": "Write a 100-word story about a robot.", \ 
  "max_tokens": 128 \ 
  }' http://${GW_IP}:8000/v1/completions
















####### Other reference commands used ###########
# Check for enabled addons
gcloud container clusters describe inference-gateway-1 --zone us-east5-b --project gemle-gke-dev --format="value(addonsConfig.gatewayControllerManagerConfig, addonsConfig.gcsFuseCsiDriverConfig)"

# Enable the GcsFuseCsiDriver addon
gcloud container clusters update inference-gateway-1 --zone us-east5-b --project gemle-gke-dev --update-addons=GcsFuseCsiDriver=ENABLED

# Check for a proxy-only subnet
gcloud compute networks subnets list --filter="purpose=REGIONAL_MANAGED_PROXY" --project=gemle-gke-dev

# Create a proxy-only subnet
gcloud compute networks subnets create proxy-only-subnet-us-east5 \
--purpose=REGIONAL_MANAGED_PROXY \
--role=ACTIVE \
--region=us-east5 \
--network=default \
--range=192.168.251.0/24

# Install Inference Gateway CRDs
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.1.0/manifests.yaml


# Create Kubernetes namespace
kubectl create namespace ${NAMESPACE}

# Create Kubernetes Secret for Hugging Face credentials
kubectl create secret generic hf-secret \
    --from-literal=hf_api_token=${HF_TOKEN} \
    --namespace ${NAMESPACE}

# Create a Cloud Storage bucket
gcloud storage buckets create gs://${GSBUCKET} \
    --uniform-bucket-level-access

# Set up a Kubernetes ServiceAccount to access the bucket
kubectl create serviceaccount ${KSA_NAME} --namespace ${NAMESPACE}

gcloud storage buckets add-iam-policy-binding gs://${GSBUCKET} \
  --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${KSA_NAME}" \
  --role "roles/storage.objectUser"

# Pre-download model and upload to GCS for stability
echo "Downloading model from Hugging Face and uploading to GCS..."
apt-get update && apt-get install -y git-lfs
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
gcloud storage cp --recursive ./Llama-3.1-8B-Instruct gs://${GSBUCKET}/Llama-3.1-8B-Instruct

# Deploy the vLLM model server
kubectl apply -f /usr/local/google/home/chavoshi/GKESamples/inference_gateway_ext/v1/vllm-llama3-8b.yaml -n ${NAMESPACE}

# View the logs from the running model server
kubectl logs -f -l app=vllm-tpu -n ${NAMESPACE}


# Create InferencePool (for KV-Cache Routing)
helm install vllm-tpu-inferencepool \
--namespace ${NAMESPACE} \
--set inferencePool.modelServers.matchLabels.app=vllm-tpu \
--set provider.name=gke \
--set inferenceExtension.monitoring.gke.enabled=true \
--version v1.1.0 \
oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool

# Apply the updated HTTPRoute
kubectl apply -f /usr/local/google/home/chavoshi/GKESamples/inference_gateway_ext/v1/main-inference-routes.yaml

# Get the Internal Gateway IP for TPU cluster
export GW_IP_TPU=$(kubectl get gateway internal-gateway -n default -o jsonpath='{.status.addresses[0].value}')
echo "TPU Gateway Internal IP: $GW_IP_TPU"

# Send a test request to the TPU model server
curl -X POST http://${GW_IP_TPU}/v1/completions \
-H "Content-Type: application/json" \
-d '{ 
"model": "meta-llama/Llama-3.1-8B-Instruct",
"prompt": "What is the Llama 3.1 model?",
"max_tokens": 50
}'

# --- Dynamic Model Rollout (Canary) ---
# Create the canary deployment YAML
curl -sL https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/release-1.0/config/manifests/vllm/gpu-deployment.yaml | \
sed -e 's/name: vllm-llama3-8b-instruct/name: vllm-llama3-8b-canary/g' \
    -e 's/app: vllm-llama3-8b-instruct/app: vllm-llama3-8b-canary/g' \
    -e 's/replicas: 3/replicas: 1/g' \
    -e 's/image: "vllm\/vllm-openai:v0.8.5"/image: "vllm\/vllm-openai:latest"/g' \
    -e 's/name: vllm-llama3-8b-instruct-adapters/name: vllm-llama3-8b-canary-adapters/g' \
    -e 's/id: food-review-1/id: food-review-canary/g' > /usr/local/google/home/chavoshi/GKESamples/inference_gateway_ext/v1/vllm-llama3-8b-canary.yaml

kubectl logs -l app=vllm-llama3-8b-instruct -f | grep -v -e "/health" -e "/metrics" -e "/v1/models"

# Validate traffic splitting
for i in {1..100}; do \ 
printf "\n Request number: $i \n\n"; 
curl -X POST http://${GW_IP}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{ 
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "What is the Llama 3.1 model?",
  "max_tokens": 50
  }'; \ 
done

# --- TPU Model Validation ---
export GW_IP=10.202.15.203 && curl -X POST http://${GW_IP}/v1/completions -H "Content-Type: application/json" -d '{ "model": "/data/lama3model-bucket/Llama-3.1-70B-Instruct", "prompt": "What is the Llama 3.1 model?", "max_tokens": 50 }'

# --- Direct Service Validation (TPU) ---
export NAMESPACE=vllm-tpu-ns
export VLLM_SERVICE_IP=$(kubectl get service vllm-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "VLLM Service IP: ${VLLM_SERVICE_IP}"

# Load test to trigger HPA
export GW_IP=$(kubectl get service vllm-service -n vllm-tpu-ns -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export GW_IP=34.144.191.173
hey -c 10 -z 1m -m POST \
  -H "Content-Type: application/json" \
  -d '{ \ 
  "model": "/data/lama3model-bucket/Llama-3.1-70B-Instruct", \ 
  "prompt": "Write a 100-word story about a robot.", \ 
  "max_tokens": 128 \ 
  }' http://${GW_IP}:8000/v1/completions
