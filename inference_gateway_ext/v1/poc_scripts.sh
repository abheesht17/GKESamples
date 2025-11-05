# Environment variables for your GKE Inference Gateway setup
export PROJECT_ID="gemle-gke-dev"
export REGION="us-central1"
export CLUSTER_NAME="inference-gateway-2"
export VPC_NETWORK_NAME="inference-gateway-2-net"
export PROXY_ONLY_SUBNET_NAME="proxy-only-subnet"
export PROXY_ONLY_SUBNET_RANGE="192.168.0.0/26"
export BUCKET="chavoshi-gkegmle"

# 2.1. Enable Inference Gateway & Get Credentials
# Enable Gateway API on the GKE cluster.
gcloud container clusters update ${CLUSTER_NAME} \
  --region ${REGION} \
  --project ${PROJECT_ID} \
  --gateway-api=standard

# Enable GcsFuseCsiDriver add-on on the GKE cluster.
gcloud container clusters update ${CLUSTER_NAME} \
  --region ${REGION} \
  --project ${PROJECT_ID} \
  --update-addons=GcsFuseCsiDriver=ENABLED

# Get credentials for the GKE cluster.
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION} --project ${PROJECT_ID}

# 2.2. Configure Internal Networking (Proxy-Only Subnet)
# Create a proxy-only subnet for the internal load balancer.
# Note: This command might fail if the subnet already exists.
gcloud compute networks subnets create ${PROXY_ONLY_SUBNET_NAME} \
  --purpose=REGIONAL_MANAGED_PROXY \
  --role=ACTIVE \
  --region ${REGION} \
  --network "${VPC_NETWORK_NAME}" \
  --range "${PROXY_ONLY_SUBNET_RANGE}" --project ${PROJECT_ID}

# 2.3. Install Inference Gateway CRDs from v1.1.0
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.1.0/manifests.yaml

# Delete the old Custom Metrics Stackdriver Adapter deployment
kubectl delete -f https://raw.githubusercontent.com/GoogleCloudPlatform/k8s-stackdriver/master/custom-metrics-stackdriver-adapter/deploy/production/adapter.yaml

# Deploy the correct Custom Metrics Stackdriver Adapter for Managed Prometheus
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/k8s-stackdriver/master/custom-metrics-stackdriver-adapter/deploy/production/adapter_new_resource_model.yaml

# Grant Monitoring Metric Writer role to the GKE node service account
# This is required for the gke-metrics-agent to write metrics to Cloud Monitoring
export GKE_NODE_SA="inference-gateway-2-gke-np-sa@gemle-gke-dev.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${GKE_NODE_SA}" \
    --role="roles/monitoring.metricWriter"

# Create a dedicated Google-managed IAM Service Account for the Custom Metrics Adapter
export CUSTOM_METRICS_IAM_SA="custom-metrics-sa@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud iam service-accounts create custom-metrics-sa --display-name "Custom Metrics Adapter Service Account" --project ${PROJECT_ID}

# Grant Monitoring Viewer role to the new custom-metrics-sa IAM Service Account
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${CUSTOM_METRICS_IAM_SA}" \
    --role="roles/monitoring.viewer"

# Grant the Kubernetes Service Account permission to use the IAM Service Account (Workload Identity)
gcloud iam service-accounts add-iam-policy-binding ${CUSTOM_METRICS_IAM_SA} \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${PROJECT_ID}.svc.id.goog[custom-metrics/custom-metrics-stackdriver-adapter]" \
    --project=${PROJECT_ID}

# Enable Workload Identity on the default node pool
gcloud container node-pools update default \
    --cluster=${CLUSTER_NAME} \
    --region=${REGION} \
    --workload-metadata=GKE_METADATA \
    --project=${PROJECT_ID}

# Enable Google Cloud Managed Service for Prometheus on the cluster
gcloud container clusters update ${CLUSTER_NAME} \
    --region=${REGION} \
    --enable-managed-prometheus \
    --project=${PROJECT_ID}

# Update the default node pool to increase max node count to 5
gcloud container node-pools update default \
    --cluster=${CLUSTER_NAME} \
    --region=${REGION} \
    --enable-autoscaling \
    --total-max-nodes=5 \
    --total-min-nodes=1 \
    --project=${PROJECT_ID}

# Restart the custom-metrics-stackdriver-adapter deployment to pick up Workload Identity
kubectl rollout restart deployment custom-metrics-stackdriver-adapter -n custom-metrics

# Deploy the model server (instruct) from release 1.1, with fixes
# 1. Set replicas to 1
# 2. Fix initContainer restartPolicy
# 3. Add toleration for NVIDIA GPU taint
curl -s https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/release-1.1/config/manifests/vllm/gpu-deployment.yaml | \
 sed 's/replicas: 3/replicas: 1/' | \
 sed 's/restartPolicy: IfNotPresent/restartPolicy: Always/' | \
 sed '/terminationGracePeriodSeconds:/a \
      tolerations:\n      - key: "nvidia.com/gpu"\
        operator: "Exists"\
        effect: "NoSchedule"' | \
 kubectl apply -f - 

# Install the InferencePool for the model (v1.1.0)
helm install vllm-llama3-8b-instruct \
  --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
  --set provider.name=gke \
  --set inferenceExtension.monitoring.gke.enabled=true \
  --version v1.1.0 \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool

# Create a load test pod
kubectl apply -f /usr/local/google/home/chavoshi/GKESamples/inference_gateway_ext/v1/load-test-pod.yaml

# To log into the 'load-test-pod' for interactive testing, run the following command:
# kubectl exec -it load-test-pod -- /bin/bash

# Once inside the 'load-test-pod', run these commands:
apt-get update && apt-get install -y curl hey # Install curl and hey load testing tools
export GW_IP="10.1.0.20"
# echo "Gateway Internal IP: ${GW_IP}"

# Test 1: Send a VALID Request
curl -X POST http://${GW_IP}/v1/completions \
-H "Content-Type: application/json" \
-d '{ 
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "What is the Llama 3.1 model?",
  "max_tokens": 50
}'

# Test 2: Send an INVALID Request
curl -X POST http://${GW_IP}/v1/completions \
-H "Content-Type: application/json" \
-d '{ 
  "model": "invalid-model-name",
  "prompt": "This request should fail.",
  "max_tokens": 50
}'

# Load Test: Generate sustained load to trigger HPA
hey -c 10 -z 1m -m POST \
-H "Content-Type: application/json" \
-d '{ 
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "Write a 100-word story about a robot.",
  "max_tokens": 128
}' http://${GW_IP}/v1/completions
