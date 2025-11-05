#!/bin/bash

# --- Environment Variables ---
export PROJECT_ID=cloud-tpu-multipod-dev
export REGION=us-central1
export ZONE=us-central1-c

# --- Network Configuration (Assumes Existing Network) ---
export RESOURCE_NAME=chavoshi-v7-inf
export NETWORK_NAME=${RESOURCE_NAME}-privatenetwork
export SUBNET_NAME=${RESOURCE_NAME}-privatesubnet

# --- New Cluster Configuration ---
export CLUSTER_NAME=chavoshi-v7-inf-v3 # New cluster name to avoid conflict
export GKE_VERSION=1.34.1-gke.2037000 # Changed from  GKE_VERSION=1.34.0-gke.2201000 due to availablity.
export CPU_MACHINE_TYPE=c2-standard-16 #n1-standard-8

# Inference Gateway 
export RELEASE=v1.0.1
export STABLE_MODEL_RELEASE=v0.3.0

echo "--- Creating GKE Cluster: ${CLUSTER_NAME} ---"

# --- Create GKE Cluster (without --enable-multi-networking) ---
gcloud container clusters create ${CLUSTER_NAME} \
    --project=${PROJECT_ID} \
    --location=${REGION} \
    --cluster-version=${GKE_VERSION} \
    --machine-type=${CPU_MACHINE_TYPE} \
    --node-locations=${ZONE} \
    --enable-dataplane-v2 \
    --enable-ip-alias \
    --network=${NETWORK_NAME} \
    --subnetwork=${SUBNET_NAME} \
    --addons=HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver,GcsFuseCsiDriver \
    --workload-pool="cloud-tpu-multipod-dev.svc.id.goog"

gcloud container clusters create ${CLUSTER_NAME} \
  --cluster-version=${GKE_VERSION} \
  --machine-type=${CPU_MACHINE_TYPE} \
  --location=${REGION} \
  --node-locations=${ZONE} \
  --project=${PROJECT_ID} \
  --enable-dataplane-v2 \
  --enable-ip-alias \
  --enable-multi-networking \
  --network=${NETWORK_NAME} \
  --subnetwork=${SUBNET_NAME} \
  --addons=HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver,GcsFuseCsiDriver \
  --workload-pool=$PROJECT_ID.svc.id.goog


# --- Enable Gateway API ---
gcloud container clusters update ${CLUSTER_NAME} \
    --project=${PROJECT_ID} \
    --region ${REGION} \
    --gateway-api=standard

# --- Get Cluster Credentials ---
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION} --project ${PROJECT_ID}


# --- Installing Gateway Components --- #
# Install CRDs 
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/$RELEASE/config/crd/bases/inference.networking.x-k8s.io_inferenceobjectives.yaml


# Deploy model server 
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/$STABLE_MODEL_RELEASE/config/manifests/vllm/cpu-deployment.yaml 
# confirm model server are ready
kubectl wait deployment/vllm-llama3-8b-instruct --for=condition=Available --timeout=5m

# Deploy the InferencePool and Endpoint Picker Extension 

kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/$RELEASE/config/manifests/inferencepool-resources.yaml

kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/$RELEASE/config/manifests/gateway/gke/healthcheck.yaml

kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/$RELEASE/config/manifests/gateway/gke/gcp-backend-policy.yaml

# Deploy inferenceobjective  
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/$RELEASE/config/manifests/inferenceobjective.yaml

# Deploy httproute  
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/$RELEASE/config/manifests/gateway/gke/httproute.yaml
#Confirm that the HTTPRoute status conditions include Accepted=True and #ResolvedRefs=True:

 kubectl wait httproute/llm-route \
--for=jsonpath='{.status.parents[0].conditions[?(@.type=="Accepted")].status}'=True \
--for=jsonpath='{.status.parents[0].conditions[?(@.type=="ResolvedRefs")].status}'=True \
--timeout=5m


# Deploy gateway
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/$RELEASE/config/manifests/gateway/gke/gateway.yaml

# Confirm that the Gateway was assigned an IP address and reports a Programmed=True status:

$ kubectl wait gateway/inference-gateway \
--for=jsonpath='{.status.addresses[0].value}' \
--for=condition=Programmed \
--timeout=5m

# Create a debug pod
kubectl apply -f debug-pod.yaml

# Install curl and hey in the debug pod
kubectl exec -it debug-pod -- apt-get update && kubectl exec -it debug-pod -- apt-get install -y curl hey


IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}')
echo $IP
PORT=80

# check the tpu endpoint
kubectl exec -it debug-pod -- curl -i ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{
"model": "Qwen/Qwen2.5-1.5B-Instruct",
"prompt": "Write as if you were a critic: San Francisco",
"max_tokens": 10,
"temperature": 0
}'

# --- Fix for Gateway Health Check Issue ---
# This firewall rule allows ingress traffic from the proxy-only subnet (10.12.0.0/24)
# to the cluster nodes on TCP port 8000, enabling the Gateway's health checks.
gcloud compute firewall-rules create allow-gateway-health-checks \
    --project=${PROJECT_ID} \
    --network=${NETWORK_NAME} \
    --action=ALLOW \
    --direction=INGRESS \
    --source-ranges=10.12.0.0/24 \
    --target-tags=$(gcloud compute instances list --project=${PROJECT_ID} --filter="name~gke-${CLUSTER_NAME}" --limit=1 --format='value(tags.items[0])') \
    --rules=tcp:8000


# Patch the inference pool to fail open 
kubectl patch inferencepool vllm-llama3-8b-instruct --type='json' -p='[{"op": "replace", "path": "/spec/endpointPickerRef/failureMode", "value": "FailOpen"}]'

echo "--- Cluster setup script finished ---"

# --- Diagnostic Commands ---

# 1. Check the HTTPRoute status.
# Look for 'Accepted' and 'ResolvedRefs' conditions to be 'True'.
# An empty status means the gateway hasn't processed the route.
echo "--- Checking HTTPRoute status ---"
kubectl get httproute llm-route -o yaml

# 2. Check the Gateway status.
# Ensure the gateway exists and has an IP address assigned.
# The 'llm-route' HTTPRoute needs this gateway to attach.
echo "--- Checking Gateway status ---"
kubectl get gateway inference-gateway -o yaml

# 3. Check the InferencePool configuration.
# This confirms the pool is targeting the correct pods via its label selector.
echo "--- Checking InferencePool configuration ---"
kubectl get inferencepool vllm-llama3-8b-instruct -o yaml

# 4. Check the backend model server pods.
# Verify that pods with the correct label are 'Running' and 'Ready'.
echo "--- Checking backend pod status ---"
kubectl get pods -l app=vllm-llama3-8b-instruct -o wide

# 5. Check the HealthCheckPolicy.
# This policy tells the gateway how to check if the backend pods are healthy.
# It should be 'Attached'.
echo "--- Checking HealthCheckPolicy ---"
kubectl get healthcheckpolicy -o yaml

# 6. Check the logs of a model server pod.
# Look for errors related to model loading or the web server.
# Replace with a real pod name from the previous command.
echo "--- Checking pod logs (replace with a real pod name) ---"
# kubectl logs vllm-llama3-8b-instruct-xxxxxxxx-xxxxx

# 7. Check firewall rules.
# This is to ensure that health checks from the proxy-only subnet (10.12.0.0/24)
# are allowed to reach the pods on the health check port (e.g., tcp:8000).
echo "--- Checking firewall rules for health checks ---"
gcloud compute firewall-rules list --project=cloud-tpu-multipod-dev --filter="direction=INGRESS AND sourceRanges:10.12.0.0/24" --format="json(name, allowed, sourceRanges, targetTags)"

# 8. Get the Gateway IP address.
# This is needed to identify the associated load balancer resources.
echo "--- Getting Gateway IP address ---"
kubectl get gateway inference-gateway -o jsonpath='{.status.addresses[0].value}'

# 9. List regional URL maps.
# Look for a URL map associated with the 'inference-gateway'.
echo "--- Listing regional URL maps ---"
gcloud compute url-maps list --project=${PROJECT_ID} --filter="region:${REGION}" --format="json(name, defaultService, hostRules)"

# 10. Describe the URL map to get the backend service name.
# This helps identify the backend service that the URL map is routing traffic to.
echo "--- Describing URL map to get backend service name (replace with actual URL map name) ---"
# gcloud compute url-maps describe <URL_MAP_NAME> --region=${REGION} --project=${PROJECT_ID} --format="json(pathMatchers[0].defaultService)"

# 11. Check the health of the backend service.
# This directly shows if the load balancer considers the backend instances healthy.
echo "--- Checking backend service health (replace with actual backend service name) ---"
# gcloud compute backend-services get-health <BACKEND_SERVICE_NAME> --region=${REGION} --project=${PROJECT_ID} --format="json"

