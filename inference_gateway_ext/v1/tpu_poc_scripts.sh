export PROJECT_ID="tpu-prod-env-one-vm"
export CLUSTER_REGION="us-east5"
export CLUSTER_ZONE="us-east5-b"
export CLUSTER_NAME="chavoshi-benchmark-us-east5b"
export GCS_BUCKET_NAME="chavoshi-dlrm-training"
export KSA_NAME="vllm-tpu-ksa"
export NAMESPACE="vllm-tpu-ns"

# Install the InferencePool
helm install vllm-tpu-inferencepool \
  --namespace ${NAMESPACE} \
  --set inferencePool.modelServers.matchLabels.app=vllm-tpu \
  --set provider.name=gke \
  --set inferenceExtension.monitoring.gke.enabled=true \
  --version v1.1.0 \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool