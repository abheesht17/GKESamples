# Following instructions @ https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-vllm-tpu


export PROJECT_ID="tpu-prod-env-one-vm"
export CLUSTER_NAME="chavoshi-benchmark-us-east5b"
export ZONE="us-east5-b"
export GSBUCKET="chavoshi_general_purpose"
export CONTROL_PLANE_LOCATION="us-east5-b"
export CLUSTER_VERSION="1.31.2-gke.1115000"
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")


# A descriptive name for the Kubernetes Service Account
export KSA_NAME="vllm-ksa"
export NAMESPACE="vllm-serving"

# This sets the project for all subsequent gcloud commands
gcloud config set project ${PROJECT_ID}

# Connect to GKE cluster 
gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${CONTROL_PLANE_LOCATION}


# Node pool create comamand 
# gcloud container node-pools create tpunodepool \
#     --location=${CONTROL_PLANE_LOCATION} \
#     --node-locations=${ZONE} \
#     --num-nodes=1 \
#     --machine-type=ct6e-standard-8t \
#     --cluster=${CLUSTER_NAME} \
#     --enable-autoscaling --total-min-nodes=1 --total-max-nodes=2


#!/bin/bash
# N=30
# export vllm_service=$(kubectl get service vllm-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' -n ${NAMESPACE})
# for i in $(seq 1 $N); do
#   while true; do
#     curl http://$vllm_service:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "meta-llama/Llama-3.1-70B", "prompt": "Write a story about san francisco", "max_tokens": 1000, "temperature": 0}'
#   done &  # Run in the background
# done
# wait
