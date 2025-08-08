#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
export PROJECT_ID="chavoshi-gke-dev"
export REGION="us-east5"
export AR_REPO_NAME="tpu-repo"
export BUCKET_NAME="dlrm-training-chavoshi-gke-dev"

# Define a new name for the preprocessing container image
export PREPROCESS_IMAGE_NAME="dlrm-preprocess"

echo ">>> 🚀 Starting parallel preprocessing job deployment..."

# 1. Delete the previous job if it exists to ensure a clean run
echo "1. Deleting previous preprocessing job if it exists..."
kubectl delete job dlrm-preprocess-job --ignore-not-found=true

# 2. Build and push the container using the new Dockerfile
echo "2. Building and pushing the preprocessing container..."
docker build -f Dockerfile.preprocess -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${PREPROCESS_IMAGE_NAME}:latest .
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${PREPROCESS_IMAGE_NAME}:latest

# 3. Apply the Kubernetes Job YAML
# 'envsubst' replaces the variables like ${BUCKET_NAME} in the YAML file
echo "3. Applying Kubernetes job YAML..."
envsubst < job_preprocess.yaml | kubectl apply -f -

echo "✅ Job 'dlrm-preprocess-job' submitted."
sleep 10
kubectl get pods
sleep 5
kubectl logs -f jobs/dlrm-preprocess-job 
