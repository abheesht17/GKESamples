#!/bin/bash

# This command ensures that the script will exit immediately if any command fails.
set -e

## ------------------- Configuration ------------------- ##
# Set environment variables from your setup.
export PROJECT_ID="chavoshi-gke-dev"
export REGION="us-east5"
export CLUSTER_NAME="tpu-cluster"
export AR_REPO_NAME="tpu-repo"
export IMAGE_TAG="latest"
export BUCKET_NAME="dlrm-training-${PROJECT_ID}"
export IMAGE_NAME="jax-dlrm-gke"

# Color definitions for echo statements
ORANGE='\033[38;5;208m'
NC='\033[0m' # No Color

## ----------------- Input Validation ------------------ ##
# Check if an input parameter for the job type was provided.
if [ -z "$1" ]; then
  echo -e "${ORANGE}🛑 Error: No job type specified.${NC}"
  echo -e "${ORANGE}Usage: $0 [v6e_32 | v6e_4]${NC}"
  echo -e "${ORANGE}  v6e_32: Runs the 32-chip benchmark job.${NC}"
  echo -e "${ORANGE}  v6e_4:  Runs the 4-chip single-node test job.${NC}"
  exit 1
fi

JOB_TYPE=$1

## ----------- Docker Build & Push (Always Runs) ----------- ##
echo -e "${ORANGE}🚀 Building and pushing the Docker image. This will run every time.${NC}"
DOCKER_IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# The docker build output will not be ORANGE, which is desired.
docker build -f Dockerfile -t ${DOCKER_IMAGE_URL} .
docker push ${DOCKER_IMAGE_URL}

echo -e "${ORANGE}✅ Docker image pushed successfully to ${DOCKER_IMAGE_URL}${NC}"

## ------------- Selective Job Deployment, Cleanup, and Logging -------------- ##
echo -e "${ORANGE}▶️ Starting process for selected job: ${JOB_TYPE}${NC}"

# Use a case statement to select the correct actions based on the input parameter.
case "$JOB_TYPE" in
  v6e_32)
    # --- Cleanup for v6e_32 ---
    echo -e "${ORANGE}🧹 Cleaning up pre-existing 'jax-dlrm-benchmark' job...${NC}"
    kubectl delete jobset jax-dlrm-benchmark-v6e-32chip --ignore-not-found=true --wait=false

    echo -e "${ORANGE}🔎 Final pod status check...${NC}"
    kubectl get pods
    
    # --- Deploy v6e_32 ---
    echo -e "${ORANGE}🚢 Deploying 'v6e_32' job...${NC}"
    envsubst < jobset_jax_v6e_32.yaml | kubectl apply -f -
    echo -e "${ORANGE}✅ Job 'v6e_32' submitted successfully.${NC}"

    # --- Log for v6e_32 ---
    echo -e "${ORANGE}📝 Tailing logs for 'jax-dlrm-benchmark-v6e-32chip-worker-0'. Press Ctrl+C to stop.${NC}"
    # Adding a brief sleep to allow pods to initialize before checking logs
    sleep 10
    kubectl logs -f jobs/jax-dlrm-benchmark-v6e-32chip-worker-0
    ;;

  v6e_4)
    # --- Cleanup for v6e_4 ---
    echo -e "${ORANGE}🧹 Cleaning up pre-existing 'jax-dlrm-singlenode-test' job...${NC}"
    kubectl delete jobset jax-dlrm-singlenode-test --ignore-not-found=true --wait=false

    echo -e "${ORANGE}🔎 Final pod status check...${NC}"
    kubectl get pods

    # --- Deploy v6e_4 ---
    echo -e "${ORANGE}🚢 Deploying 'v6e_4' job...${NC}"
    envsubst < jobset_jax_singlenode_4chip.yaml | kubectl apply -f -
    echo -e "${ORANGE}✅ Job 'v6e_4' submitted successfully.${NC}"

    # --- Log for v6e_4 ---
    echo -e "${ORANGE}📝 Tailing logs for 'jobs/jax-dlrm-singlenode-test-worker-0'. Press Ctrl+C to stop.${NC}"
    # Adding a brief sleep to allow pods to initialize before checking logs
    sleep 10
    kubectl logs -f jobs/jax-dlrm-singlenode-test-worker-0
    ;;

  *)
    echo -e "${ORANGE}🛑 Error: Invalid job type '${JOB_TYPE}'. Please use 'v6e_32' or 'v6e_4'.${NC}" >&2
    exit 1
    ;;
esac


echo -e "${ORANGE}✅ Script finished.${NC}"
