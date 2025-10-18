#!/bin/bash
set -e

# --- Color definitions ---
ORANGE='\033[38;5;208m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

## ------------------- Argument Parsing ------------------- ##
if [ "$#" -lt 1 ]; then
    echo -e "${ORANGE}Usage: $0 {v6e-16} [--rebuild]${NC}"
    exit 1
fi

CONFIG=$1
REBUILD_FLAG=false

# Check for the --rebuild flag starting from the 2nd argument
for arg in "${@:2}"; do
    if [ "$arg" == "--rebuild" ]; then
        REBUILD_FLAG=true
    fi
done

## ------------------- Configuration ------------------- ##
echo -e "${GREEN}Setting up configuration${NC}"

export PROJECT_ID="tpu-prod-env-one-vm"
export CLUSTER_REGION="us-east5"
export CLUSTER_ZONE="us-east5-b"
export CLUSTER_NAME="chavoshi-benchmark-us-east5b"
export AR_REGION="us-east5"
export GCS_BUCKET_NAME="chavoshi-dlrm-training"
# Flag for zonal cluster
export GKE_LOCATION_FLAG="--zone ${CLUSTER_ZONE}"

# Shared configuration
export AR_REPO_NAME="tpu-repo"
export IMAGE_TAG=$(date +%s)
export MASTER_IMAGE_NAME="dlrm-master-timed"
export WORKER_IMAGE_NAME="dlrm-worker-timed"

# Set YAML Template and Job Name based on the selected configuration
case $CONFIG in
    v6e-16)
        export YAML_FILE="tfjob_v6e_16_gcsfuse.yaml"
        export TFJOB_NAME="dlrm-dcn-v2-tfjob-v6e-16"
        ;;
    *)
        echo -e "${ORANGE}Error: Invalid configuration '$CONFIG'.${NC}"
        echo -e "${ORANGE}Please choose from: {v6e-16}${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}Running with job configuration: $CONFIG${NC}"

## ------------------- Define Image URLs ------------------- ##
export MASTER_IMAGE_URL="${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${MASTER_IMAGE_NAME}:${IMAGE_TAG}"
export WORKER_IMAGE_URL="${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${WORKER_IMAGE_NAME}:${IMAGE_TAG}"

## ------------------- Authenticate & Configure ------------------- ##
echo -e "${ORANGE}🔌 Connecting to GKE cluster: ${CLUSTER_NAME}...${NC}"
gcloud container clusters get-credentials ${CLUSTER_NAME} ${GKE_LOCATION_FLAG} --project ${PROJECT_ID}

echo -e "${ORANGE}🔐 Configuring Docker...${NC}"
gcloud auth configure-docker "${AR_REGION}-docker.pkg.dev"

## ------------------- Ensure Artifact Registry Repo Exists ------------------- ##
echo -e "${ORANGE}🔎 Checking for Artifact Registry repository '${AR_REPO_NAME}'...${NC}"
if ! gcloud artifacts repositories describe ${AR_REPO_NAME} --location=${AR_REGION} --project=${PROJECT_ID} &> /dev/null
then
    echo -e "${ORANGE}Repository not found. Creating '${AR_REPO_NAME}' in ${AR_REGION}...${NC}"
    gcloud artifacts repositories create ${AR_REPO_NAME} \
        --repository-format=docker \
        --location=${AR_REGION} \
        --project=${PROJECT_ID} \
        --description="Docker repository for TPU jobs"
    echo -e "${GREEN}✅ Repository created successfully.${NC}"
else
    echo -e "${GREEN}✅ Repository already exists.${NC}"
fi

## ------------------- Conditional Docker Build & Push ------------------- ##
if [ "$REBUILD_FLAG" = true ]; then
    echo -e "${ORANGE}🚀 Rebuilding Docker images as requested...${NC}"

    echo -e "${ORANGE}Building Master image...${NC}"
    docker build --build-arg CACHE_BUSTER=${IMAGE_TAG} -f master.Dockerfile -t ${MASTER_IMAGE_URL} .
    docker push ${MASTER_IMAGE_URL}
    echo -e "${GREEN}✅ Master image pushed.${NC}"

    echo -e "${ORANGE}Building Worker image...${NC}"
    docker build -f worker.Dockerfile -t ${WORKER_IMAGE_URL} .
    docker push ${WORKER_IMAGE_URL}
    echo -e "${GREEN}✅ Worker image pushed.${NC}"
else
    echo -e "${GREEN}Skipping Docker build. Using existing images.${NC}"
fi

## ------------------- TFJob Deployment & Logging ------------------- ##
echo -e "${ORANGE}▶️  Starting process for TFJob: ${TFJOB_NAME}${NC}"
echo -e "${ORANGE}🧹 Cleaning up any pre-existing TFJob '${TFJOB_NAME}'...${NC}"
kubectl delete tfjob ${TFJOB_NAME} -n default --ignore-not-found=true --wait=false

echo -e "${ORANGE}🚢 Generating and deploying TFJob from template '${YAML_FILE}'...${NC}"
envsubst < "${YAML_FILE}" | kubectl apply -f -
echo -e "${GREEN}✅ TFJob '${TFJOB_NAME}' submitted successfully.${NC}"

echo -e "${ORANGE}⏳ Waiting for the MASTER pod to start running...${NC}"
kubectl wait --for=condition=Ready pod -l training.kubeflow.org/replica-type=master,training.kubeflow.org/job-name=${TFJOB_NAME} --timeout=15m

echo -e "${ORANGE}🪵 Tailing logs for the MASTER pod. Training output appears here. Press Ctrl+C to stop.${NC}"
kubectl logs -f -l training.kubeflow.org/replica-type=master,training.kubeflow.org/job-name=${TFJOB_NAME}

echo -e "${GREEN}✅ Script finished.${NC}"
