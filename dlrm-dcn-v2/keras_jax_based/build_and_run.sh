#!/bin/bash
set -e

# --- Color definitions ---
ORANGE='\033[38;5;208m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

## ------------------- Argument Parsing ------------------- ##
# Set 'prod' as the default project alias.
PROJECT_ALIAS="prod"

# If the first argument is explicitly 'prod', consume it.
# This makes the 'prod' argument optional.
if [ "$1" == "prod" ]; then
  shift # After shift, $2 becomes $1, etc.
fi

# After potentially shifting, we require at least one argument for the configuration.
if [ "$#" -lt 1 ]; then
    echo -e "${ORANGE}Error: Invalid configuration '$CONFIG'. Choose from v6e-{16|32|64|128} or v5p-{16|32|64|128}.${NC}"
    exit 1
fi

CONFIG=$1
REBUILD_FLAG=false

# Check for the --rebuild flag in the remaining arguments (now starting from the 2nd).
for arg in "${@:2}"; do
    if [ "$arg" == "--rebuild" ]; then
        REBUILD_FLAG=true
    fi
done

## ------------------- Dynamic Configuration ------------------- ##
echo -e "${GREEN}Setting up configuration for project alias: '${PROJECT_ALIAS}'${NC}"

case $PROJECT_ALIAS in
    prod) 
        export PROJECT_ID="tpu-prod-env-one-vm"
        export CLUSTER_ZONE="us-east5-b"
        export CLUSTER_NAME="chavoshi-benchmark-us-east5b"
        export AR_REGION="us-east5"
        export BUCKET_NAME="chavoshi-dlrm-dnc-v2-benchmark"
        export GKE_LOCATION_FLAG="--zone ${CLUSTER_ZONE}"
        ;;    
    *) 
        echo -e "${ORANGE}Error: Invalid project alias '$PROJECT_ALIAS'. Only 'prod' is configured.${NC}"
        exit 1
        ;;
esac

# Shared configuration for the JAX job
export AR_REPO_NAME="tpu-repo"
export IMAGE_TAG="latest"
export IMAGE_NAME="dlrm-keras-jax"
# The name of the replicatedJob inside the YAML, used for label selection.
export REPLICATED_JOB_NAME="dlrm-job"

# Set YAML Template and Job Name based on the selected configuration
case $CONFIG in
    v6e-16)
        export YAML_FILE="jobset_jax_v6e_16.yaml"
        export JOB_NAME="keras-jax-16-dlrm-jobset"
        ;;    
    v6e-32)
        export YAML_FILE="jobset_jax_v6e_32.yaml"
        export JOB_NAME="keras-jax-32-dlrm-jobset"
        ;;    
    v6e-64)
        export YAML_FILE="jobset_jax_v6e_64.yaml"
        export JOB_NAME="keras-jax-64-dlrm-jobset"
        ;;    
    v6e-128)
        export YAML_FILE="jobset_jax_v6e_128.yaml"
        export JOB_NAME="keras-jax-128-dlrm-jobset"
        ;;    
    v6e-256)
        export YAML_FILE="jobset_jax_v6e_256.yaml"
        export JOB_NAME="keras-jax-256-dlrm-jobset"
        ;;    
    v5p-16)
        export YAML_FILE="jobset_jax_v5p_16.yaml"
        export JOB_NAME="keras-jax-v5p-16-dlrm-jobset"
        ;;    
    v5p-32)
        export YAML_FILE="jobset_jax_v5p_32.yaml"
        export JOB_NAME="keras-jax-v5p-32-dlrm-jobset"
        ;;    
    v5p-64)
        export YAML_FILE="jobset_jax_v5p_64.yaml"
        export JOB_NAME="keras-jax-v5p-64-dlrm-jobset"
        ;;    
    v5p-128)
        export YAML_FILE="jobset_jax_v5p_128.yaml"
        export JOB_NAME="keras-jax-v5p-128-dlrm-jobset"
        ;;    
    *)
        echo -e "${ORANGE}Error: Invalid configuration '$CONFIG'. Choose from v6e-{16|32|64|128|256} or v5p-{16|32|64|128}.${NC}"
        exit 1
        ;;    
esac

## ------------------- Define Image URL ------------------- ##
export IMAGE_URI="${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"
export BUCKET_PATH="gs://${BUCKET_NAME}"
export CHECKPOINT_DIR="${BUCKET_PATH}/dlrm-dcn-v2-checkpoints-jobset-gcsfuse"
export DATA_DIR="${BUCKET_PATH}/criteo-terabyte-preprocessed"


## ------------------- Authenticate & Configure ------------------- ##
echo -e "${ORANGE}🔌 Connecting to GKE cluster: ${CLUSTER_NAME}...${NC}"
gcloud container clusters get-credentials ${CLUSTER_NAME} ${GKE_LOCATION_FLAG} --project ${PROJECT_ID}

echo -e "${ORANGE}🔐 Configuring Docker...${NC}"
gcloud auth configure-docker "${AR_REGION}-docker.pkg.dev"

## ------------------- Ensure Artifact Registry Repo Exists ------------------- ##
echo -e "${ORANGE}🔎 Checking for Artifact Registry repository '${AR_REPO_NAME}'...${NC}"
if ! gcloud artifacts repositories describe ${AR_REPO_NAME} --location=${AR_REGION} --project=${PROJECT_ID} &> /dev/null
then
    echo -e "${ORANGE}Repo not found. Creating '${AR_REPO_NAME}'...${NC}"
    gcloud artifacts repositories create ${AR_REPO_NAME} \
        --repository-format=docker \
        --location=${AR_REGION} \
        --project=${PROJECT_ID}
else
    echo -e "${GREEN}✅ Repository already exists.${NC}"
fi

## ------------------- Conditional Docker Build & Push ------------------- ##
if [ "$REBUILD_FLAG" = true ]; then
    echo -e "${ORANGE}🚀 Rebuilding JAX Docker image as requested...${NC}"
    docker build -f Dockerfile -t ${IMAGE_URI} .
    docker push ${IMAGE_URI}
    echo -e "${GREEN}✅ JAX image pushed to ${IMAGE_URI}.${NC}"
else
    echo -e "${GREEN}Skipping Docker build. Using existing images.${NC}"
fi

## ------------------- JobSet Deployment & Logging ------------------- ##
echo -e "${ORANGE}▶️  Starting process for JobSet: ${JOB_NAME}${NC}"
echo -e "${ORANGE}🧹 Cleaning up any pre-existing JobSet '${JOB_NAME}'...${NC}"

# This prevents pods from getting stuck in a "Terminating" state.
kubectl delete pod -n default -l jobset.sigs.k8s.io/jobset-name=${JOB_NAME} --grace-period=0 --force --ignore-not-found=true

# Delete the JobSet itself.
kubectl delete jobset ${JOB_NAME} -n default --ignore-not-found=true --wait=true

echo -e "${ORANGE}🚢 Generating and deploying JobSet from template '${YAML_FILE}'...${NC}"
envsubst < "${YAML_FILE}" | kubectl apply -f -
echo -e "${GREEN}✅ JobSet '${JOB_NAME}' submitted successfully.${NC}"

echo -e "${ORANGE}⏳ Waiting for the main pod (coordinator) to enter the Running phase...${NC}"
kubectl get pods
# This command now waits for the pod's phase to be 'Running', which happens after init containers complete.
kubectl wait --for=jsonpath='{.status.phase}'=Running pod -l jobset.sigs.k8s.io/jobset-name=${JOB_NAME},jobset.sigs.k8s.io/replicatedjob-name=${REPLICATED_JOB_NAME},jobset.sigs.k8s.io/job-index=0 --timeout=15m

echo -e "${ORANGE}🪵 Tailing logs for the main pod. Training output appears here. Press Ctrl-C to stop.${NC}"
# The label selector here is also corrected to match the JobSet standard.
# We log the specific JAX container, not the GCS FUSE sidecar.
kubectl logs -f -l jobset.sigs.k8s.io/jobset-name=${JOB_NAME},jobset.sigs.k8s.io/replicatedjob-name=${REPLICATED_JOB_NAME},jobset.sigs.k8s.io/job-index=0 -c keras-jax-dlrm

echo -e "${GREEN}✅ Script finished.${NC}"