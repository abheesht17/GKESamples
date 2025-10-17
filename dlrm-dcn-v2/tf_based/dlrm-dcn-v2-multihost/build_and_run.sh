#!/bin/bash
set -e

# --- Color definitions ---
ORANGE='\033[38;5;208m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

## ------------------- Argument Parsing ------------------- ##
if [ "$#" -lt 1 ]; then
    echo -e "${ORANGE}Usage: $0 {v6e-16 | v6e-128} [--rebuild]${NC}"
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

## ------------------- Static Configuration ------------------- ##
export PROJECT_ID="tpu-prod-env-one-vm"
export CLUSTER_REGION="us-east5"
export CLUSTER_ZONE="us-east5-b"
export CLUSTER_NAME="chavoshi-benchmark-us-east5b"
export AR_REGION="us-east5"
export GCS_BUCKET_NAME="chavoshi-dlrm-training"
export GKE_LOCATION_FLAG="--zone ${CLUSTER_ZONE}"
export GSA_NAME="dlrm-job-sa"
export KSA_NAME="dlrm-job-sa"

# Shared configuration
export AR_REPO_NAME="tpu-repo"
export IMAGE_TAG="latest"
export UNIFIED_IMAGE_NAME="tf-dlrm-unified"

# Set YAML Template and Job Name based on the selected configuration
case $CONFIG in
    v6e-16)
        export YAML_FILE="tfjob_v6e_16_gcsfuse.yaml"
        export TFJOB_NAME="tf-16-dlrm-tfjob-gcsfuse"
        ;;
    v6e-128)
        export YAML_FILE="tfjob_v6e_128_gcsfuse.yaml"
        export TFJOB_NAME="tf-128-dlrm-tfjob-gcsfuse"
        ;;
    *)
        echo -e "${ORANGE}Error: Invalid configuration '$CONFIG'.${NC}"
        echo -e "${ORANGE}Please choose from: {v6e-16 | v6e-128}${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}Running with job configuration: $CONFIG${NC}"

## ------------------- Define Image URLs ------------------- ##
export UNIFIED_IMAGE_URL="${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${UNIFIED_IMAGE_NAME}:${IMAGE_TAG}"

## ------------------- Authenticate & Configure ------------------- ##
echo -e "${ORANGE}🔌 Connecting to GKE cluster: ${CLUSTER_NAME}...${NC}"
gcloud container clusters get-credentials ${CLUSTER_NAME} ${GKE_LOCATION_FLAG} --project ${PROJECT_ID}

echo -e "${ORANGE}🔐 Configuring Docker...${NC}"
gcloud auth configure-docker "${AR_REGION}-docker.pkg.dev"

## ------------------- Service Account & IAM Setup ------------------- ##
echo -e "${ORANGE}🛠️  Setting up Service Accounts and IAM permissions...${NC}"

# Create Google Service Account (GSA)
if ! gcloud iam service-accounts describe ${GSA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com --project=${PROJECT_ID} &> /dev/null; then
    echo "Creating GSA '${GSA_NAME}'..."
    gcloud iam service-accounts create ${GSA_NAME} --project=${PROJECT_ID}
else
    echo "✅ GSA '${GSA_NAME}' already exists."
fi

# Grant GSA permissions on the GCS bucket
echo "Granting GSA permissions to GCS bucket '${GCS_BUCKET_NAME}'..."
gsutil iam ch serviceAccount:${GSA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com:objectAdmin gs://${GCS_BUCKET_NAME}

# Create Kubernetes Service Account (KSA) and bind to GSA for Workload Identity
echo "Applying KSA configuration..."
envsubst < service-account.yaml | kubectl apply -f -

echo "Binding KSA to GSA for Workload Identity..."
gcloud iam service-accounts add-iam-policy-binding ${GSA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:${PROJECT_ID}.svc.id.goog[default/${KSA_NAME}]" \
    --project ${PROJECT_ID}

echo -e "${GREEN}✅ Service Account setup complete.${NC}"


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
    echo -e "${ORANGE}🚀 Rebuilding Docker image as requested...${NC}"

    echo -e "${ORANGE}Building unified image...${NC}"
    docker build -f Dockerfile -t ${UNIFIED_IMAGE_URL} .
    docker push ${UNIFIED_IMAGE_URL}
    echo -e "${GREEN}✅ Unified image pushed.${NC}"
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
