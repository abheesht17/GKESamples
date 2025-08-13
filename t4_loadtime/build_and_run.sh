#!/bin/bash
cd /usr/local/google/home/chavoshi/GKESamples/t4_loadtime
set -e # Exit immediately if a command exits with a non-zero status.

# --- Step 1: Define Environment Variables ---
echo "▶️ Setting up environment variables..."
export PROJECT_ID="chavoshi-gke-dev"
export ZONE="us-west1-b"
export REGION="us-west1" # Extracted from ZONE for regional services like AR
export CLUSTER_NAME="gpu-repro-cluster-west1b"
export AR_REPO="gpu-repro-repo" # Name for your Artifact Registry repo
export IMAGE_NAME="gstreamer-repro"
export IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:latest"
export NODE_POOL_NAME="t4-r550-driver-pool-32"

echo "  Project: ${PROJECT_ID}"
echo "  Cluster: ${CLUSTER_NAME} in ${ZONE}"
echo "  Image Tag: ${IMAGE_TAG}"
echo ""

echo "  Deleting existing deployment (if it exists)..."
kubectl delete deployment gstreamer-test  --grace-period=0 --force --ignore-not-found=true


# --- Step 2: Check for and Create Artifact Registry Repository ---
echo "▶️ Checking for Artifact Registry repository '${AR_REPO}'..."
if ! gcloud artifacts repositories describe "${AR_REPO}" --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "  Repository not found. Creating it now..."
  gcloud artifacts repositories create "${AR_REPO}" \
    --project="${PROJECT_ID}" \
    --repository-format="docker" \
    --location="${REGION}" \
    --description="Repository for GKE repro images"
  echo "  Repository created successfully."
else
  echo "  Repository already exists. Skipping creation."
fi
echo ""

# --- Step 3: Get GKE Credentials and Configure Docker ---
echo "▶️ Configuring access..."
echo "  Getting GKE credentials for ${CLUSTER_NAME}..."
gcloud container clusters get-credentials "${CLUSTER_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}"

echo "  Configuring Docker to authenticate with Artifact Registry..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev"
echo ""

# --- Step 4: Build and Push Docker Image ---
echo "▶️ Building and pushing Docker image..."
docker build -t "${IMAGE_TAG}" .
docker push "${IMAGE_TAG}"
echo "  Image pushed successfully."
echo ""

# --- Step 5: Deploy to GKE ---
echo "▶️ Deploying to GKE cluster..."

# --- NEW COMMAND ADDED HERE ---

echo "  Applying new deployment..."
envsubst < deployment.yaml | kubectl apply -f -
echo ""

# --- Final Instructions ---
echo "✅ Deployment complete!"
echo "To monitor the pod logs, run the following command:"
echo "kubectl logs -f -l app=gstreamer-test --tail=-1"

kubectl get pods -w
