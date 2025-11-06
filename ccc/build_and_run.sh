#!/bin/bash
set -euo pipefail

# ==============================================================================
# GKE ComputeClass Reservation with Spot Fallback Validation Script
#
# This script demonstrates and validates the fallback mechanism from a
# simulated reservation (a capacity-limited node pool) to Spot VMs.
# ==============================================================================

# --- Configuration ---
export PROJECT_ID="gemle-gke-dev"
export REGION="us-central1"
# Use a new, unique name for the test cluster to avoid conflicts.
export CLUSTER_NAME="ccc-fallback-test-cluster"

# --- Step 1: Create a new GKE Cluster ---
echo "--------------------------------------------------"
echo "Creating a new regional GKE cluster named '${CLUSTER_NAME}'..."
echo "This may take several minutes."
echo "--------------------------------------------------"
gcloud container clusters create ${CLUSTER_NAME} \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    --machine-type=e2-medium \
    --num-nodes=1 \
    --enable-autoscaling --min-nodes=1 --max-nodes=2

# --- Step 2: Create the 'Simulated Reservation' Node Pool ---
echo "--------------------------------------------------"
echo "Creating a static node pool to simulate a reservation with a hard limit."
echo "We are setting max-nodes=1, which creates ONE e2-medium node PER ZONE in the region (${REGION})."
echo "--------------------------------------------------"
gcloud container node-pools create simulated-reservation-pool \
    --cluster=${CLUSTER_NAME} \
    --region=${REGION} \
    --machine-type=e2-medium \
    --num-nodes=1 --enable-autoscaling --min-nodes=1 --max-nodes=1 \
    --node-labels="cloud.google.com/compute-class=cpu-fallback-test" \
    --node-taints="cloud.google.com/compute-class=cpu-fallback-test:NoSchedule"

# --- Step 3: Create the ComputeClass ---
echo "--------------------------------------------------"
echo "Applying the ComputeClass manifest."
echo "This tells GKE to prioritize 'simulated-reservation-pool' and fallback to Spot VMs."
echo "--------------------------------------------------"
kubectl apply -f - <<EOF
apiVersion: cloud.google.com/v1
kind: ComputeClass
metadata:
  name: cpu-fallback-test
spec:
  priorities:
  - nodepools: ['simulated-reservation-pool']  # Priority 1: The "full" static pool
  - spot: true                                 # Priority 2: Fallback Spot VMs
    machineFamily: e2
  nodePoolAutoCreation:
    enabled: true
EOF

# --- Step 4: Deploy the Test Workload ---
echo "--------------------------------------------------"
echo "Deploying the test application with 4 replicas."
echo "Since our static pool has 3 nodes (1 per zone), this will force one pod to require a fallback node."
echo "--------------------------------------------------"
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fallback-test-app
spec:
  replicas: 4
  selector:
    matchLabels:
      app: fallback-test
  template:
    metadata:
      labels:
        app: fallback-test
    spec:
      nodeSelector:
        cloud.google.com/compute-class: cpu-fallback-test
      tolerations:
      - key: "cloud.google.com/compute-class"
        operator: "Exists"
        effect: "NoSchedule"
      containers:
      - name: nginx
        image: nginx
        resources:
          requests:
            cpu: "100m" # Using a small request to ensure pods can be scheduled
EOF

# --- Step 5: Wait for Autoscaling and Validation ---
echo "--------------------------------------------------"
echo "Waiting for 90 seconds to allow the cluster autoscaler to provision the fallback Spot VM..."
echo "--------------------------------------------------"
sleep 90

echo "--------------------------------------------------"
echo "Validation Commands:"
echo "--------------------------------------------------"

# Validate that the static pool has 3 nodes (one for each zone in us-central1)
echo "# 1. Check the nodes in the simulated reservation pool."
echo "#    EXPECT: 3 nodes with the name 'gke-${CLUSTER_NAME}-simulated-reserv-...'"
kubectl get nodes -l cloud.google.com/gke-nodepool=simulated-reservation-pool --no-headers | wc -l
echo ""

# Validate that a new Spot VM node was created.
echo "# 2. Check for the auto-created Spot VM node."
echo "#    EXPECT: 1 node with the label 'cloud.google.com/gke-provisioning=spot'."
kubectl get nodes -l cloud.google.com/gke-provisioning=spot --no-headers | wc -l
echo ""

# Validate the final pod distribution.
echo "# 3. Check the pod distribution across nodes."
echo "#    EXPECT: 3 pods running on 'simulated-reservation-pool' nodes and 1 pod on the new Spot node."
kubectl get pods -o wide -l app=fallback-test

echo "--------------------------------------------------"
echo "Validation complete."
echo "To clean up, run the following commands:"
echo "gcloud container clusters delete ${CLUSTER_NAME} --region=${REGION} --project=${PROJECT_ID}"
echo "--------------------------------------------------"
