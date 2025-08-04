#!/bin/bash

# --- Core Configuration from your Cluster ---

export PROJECT_ID="tpu-prod-env-one-vm"
export CLUSTER_ZONE="us-east5-b"
export NETWORK_NAME="chavoshi-test"
export NETWORK_FW_NAME="${NETWORK_NAME}-fw"

## v6e-128 TPU Node.
export NODE_ID="v6e-128-sp"
export ACCELERATOR_TYPE="v6e-128"
export RUNTIME_VERSION="v2-alpha-tpuv6e"

# Your project's default service account.
export SERVICE_ACCOUNT="default@${PROJECT_ID}.iam.gserviceaccount.com"

# Your `v6e-128-sp` node pool is a single, large slice of TPUs.
export NUM_SLICES="1"

export QUEUED_RESOURCE_ID="dlrm-queued-request"
export VALID_DURATION="24h"
export NETWORK_NAME="chavoshi-test"

# --- This is for demonstration only; you don't need to run this ---

## Optimize network performance
export RESOURCE_NAME="chavoshi-benchmark"
export NETWORK_NAME="${RESOURCE_NAME}-privatenetwork"
export NETWORK_FW_NAME="${RESOURCE_NAME}-privatefirewall"

## Using multi-NIC (option for Multislice)
# A base name for the resources, consistent with the previous example.
export RESOURCE_NAME="chavoshi-benchmark"

# Variables for the secondary network interface.
export NETWORK_NAME_2="${RESOURCE_NAME}-secondary-net"
export SUBNET_NAME_2="${RESOURCE_NAME}-secondary-subnet"
export FIREWALL_RULE_NAME="${RESOURCE_NAME}-secondary-fw"
export ROUTER_NAME="${RESOURCE_NAME}-secondary-router"
export NAT_CONFIG="${RESOURCE_NAME}-secondary-nat"


### Create an XPK cluster with single NIC support
# --- Variables for your EXISTING cluster ---
export CLUSTER_NAME="chavoshi-benchmark-us-east5b"
export ZONE="us-east5-b"
export PROJECT_ID="tpu-prod-env-one-vm"
export TPU_TYPE="v6e-128"
export NUM_SLICES="1"
export REGION="us-east5-b"

export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"



### Launch workload 
# Your cluster and project details
export PROJECT_ID="tpu-prod-env-one-vm"
export CLUSTER_NAME="chavoshi-benchmark-us-east5b"
export ZONE="us-east5-b"

# The specs for the TPU slice you want to target
export TPU_TYPE="v6e-128"
export NUM_SLICES="1"

# The generic Python image, since the command installs JAX
export DOCKER_IMAGE="python:3.10"

# The command from your YAML file, stored in a variable
export JAX_TEST_COMMAND="pip install -U --pre jax jaxlib libtpu-nightly requests -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && JAX_PLATFORMS=tpu,cpu ENABLE_PJRT_COMPATIBILITY=true python -c 'import jax; print(\"Total TPU chips:\", jax.device_count())'"

xpk workload create \
  --cluster=${CLUSTER_NAME} \
  --workload="jax-test-via-xpk" \
  --docker-image=${DOCKER_IMAGE} \
  --tpu-type=${TPU_TYPE} \
  --num-slices=${NUM_SLICES} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --command="${JAX_TEST_COMMAND}"
