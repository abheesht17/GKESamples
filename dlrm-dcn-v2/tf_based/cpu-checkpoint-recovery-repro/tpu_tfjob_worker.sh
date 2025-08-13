#!/bin/bash

# ==============================================================================
# Script to run a TPU worker for TFJob on multi-host TPUs.
# ==============================================================================

set -euxo pipefail


IFS='-' read -r -a TEMP_ARRAY <<<"$HOSTNAME"
export TPU_WORKER_ID="${TEMP_ARRAY[${#TEMP_ARRAY[@]} - 1]}"
IFS=',' read -r -a hostnames <<<"$TPU_WORKER_HOSTNAMES"
export TPU_HOSTNAME_OVERRIDE="${hostnames[${TPU_WORKER_ID}]}"
export LIBTPU_INIT_ARGS="--xla_sc_splitting_along_feature_dimension=auto --copy_with_dynamic_shape_op_output_pjrt_buffer=true --xla_tpu_embedding_table_oblongness_threshold=inf"
export TF_XLA_FLAGS="--tf_mlir_enable_mlir_bridge=true --tf_xla_sparse_core_disable_table_stacking=true --tf_mlir_enable_convert_control_to_data_outputs_pass=true --tf_mlir_enable_merge_control_flow_pass=true --tf_mlir_enable_tpu_variable_runtime_reformatting_pass=false --tf_xla_disable_full_embedding_pipelining=true"
export NEXT_PLUGGABLE_DEVICE_USE_C_API="true"

export TF_PLUGGABLE_DEVICE_LIBRARY_PATH="/usr/local/lib/python3.10/site-packages/libtpu/libtpu.so"
export TPU_LIBRARY_PATH="/usr/local/lib/python3.10/site-packages/libtpu/libtpu.so"


export GCS_RESOLVE_REFRESH_SECS="60"
export GCS_REQUEST_CONNECTION_TIMEOUT_SECS="300"
export GCS_METADATA_REQUEST_TIMEOUT_SECS="300"
export GCS_READ_REQUEST_TIMEOUT_SECS="300"
export GCS_WRITE_REQUEST_TIMEOUT_SECS="600"


unset TF_CONFIG

printenv

python3 -c "
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.training import server_lib
import tensorflow as tf
import random
import numpy as np
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

tf.profiler.experimental.server.start(6009)
server_def = tensorflow_server_pb2.ServerDef(protocol='grpc')
job_def = server_def.cluster.job.add()
job_def.name = 'tpu_worker'
job_def.tasks[0] = 'localhost:8470'
server_def.job_name = 'tpu_worker'
server_def.task_index = 0

config = config_pb2.ConfigProto()

# Create GRPC Server instance
server = server_lib.Server(server_def, config=config)

# join() is blocking, unlike start()
server.join()
"
