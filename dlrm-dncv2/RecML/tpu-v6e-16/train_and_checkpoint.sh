#!/bin/bash

export TPU_NAME="chavoshi-dlrm-dnc-v2-benchmark"
export LIBTPU_INIT_ARGS=
export XLA_FLAGS=

# Variables from the original script, kept for consistency.
export LEARNING_RATE=0.0034
export BATCH_SIZE=16896
export EMBEDDING_SIZE=128
export NUM_STEPS=28000
export CHECKPOINT_INTERVAL=28000
export EVAL_INTERVAL=28000
export EVAL_STEPS=660
export MODE=train
export EMBEDDING_THRESHOLD=21000
export LOGGING_INTERVAL=1500
export RESTORE_CHECKPOINT=true

# Use the GCS bucket you provided for model checkpoints. This is essential for
# multi-host training as all pods need a shared location to save and restore from.
export MODEL_DIR="gs://chavoshi-dlrm-training/v6e-16-sc/"

# Use the public Criteo dataset paths from the original script.
export FILE_PATTERN="gs://qinyiyan-vm/mlperf-dataset/criteo_merge_balanced_4224/train-*"
export EVAL_FILE_PATTERN="gs://qinyiyan-vm/mlperf-dataset/criteo_merge_balanced_4224/train-*"

python recml/inference/models/jax/DLRM_DCNv2/dlrm_main.py \
--learning_rate=${LEARNING_RATE} \
--batch_size=${BATCH_SIZE} \
--embedding_size=${EMBEDDING_SIZE} \
--embedding_threshold=${EMBEDDING_THRESHOLD} \
--model_dir=${MODEL_DIR} \
--file_pattern=${FILE_PATTERN} \
--num_steps=${NUM_STEPS} \
--save_checkpoint_interval=${CHECKPOINT_INTERVAL} \
--restore_checkpoint=${RESTORE_CHECKPOINT} \
--eval_interval=${EVAL_INTERVAL} \
--eval_file_pattern=${EVAL_FILE_PATTERN} \
--eval_steps=${EVAL_STEPS}  \
--mode=${MODE} \
--logging_interval=${LOGGING_INTERVAL}

