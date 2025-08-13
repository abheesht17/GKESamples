#!/bin/bash

# This script downloads the Criteo Terabyte Click Logs dataset files
# (day_0.gz through day_23.gz) from the Hugging Face repository in parallel,
# and then uploads them to a specified Google Cloud Storage bucket.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# The base URL for the dataset files
BASE_URL="https://huggingface.co/datasets/criteo/CriteoClickLogs/resolve/main"
# The GCS bucket and paths for uploading the data
GCS_BUCKET="gs://dlrm-training-chavoshi-gke-dev"
GCS_TRAIN_PATH="${GCS_BUCKET}/criteo_raw/train/"
GCS_EVAL_PATH="${GCS_BUCKET}/criteo_raw/eval/"
# Local directory to store the raw data
LOCAL_DATA_DIR="criteo_raw_data"


# --- Download Step ---
# Create a directory to store the raw data if it doesn't exist
mkdir -p "${LOCAL_DATA_DIR}"
cd "${LOCAL_DATA_DIR}"

echo "--- Starting Criteo Dataset Download in Parallel (day_0 to day_23) ---"
echo "Files will be saved in the '${LOCAL_DATA_DIR}' directory."

# Loop through the days from 0 to 23 to download files
for i in {0..23}
do
  FILE_URL="${BASE_URL}/day_${i}.gz"
  LOCAL_FILE="day_${i}.gz"
  echo "Starting download for ${LOCAL_FILE}..."
  # Use wget to download the file in the background
  wget -c -q -O "${LOCAL_FILE}" "${FILE_URL}" &
done

# Wait for all background wget processes to complete
echo "--- Waiting for all downloads to finish... ---"
wait
echo "--- All 24 files have been downloaded successfully. ---"


# --- Upload to GCS Step ---
echo ""
echo "--- Starting upload to Google Cloud Storage ---"

# Upload day_0 through day_22 to the training directory in parallel
echo "Uploading training files (day_0 to day_22) to ${GCS_TRAIN_PATH}..."
# Create an array of files to upload
train_files=()
for i in {0..22}; do
  train_files+=("day_${i}.gz")
done
# Use gsutil with the -m flag for parallel uploads
gsutil -m cp "${train_files[@]}" "${GCS_TRAIN_PATH}"
echo "Training files uploaded."

# Upload day_23 to the evaluation directory
echo "Uploading evaluation file (day_23) to ${GCS_EVAL_PATH}..."
gsutil cp "day_23.gz" "${GCS_EVAL_PATH}"
echo "Evaluation file uploaded."


cd ..

echo ""
echo "--- All files have been downloaded and uploaded to GCS successfully. ---"

