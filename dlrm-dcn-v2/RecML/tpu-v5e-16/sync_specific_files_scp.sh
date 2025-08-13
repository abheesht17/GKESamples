#!/bin/bash

WORKERS=(t1v-n-99bbd697-w-3 t1v-n-99bbd697-w-2 t1v-n-99bbd697-w-0 t1v-n-99bbd697-w-1)
# REMOTE_DEST_PATH="~/RecML/recml/inference/models/jax/DLRM_DCNv2/" 
# REMOTE_DEST_PATH="~/RecML/" 
# REMOTE_DEST_PATH="~/RecML/recml/inference/benchmarks/DLRM_DCNv2/"

if [ "$#" -eq 0 ]; then
    echo "ERROR: No files provided."
    echo "Usage: ./sync_specific_files_scp.sh [file1] "
    exit 1
fi

FILES_TO_SYNC=("$@")

echo "Starting targeted sync for ${#FILES_TO_SYNC[@]} to ${REMOTE_DEST_PATH}${FILES_TO_SYNC} file(s) to all workers..."
echo ""

for worker in "${WORKERS[@]}"; do
  echo "-----------------------------------------------------"
  echo ">>> Syncing to worker: $worker"
  echo "-----------------------------------------------------"
  
  gcloud compute scp --recurse ${FILES_TO_SYNC[@]} $worker:${REMOTE_DEST_PATH}${FILES_TO_SYNC} --project=$PROJECT --zone=$ZONE

  if [ $? -eq 0 ]; then
    echo ">>> Successfully synced files to $worker."
  else
    echo ">>> ERROR: Failed to sync to $worker."
  fi
  echo ""
done

echo "Targeted synchronization complete."
