
# Set Env variables
export TPU_NAME="chavoshi-dlrm-dnc-v2-benchmark"
export PROJECT="chavoshi-gke-dev" #"tpu-prod-env-one-vm" or	"chavoshi-gke-dev"
export ZONE="us-central1-a"


# Run the benchmark
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="cd RecML/recml/inference/benchmarks/DLRM_DCNv2/ && chmod +x ./train_and_checkpoint.sh && cd ~/RecML && TPU_NAME=${TPU_NAME} ./recml/inference/benchmarks/DLRM_DCNv2/train_and_checkpoint.sh" 2>&1 | tee tpuv5e-16.txt

# exit to avoid killing the job
exit 0

# kill all jobs
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all   --command="pkill -f train_and_checkpoint.sh"
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all   --command="pkill -f dlrm_main.py"

# tpu setup commands 
gcloud alpha compute tpus tpu-vm create ${TPU_NAME}   --zone=${ZONE}   --accelerator-type=v6e-16   --version=v2-alpha-tpuv6e   --project=${PROJECT}   --metadata=enable-oslogin=TRUE   --scopes=https://www.googleapis.com/auth/cloud-platform 

## ssh to specific workers
gcloud compute tpus tpu-vm ssh --zone "us-east5-a" "chavoshi-dlrm-dnc-v2-benchmark" --project "tpu-prod-env-one-vm"

## Resolving Setup tools issues
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="pip install 'setuptools<60'" 
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="pip install promise==2.3" 

## Downloading Repo
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="git clone https://github.com/AI-Hypercomputer/RecML.git" 

# Pusher to update the remote workers
export REMOTE_DEST_PATH="~/RecML/recml/inference/models/jax/DLRM_DCNv2/" 
./sync_specific_files_scp.sh  dlrm_main.py 
./sync_specific_files_scp.sh  dlrm_model.py
export REMOTE_DEST_PATH="~/RecML/" 
./sync_specific_files_scp.sh  requirements.txt
export REMOTE_DEST_PATH="~/RecML/recml/inference/benchmarks/DLRM_DCNv2/"
./sync_specific_files_scp.sh   train_and_checkpoint.sh 

## Installing requirements.txt
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="cd RecML && pip install -r requirements.txt"  

## Updating Path for downloaded deps
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="PATH=$PATH:/home/depksingh_google_com/.local/bin" 

## Downloading Metrax
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="pip install -U tensorflow-cpu  dm-tree flax google-metrax scikit-learn" 

## Downloading JAX TPU dependency
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="pip install -U https://storage.googleapis.com/jax-tpu-embedding-whls/20250604/jax_tpu_embedding-0.1.0.dev20250604-cp310-cp310-manylinux_2_35_x86_64.whl --force"

## Downloading Jax Lib
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="pip install -U --pre jax jaxlib libtpu requests -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --force"
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="pip install -U \"jax[tpu]>0.4.23\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"

