# Use a Python 3.10 base image, which is a good choice for modern JAX.
FROM python:3.10-slim

# Set the working directory inside the container.
WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    git clone https://github.com/AI-Hypercomputer/RecML.git . && \
    rm -rf /var/lib/apt/lists/*

RUN \
    # 1. Upgrade pip and install setuptools first to handle complex packages.
    pip install --no-cache-dir --upgrade pip "setuptools<60" && \
    \
    pip install --no-cache-dir protobuf==4.21.12 && \
    \
    # 3. Install the CPU-only version of TensorFlow.
    pip install --no-cache-dir tensorflow-cpu==2.19.0 && \
    \
    # 4. Install packages from requirements.txt, but IGNORE the ones we have
    # manually installed to prevent version conflicts.
    grep -vE "tensorflow|protobuf" requirements.txt | pip install --no-cache-dir -r /dev/stdin && \
    \
    # 5. Install nightly JAX, jaxlib, and libtpu for TPU support.
    # Using --force-reinstall helps resolve dependency conflicts from other packages.
    pip install --no-cache-dir -U --pre jax jaxlib libtpu requests \
      -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html \
      -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --force-reinstall && \
    \
    # 6. Install the specific JAX TPU Embedding dependency.
    pip install --no-cache-dir -U https://storage.googleapis.com/jax-tpu-embedding-whls/20250604/jax_tpu_embedding-0.1.0.dev20250604-cp310-cp310-manylinux_2_35_x86_64.whl --force-reinstall && \
    \
    # 7. Install the remaining high-level libraries.
    pip install --no-cache-dir -U dm-tree flax google-metrax scikit-learn && \
    \
    pip install --no-cache-dir protobuf==4.21.12 && \
    \
    # 8. As the very last step, force-reinstall a numpy version that is
    # compatible with TensorFlow. JAX/nightly builds often install a bleeding-edge
    # numpy, so this step ensures the environment is stable.
    pip install --no-cache-dir "numpy<2.2.0,>=1.26.0" --force-reinstall

RUN pip install --no-cache-dir -U "jax[tpu]>0.4.23" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 9. Copy local, modified dlrm_main.py to overwrite the one from the git repo.
COPY dlrm_main.py /app/recml/inference/models/jax/DLRM_DCNv2/dlrm_main.py 

# 10. Copy the customized training script into the container.
# This script should be in the same directory where you run `docker build`.
COPY train_and_checkpoint.sh .

# 11. Make the training script executable.
RUN chmod +x train_and_checkpoint.sh

