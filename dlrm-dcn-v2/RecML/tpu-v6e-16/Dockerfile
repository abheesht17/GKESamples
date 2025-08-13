# Use a Python 3.10 base image to match the JAX TPU Embedding wheel
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install git and sed for setup
RUN apt-get update && apt-get install -y git sed && rm -rf /var/lib/apt/lists/*

# 1. Clone the RecML repository
RUN git clone https://github.com/AI-Hypercomputer/RecML.git .

# 2. Modify requirements.txt to prevent conflicts by commenting out
# tensorflow and protobuf, which we will install manually.
RUN sed -i '/^tensorflow==/s/^/#/' requirements.txt
RUN sed -i '/^protobuf==/s/^/#/' requirements.txt

# 3. Install the CPU-only version of TensorFlow FIRST. This ensures any
# other package that depends on 'tensorflow' will use this CPU version.
RUN pip install tensorflow-cpu==2.19.0

# 4. Resolve the setuptools issue before proceeding
RUN pip install "setuptools<60"
RUN pip install promise==2.3

# 5. Install the rest of the requirements. Pip will see 'tensorflow' is already
# satisfied and will not attempt to install the GPU version.
RUN pip install -r requirements.txt

# 6. Install nightly Jax and TPU libraries. This step will likely install a
# version of numpy that is incompatible with our TensorFlow version.
RUN pip install -U --pre jax jaxlib libtpu requests -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --force

# 7. IMPORTANT: Re-pin numpy to a version compatible with tensorflow-cpu==2.19.0.
# The compatible range is >=1.26.0 and <2.2.0.
RUN pip install "numpy<2.2.0,>=1.26.0" --force-reinstall

# 8. Install the specific JAX TPU Embedding dependency
RUN pip install -U https://storage.googleapis.com/jax-tpu-embedding-whls/20250604/jax_tpu_embedding-0.1.0.dev20250604-cp310-cp310-manylinux_2_35_x86_64.whl --force

# 9. Install other specified packages
RUN pip install -U dm-tree flax google-metrax scikit-learn

# 10. Copy your GKE-specific run script into the container
COPY train_and_checkpoint.sh .

# 11. Make the new script executable
RUN chmod +x train_and_checkpoint.sh

# 12. Set the entrypoint to the new script
ENTRYPOINT ["./train_and_checkpoint.sh"]

