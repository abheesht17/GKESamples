# Unified Docker image for both Master and Worker
FROM python:3.10

ENV GCS_CLIENT_CACHE_TYPE "None"
ENV GCS_READ_CACHE_MAX_SIZE_MB "0"
ENV GCS_READ_CACHE_BLOCK_SIZE_MB "0"
ENV TPU_STDERR_LOG_LEVEL "0"
ENV TF_USE_LEGACY_KERAS "1"

# Install necessary packages, including a pinned version of tensorflow-tpu and tensorboard
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    tensorflow==2.19.0 \
    tensorflow-datasets \
    pyyaml \
    gin-config \
    tensorboard \
    tf-keras

# The TPU wheel must be installed in a separate step
RUN pip install --no-cache-dir https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/tensorflow/tf-2.19.0/experimental/tensorflow_tpu-2.19.0-cp310-cp310-linux_x86_64.whl

# Clone TFRS to a dir
RUN git clone https://github.com/tensorflow/recommenders.git /recommenders
ENV PYTHONPATH "${PYTHONPATH}:/recommenders"
RUN git clone https://github.com/ACW101/models.git /models
ENV PYTHONPATH "${PYTHONPATH}:/models"

# Add entrypoint and worker script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Overwrite the tpu_embedding_layer.py with the patched version
COPY tpu_embedding_layer.py /recommenders/tensorflow_recommenders/layers/embedding/tpu_embedding_layer.py

COPY tpu_tfjob_worker.sh /usr/local/bin/tpu_tfjob_worker.sh
RUN chmod +x /usr/local/bin/tpu_tfjob_worker.sh
