# master image
FROM python:3.10

ENV GCS_CLIENT_CACHE_TYPE="None"
ENV GCS_READ_CACHE_MAX_SIZE_MB="0"
ENV GCS_READ_CACHE_BLOCK_SIZE_MB="0"
ENV TPU_STDERR_LOG_LEVEL="0"
ENV TF_USE_LEGACY_KERAS="1"

# Install TPU Tensorflow package
RUN pip install \
   --no-cache-dir \
   --upgrade \
   pip
RUN pip install --no-cache-dir tf-keras tensorflow-datasets pyyaml gin-config tensorflow-tpu==2.19.1 -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force

# Clone TFRS to a dir and check out the specific working commit
RUN git clone https://github.com/tensorflow/recommenders.git /recommenders && \
    cd /recommenders && \
    git checkout b639fe3a15ce00acf765a005c78fe264d2df7931
ENV PYTHONPATH "${PYTHONPATH}:/recommenders"

# Clone models to a dir and check out the specific working commit
RUN git clone https://github.com/ACW101/models.git /models && \
    cd /models && \
    git checkout 92cd14dfe3ff119f5c979d331768632784b448dc
ENV PYTHONPATH "${PYTHONPATH}:/models"
