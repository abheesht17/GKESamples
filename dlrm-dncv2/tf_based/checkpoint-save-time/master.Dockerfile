# master image
FROM python:3.10

ENV GCS_CLIENT_CACHE_TYPE "None"
ENV GCS_READ_CACHE_MAX_SIZE_MB "0"
ENV GCS_READ_CACHE_BLOCK_SIZE_MB "0"
ENV TPU_STDERR_LOG_LEVEL "0"
ENV TF_USE_LEGACY_KERAS "1"

# Install TPU Tensorflow package
RUN pip install \
   --no-cache-dir \
   --upgrade \
   pip
RUN pip install --no-cache-dir tf-keras tensorflow-datasets pyyaml gin-config
RUN pip uninstall -y tf-nightly
# RUN pip install --no-cache-dir tensorflow-tpu -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force

RUN pip install --no-cache-dir \
   https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/tensorflow/tf-2.19.0/experimental/tensorflow_tpu-2.19.0-cp310-cp310-linux_x86_64.whl

# Clone TFRS to a dir
RUN git clone https://github.com/tensorflow/recommenders.git /recommenders
ENV PYTHONPATH "${PYTHONPATH}:/recommenders"
RUN git clone https://github.com/ACW101/models.git /models
ENV PYTHONPATH "${PYTHONPATH}:/models"

COPY train.py /models/official/recommendation/ranking/train.py

# Add entrypoint.sh
ADD entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
