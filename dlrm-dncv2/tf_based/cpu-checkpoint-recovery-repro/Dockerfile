FROM python:3.10

ENV GCS_CLIENT_CACHE_TYPE "None"
ENV GCS_READ_CACHE_MAX_SIZE_MB "0"
ENV GCS_READ_CACHE_BLOCK_SIZE_MB "0"
ENV TF_USE_LEGACY_KERAS "1"

ENV TPU_STDERR_LOG_LEVEL "0"

ENV ENABLE_ICI_RESILIENCY "false"
ENV ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY "false"

RUN apt-get update && \
    apt-get install libtcmalloc-minimal4 && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
ENV LD_PRELOAD "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

RUN pip install \
--no-cache-dir \
--upgrade \
pip
RUN pip install --no-cache-dir tf-keras==2.19.0 tensorflow-serving-api==2.19.0
RUN pip uninstall -y tensorflow
# RUN pip install --no-cache-dir https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/tensorflow/tf-2.19.0/experimental/tensorflow_tpu-2.19.0-cp310-cp310-linux_x86_64.whl 
RUN pip install --no-cache-dir https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/tensorflow/tf-2.19.0/tensorflow_tpu-2.19.0-cp310-cp310-linux_x86_64.whl
RUN pip install --no-cache-dir libtpu==0.0.12
RUN pip install --no-cache-dir tensorflow-datasets

RUN git clone https://github.com/tensorflow/recommenders.git /recommenders
ENV PYTHONPATH "${PYTHONPATH}:/recommenders"

COPY tpu_embedding_for_serving.py /usr/local/lib/python3.10/site-packages/tensorflow/python/tpu/tpu_embedding_for_serving.py
COPY tpu_embedding_base.py /usr/local/lib/python3.10/site-packages/tensorflow/python/tpu/tpu_embedding_base.py
COPY tpu_embedding_layer.py /recommenders/tensorflow_recommenders/layers/embedding/tpu_embedding_layer.py
COPY tpu-runtime.service /etc/systemd/system/tpu-runtime.service
COPY training_code.py /training_code.py
COPY tpu_tfjob_worker.sh /usr/local/bin/tpu_tfjob_worker.sh
RUN chmod +x /usr/local/bin/tpu_tfjob_worker.sh

