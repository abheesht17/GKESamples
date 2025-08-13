# worker image
FROM python:3.10
RUN pip install \
   --no-cache-dir \
   --upgrade \
   pip
# RUN pip install --no-cache-dir tensorflow-tpu -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force

RUN pip install --no-cache-dir \
   https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/tensorflow/tf-2.19.0/experimental/tensorflow_tpu-2.19.0-cp310-cp310-linux_x86_64.whl

COPY tpu_tfjob_worker.sh /usr/local/bin/tpu_tfjob_worker.sh
RUN chmod +x /usr/local/bin/tpu_tfjob_worker.sh

ENTRYPOINT ["/usr/local/bin/tpu_tfjob_worker.sh"]
