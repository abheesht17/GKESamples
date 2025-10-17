# worker image
FROM python:3.10
RUN pip install \
   --no-cache-dir \
   --upgrade \
   pip
RUN pip install --no-cache-dir tensorflow-tpu -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force

COPY tpu_tfjob_worker.sh /usr/local/bin/tpu_tfjob_worker.sh
RUN chmod +x /usr/local/bin/tpu_tfjob_worker.sh

ENTRYPOINT ["/usr/local/bin/tpu_tfjob_worker.sh"]
