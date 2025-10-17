# commands to run in a container to launch tensorboard for tracking
kubectl exec jax-16-dlrm-jobset-dlrm-job-0-0-lcvw5 -it -- bash
kubectl port-forward  jax-16-dlrm-jobset-dlrm-job-0-0-lcvw5 8080:8080

# run within container
python3 -m venv jax-env
source jax-env/bin/activate
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -U tensorboard tensorboard-plugin-profile

tensorboard --logdir=/gcs/benchmark/model-output/ --port=8080

tensorboard --logdir=/gcs/benchmark/model-output/jax-v5p-16-dlrm-jobset/jax_profiler/ --port=8080
