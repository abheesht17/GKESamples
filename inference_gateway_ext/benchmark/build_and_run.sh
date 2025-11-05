gcloud container clusters get-credentials inference-gateway-1 --zone us-east5-b --project gemle-gke-dev

kubectl apply -f vllm-llama3-70b-tpu.yaml
