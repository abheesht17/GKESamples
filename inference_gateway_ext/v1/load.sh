#!/bin/bash
N=200
export vllm_service=$(kubectl get service vllm-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' -n vllm-tpu-ns)
for i in $(seq 1 $N); do
  while true; do
    curl http://$vllm_service:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/data/lama3model-bucket/Llama-3.1-70B-Instruct", "prompt": "Write a story about san francisco", "max_tokens": 1000, "temperature": 0}'
  done &  # Run in the background
done
wait