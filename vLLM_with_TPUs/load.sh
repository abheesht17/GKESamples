#!/bin/bash
N=60
for i in $(seq 1 $N); do
  while true; do
    curl http://localhost:8080/v1/completions -H "Content-Type: application/json" -d '{"model": "meta-llama/Llama-3.1-70B", "prompt": "Write a story about san francisco", "max_tokens": 1000, "temperature": 0}'
  done &  # Run in the background
done
wait
