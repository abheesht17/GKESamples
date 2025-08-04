FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    tensorflow-cpu \
    google-cloud-storage \
    absl-py \
    numpy

# Copy the application code into the container
COPY preprocess_for_jax.py .
COPY dataloader.py .

# Set the entrypoint for the container
CMD ["python3", "preprocess_for_jax.py"]
