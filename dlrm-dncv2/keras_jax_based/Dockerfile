FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install --no-cache-dir tensorflow-cpu
RUN pip install --no-cache-dir "keras-rs @ git+https://github.com/keras-team/keras-rs.git"
RUN pip install --no-cache-dir "keras @ git+https://github.com/keras-team/keras.git"


COPY train_jax.py .


CMD ["python3", "train_jax.py"]

