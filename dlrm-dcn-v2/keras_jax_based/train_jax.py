import os
import time
import numpy as np
import tensorflow as tf
import keras
import jax
import keras_rs
from typing import Dict

VOCAB_SIZES = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
    40000000, 40000000, 590152, 12973, 108, 36
]
NUM_SPARSE_FEATURES = len(VOCAB_SIZES)
NUM_DENSE_FEATURES = 13
EMBEDDING_DIM = 128
DCN_LOW_RANK_DIM = 512
DCN_NUM_LAYERS = 3

def create_distributed_embedding_layer(per_replica_batch_size: int) -> keras_rs.layers.DistributedEmbedding:
    """Creates a single DistributedEmbedding layer for all sparse features."""
    feature_configs = {}
    for i, vocab_size in enumerate(VOCAB_SIZES):
        table = keras_rs.layers.TableConfig(
            name=f"table_{i}",
            vocabulary_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            optimizer=keras.optimizers.Adagrad(learning_rate=0.05),
            placement="auto",
        )
        feature_configs[f"feature_{i}"] = keras_rs.layers.FeatureConfig(
            name=f"feature_{i}",
            table=table,
            input_shape=(per_replica_batch_size,),
            output_shape=(per_replica_batch_size, EMBEDDING_DIM),
        )
    return keras_rs.layers.DistributedEmbedding(
        feature_configs=feature_configs,
        name="distributed_embedding"
    )

class DLRM(keras.Model):
    """
    DLRM model implemented using the Model Subclassing API to handle
    the custom preprocessed input from DistributedEmbedding.
    """
    def __init__(self, embedding_layer: keras_rs.layers.DistributedEmbedding, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer
        
        self.bottom_mlp = keras.Sequential([
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(EMBEDDING_DIM, activation="relu")
        ], name="bottom_mlp")

        self.feature_cross_layers = []
        for _ in range(DCN_NUM_LAYERS):
            self.feature_cross_layers.append(
                keras_rs.layers.FeatureCross(
                    projection_dim=DCN_LOW_RANK_DIM, use_bias=True
                )
            )

        self.top_mlp = keras.Sequential([
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ], name="top_mlp")

        self.concat = keras.layers.Concatenate(axis=1)

    def call(self, inputs: Dict[str, tf.Tensor]):
        dense_features = inputs["dense_features"]
        preprocessed_sparse = inputs["preprocessed_sparse"]

        bottom_mlp_output = self.bottom_mlp(dense_features)

        embedding_outputs_dict = self.embedding_layer(preprocessed_sparse)
        embedding_vectors = list(embedding_outputs_dict.values())

        x0 = self.concat(embedding_vectors + [bottom_mlp_output])

        x = x0
        for cross_layer in self.feature_cross_layers:
            x = cross_layer(x0, x)
        interaction_output = x

        top_mlp_input = self.concat([bottom_mlp_output, interaction_output])
        
        return self.top_mlp(top_mlp_input)

def create_dataset_from_tfrecords(input_path, is_training, global_batch_size):
    feature_spec = {
        'label': tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=None)
    }
    for i in range(1, NUM_DENSE_FEATURES + 1):
        feature_spec[f'dense-feature-{i}'] = tf.io.FixedLenFeature(
            [1], dtype=tf.float32, default_value=None
        )
    for i in range(NUM_DENSE_FEATURES + 1, NUM_DENSE_FEATURES + NUM_SPARSE_FEATURES + 1):
        feature_spec[f'sparse-feature-{i}'] = tf.io.VarLenFeature(dtype=tf.int64)

    def _parse_fn(features):
        label = tf.cast(features.pop('label'), tf.float32)
        dense_features_list = []
        for i in range(1, NUM_DENSE_FEATURES + 1):
            dense_feat = features[f'dense-feature-{i}']
            dense_features_list.append(dense_feat)
        dense_features = tf.concat(dense_features_list, axis=1)
        sparse_features = {}
        for i in range(NUM_DENSE_FEATURES + 1, NUM_DENSE_FEATURES + NUM_SPARSE_FEATURES + 1):
            model_feature_index = i - (NUM_DENSE_FEATURES + 1)
            sparse_features[f"feature_{model_feature_index}"] = tf.sparse.to_dense(features[f'sparse-feature-{i}'])
        return {"dense_features": dense_features, "sparse_features": sparse_features}, label

    dataset = tf.data.Dataset.list_files(input_path, shuffle=is_training)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not is_training,
    )
    dataset = dataset.batch(global_batch_size, drop_remainder=is_training)
    dataset = dataset.map(
        lambda x: tf.io.parse_example(x, feature_spec),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(_parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)

class ThroughputLogger(keras.callbacks.Callback):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.total_examples = 0
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.total_examples = 0
    def on_train_batch_end(self, batch, logs=None):
        self.total_examples += self.batch_size
    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - self.epoch_start_time
        if elapsed_time > 0:
            throughput = self.total_examples / elapsed_time
            print(f"\nEpoch {epoch + 1} - Throughput: {throughput:.2f} examples/sec")
            if logs is not None:
                logs['throughput'] = throughput

def main():
    if os.environ.get("JAX_PROCESS_ID"):
        print("--- Initializing JAX for Multi-Host Environment ---")
        jax.distributed.initialize()
    else:
        print("--- Running in Single-Host Mode ---")

    devices = jax.devices()
    print(f"Global JAX devices: {devices}")

    if os.environ.get("JAX_PROCESS_ID"):
        mesh = keras.distribution.DeviceMesh(shape=(len(devices),), axis_names=("batch",), devices=devices)
        distribution = keras.distribution.DataParallel(device_mesh=mesh)
        keras.distribution.set_distribution(distribution)
        print("Keras distribution set for JAX DataParallel.")

    GLOBAL_BATCH_SIZE = 32768
    num_replicas = jax.device_count()
    per_replica_batch_size = GLOBAL_BATCH_SIZE // num_replicas if num_replicas > 0 else GLOBAL_BATCH_SIZE

    print("--- Batch Size Configuration ---")
    print(f"Global batch size: {GLOBAL_BATCH_SIZE}")
    print(f"Number of replicas: {num_replicas}")
    print(f"Per-replica batch size: {per_replica_batch_size}")

    print("--- Setting up DistributedEmbedding and preprocessing pipeline ---")
    
    embedding_layer = create_distributed_embedding_layer(per_replica_batch_size)
    
    train_data_path = os.environ.get("TRAIN_DATA_PATH", "gs://zyc_dlrm/dataset/tb_tf_record_train_val/train/day_*/*")
    eval_data_path = os.environ.get("EVAL_DATA_PATH", "gs://zyc_dlrm/dataset/tb_tf_record_train_val/eval/day_*/*")
    
    train_dataset = create_dataset_from_tfrecords(train_data_path, is_training=True, global_batch_size=GLOBAL_BATCH_SIZE)
    eval_dataset = create_dataset_from_tfrecords(eval_data_path, is_training=False, global_batch_size=GLOBAL_BATCH_SIZE)

    sample_inputs, _ = next(iter(train_dataset))
    preprocessed_output = embedding_layer.preprocess(sample_inputs["sparse_features"])
    preprocessed_spec = tf.nest.map_structure(tf.TensorSpec.from_tensor, preprocessed_output)
    
    output_signature = (
        {
            "dense_features": tf.TensorSpec(shape=(None, NUM_DENSE_FEATURES), dtype=tf.float32),
            "preprocessed_sparse": preprocessed_spec,
        },
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    )
    
    def preprocessed_generator(dataset, is_training):
        for inputs, labels in dataset:
            preprocessed_sparse = embedding_layer.preprocess(inputs["sparse_features"], training=is_training)
            yield {"dense_features": inputs["dense_features"], "preprocessed_sparse": preprocessed_sparse}, labels

    preprocessed_train_dataset = tf.data.Dataset.from_generator(
        lambda: preprocessed_generator(train_dataset, is_training=True),
        output_signature=output_signature
    )
    preprocessed_eval_dataset = tf.data.Dataset.from_generator(
        lambda: preprocessed_generator(eval_dataset, is_training=False),
        output_signature=output_signature
    )

    model = DLRM(embedding_layer=embedding_layer)
    optimizer = keras.optimizers.Adam(learning_rate=0.00025) 
    
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "auc"], jit_compile=True)
    
    throughput_callback = ThroughputLogger(batch_size=GLOBAL_BATCH_SIZE)
    train_steps = 2
    validation_steps = 1

    print("--- Starting training with DistributedEmbedding ---")
    model.fit(
        preprocessed_train_dataset,
        epochs=1,
        steps_per_epoch=train_steps,
        validation_data=preprocessed_eval_dataset,
        validation_steps=validation_steps,
        callbacks=[throughput_callback]
    )

    print("Training complete.")

if __name__ == "__main__":
    main()
