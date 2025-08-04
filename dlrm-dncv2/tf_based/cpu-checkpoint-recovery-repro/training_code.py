"""
Basic TPU embedding example for testing TPU functionality.

This script demonstrates TPU-based MovieLens embedding model training and
evaluation. It supports both training and evaluation modes, with configurable
checkpoint handling.
"""
import argparse
import json
import os
import random
from typing import List

import numpy as np
os.environ['TPU_LOAD_LIBRARY'] = '0'
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from tensorflow_recommenders.layers.embedding import TPUEmbedding


def _get_grpc_endpoint_for_tpu_cluster_resolver(port_number: int = 8470) -> str:
  grpc_endpoint = []

  tf_config = json.loads(os.environ['TF_CONFIG'])
  cluster_spec = tf_config['cluster']
  replica_type: str
  replica_addresses: List[str]
  for replica_type, replica_addresses in cluster_spec.items():
    if replica_type == "worker":
      for address in replica_addresses:
        ip, _ = address.split(':')
        grpc_endpoint.append(f'grpc://{ip}:{port_number}')

  if len(grpc_endpoint) == 0:
    raise ValueError('No grpc endpoint found.')

  return ','.join(grpc_endpoint)


# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Constants
GCS_BUCKET = 'gs://chavoshi-dlrm-training/'
GLOBAL_BATCH_SIZE = 16*64
MOVIE_VOCAB_SIZE = 2048
USER_VOCAB_SIZE = 2048
EMBED_DIM = 64
TRAIN_SIZE = 80_000
TEST_SIZE = 19_968
SHUFFLE_BUF = 100_000

use_cpu_strategy = os.environ.get('USE_CPU_STRATEGY', 'false').lower() == 'true'

if not use_cpu_strategy:
  tpu_name = _get_grpc_endpoint_for_tpu_cluster_resolver()
  print(f'Using TPU: {tpu_name}')
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
  tf.config.experimental_connect_to_cluster(resolver)

  topology = tf.tpu.experimental.initialize_tpu_system(resolver)


  def shuffle(endpoints, device_assignment) -> List[str]:
    dc_idx_mapping = {}
    for i, coordinate in enumerate(device_assignment.topology.device_coordinates):
      dc_idx_mapping[coordinate.tobytes()] = i

    endpoints_as_list = endpoints.split(',')
    shuffled_endpoints = []
    for da_hash in sorted(dc_idx_mapping.keys()):
      shuffled_endpoints.append(endpoints_as_list[dc_idx_mapping[da_hash]])
    return ",".join(shuffled_endpoints)


  hardware_feature = tf.tpu.experimental.HardwareFeature(resolver.tpu_hardware_feature)
  embedding_v2 = (tf.tpu.experimental.HardwareFeature.EmbeddingFeature.V2)

  shuffle_endpoints = os.environ.get('SHUFFLE_ENDPOINTS', 'false').lower() == 'true'

  enable_device_assignment = os.environ.get('ENABLE_DEVICE_ASSIGNMENT', 'false').lower() == 'true'

  print("====================================> run time configurations are as follows:")
  print(f'Shuffle endpoints: {shuffle_endpoints}')
  print(f'Enable device assignment: {enable_device_assignment}')
  print(f'Hardware feature: {hardware_feature}')


  if hardware_feature.embedding_feature == embedding_v2:
    tpu_system_metadata = resolver.get_tpu_system_metadata()
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(topology, num_replicas=tpu_system_metadata.num_cores)

    tpu_name = _get_grpc_endpoint_for_tpu_cluster_resolver()
    shuffled_endpoints = shuffle(tpu_name, device_assignment)
    print(f'Shuffled endpoints: {shuffled_endpoints}')

    if shuffle_endpoints and tpu_name != shuffled_endpoints:
      print('shuffling')
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=shuffled_endpoints)
      tf.config.experimental_connect_to_cluster(resolver)
      topology = tf.tpu.experimental.initialize_tpu_system(resolver)
  else:
    device_assignment = None

if use_cpu_strategy:
  print('using cpu strategy')
  strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
elif enable_device_assignment and not shuffle_endpoints:
  print('passing device assignment')
  strategy = tf.distribute.TPUStrategy(resolver, experimental_device_assignment=device_assignment)
else:
  print('not passing device assignment')
  strategy = tf.distribute.TPUStrategy(resolver)


PER_REPLICA_BATCH_SIZE = GLOBAL_BATCH_SIZE // strategy.num_replicas_in_sync
print(f'Per-replica batch size: {PER_REPLICA_BATCH_SIZE}')

ratings = tfds.load(
    'movielens/100k-ratings', split='train', data_dir=GCS_BUCKET, shuffle_files=False).map(
        lambda x: {
            'movie_id': tf.cast(tf.strings.to_number(x['movie_id']), tf.int32),
            'user_id': tf.cast(tf.strings.to_number(x['user_id']), tf.int32),
            'user_rating': x['user_rating']
        })


def prepare_dataset(split):
  """Prepare dataset for training or testing.

  Args:
    split: Either 'train' or 'test' to determine dataset split.q

  Returns:
    Prepared tf.data.Dataset with batching and distribution options.
  """
  ds = ratings.shuffle(SHUFFLE_BUF, seed=SEED, reshuffle_each_iteration=False)
  if split == 'train':
    ds = ds.take(TRAIN_SIZE)
  else:
    ds = ds.skip(TRAIN_SIZE).take(TEST_SIZE)
  batch_size = GLOBAL_BATCH_SIZE
  ds = ds.batch(batch_size, drop_remainder=True).cache().repeat()
  options = tf.data.Options()
  auto_shard = tf.data.experimental.AutoShardPolicy.DATA
  options.experimental_distribute.auto_shard_policy = auto_shard
  return ds.with_options(options)


train_ds = prepare_dataset('train')
test_ds = prepare_dataset('test')
input_options = tf.distribute.InputOptions(experimental_fetch_to_device=False)
dist_train_ds = strategy.experimental_distribute_dataset(train_ds, options=input_options)
dist_test_ds = strategy.experimental_distribute_dataset(test_ds, options=input_options)

optimizer = tf.keras.optimizers.legacy.Adagrad(learning_rate=0.1)

user_table = tf.tpu.experimental.embedding.TableConfig(vocabulary_size=USER_VOCAB_SIZE, dim=EMBED_DIM, name='user_id')
movie_table = tf.tpu.experimental.embedding.TableConfig(
    vocabulary_size=MOVIE_VOCAB_SIZE, dim=EMBED_DIM, name='movie_id')
feature_config = {
    'user_id':
        tf.tpu.experimental.embedding.FeatureConfig(
            table=user_table, output_shape=[PER_REPLICA_BATCH_SIZE], name='user_id'),
    'movie_id':
        tf.tpu.experimental.embedding.FeatureConfig(
            table=movie_table, output_shape=[PER_REPLICA_BATCH_SIZE], name='movie_id'),
}


class EmbeddingModel(tfrs.models.Model):
  """MovieLens embedding model for TPU training."""

  def __init__(self):
    """Initialize the embedding model."""
    super().__init__()
    self.embedding_layer = TPUEmbedding(
        feature_config, optimizer, batch_size=PER_REPLICA_BATCH_SIZE, pipeline_execution_with_tensor_core=True)
    self.ratings = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    self.task = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
        metrics=[tf.keras.metrics.RootMeanSquaredError()])

  def compute_loss(self, features, training=False):
    """Compute loss for the model.

    Args:
      features: Input features dictionary.
      training: Whether in training mode (unused but kept for compatibility).

    Returns:
      Computed loss value.
    """
    del training 
    emb = self.embedding_layer({'user_id': features['user_id'], 'movie_id': features['movie_id']})
    preds = self.ratings(tf.concat([emb['user_id'], emb['movie_id']], axis=1))
    return (tf.reduce_sum(self.task(labels=features['user_rating'], predictions=preds)) *
            (1 / (PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--training', action='store_true', help='Run training before evaluation')
  parser.add_argument(
      '--checkpoint-dir',
      type=str,
      default=os.path.join(GCS_BUCKET, 'checkpoints-v6e-16-tpu-emb-24'),
      help='Directory to save/load checkpoints (default: GCS_BUCKET/checkpoints-v6e-16-tpu-emb-24)')

  args = parser.parse_args()
  
  CHECKPOINT_DIR = args.checkpoint_dir

  with strategy.scope():
    model = EmbeddingModel()
    # model.compile(optimizer=optimizer)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=3)
    if manager.latest_checkpoint:
      checkpoint.restore(manager.latest_checkpoint).expect_partial()
      print(f'Restored from {manager.latest_checkpoint}')
    else:
      print('Initializing from scratch.')

  model.compile(optimizer=optimizer)

  if args.training:
    print('Starting training...')
    model.fit(dist_train_ds, steps_per_epoch=10, epochs=10)
    ckpt_path = manager.save()
    print(f'Checkpoint saved at: {ckpt_path}')

  steps = TEST_SIZE // GLOBAL_BATCH_SIZE
  print('Starting evaluation...')
  results = model.evaluate(dist_test_ds, steps=steps)
  print(f'Test evaluation results (loss, RMSE): {results}')
