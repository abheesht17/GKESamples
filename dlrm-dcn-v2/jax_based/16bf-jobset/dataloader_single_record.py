# Copyright 2024 RecML authors <recommendations-ml@google.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data loader for Criteo dataset optimized for JAX training."""

import dataclasses
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

dataclass = dataclasses.dataclass
# Use TF Autotune for optimal performance
PARALELLISM = tf.data.AUTOTUNE


@dataclass
class DataConfig:
  """Configuration for data loading parameters."""
  global_batch_size: int
  is_training: bool
  use_cached_data: bool = False


class CriteoDataLoader:
  """Data loader for Criteo dataset optimized for JAX training."""

  def __init__(
      self,
      file_pattern: str,
      params: DataConfig,
      num_dense_features: int,
      vocab_sizes: List[int],
      multi_hot_sizes: List[int],
      embedding_threshold: int = 21000,
      shuffle_buffer: int = 256,
      prefetch_size: int = 256,
  ):
    self._file_pattern = file_pattern
    self._params = params
    self._num_dense_features = num_dense_features
    self._vocab_sizes = vocab_sizes
    self._multi_hot_sizes = multi_hot_sizes
    self._embedding_threshold = embedding_threshold
    self._shuffle_buffer = shuffle_buffer
    self._prefetch_size = prefetch_size

    # Use the correct feature names identified by the debugger.
    self.label_features = "label"
    self.dense_features = [f"dense-feature-{i}" for i in range(1, 14)]
    self.sparse_features = [f"sparse-feature-{i}" for i in range(14, 40)]

  def _get_feature_spec(self) -> Dict[str, tf.io.FixedLenFeature]:
    """Creates the feature specification for parsing a SINGLE TFRecord."""
    feature_spec = {
        self.label_features: tf.io.FixedLenFeature([1], dtype=tf.int64)
    }
    for dense_feat in self.dense_features:
      feature_spec[dense_feat] = tf.io.FixedLenFeature([1], dtype=tf.float32)

    for sparse_feat in self.sparse_features:
      feature_spec[sparse_feat] = tf.io.VarLenFeature(dtype=tf.int64)
    return feature_spec

  def _parse_example(self, serialized_example: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Parses a single serialized TFRecord example."""
    feature_spec = self._get_feature_spec()
    parsed_features = tf.io.parse_single_example(serialized_example, feature_spec)

    labels = parsed_features[self.label_features]
    dense_features_list = [parsed_features[feat] for feat in self.dense_features]
    dense_features = tf.concat(dense_features_list, axis=-1)

    sparse_features_map = {}
    for i, sparse_ft_name in enumerate(self.sparse_features):
      sparse_tensor = parsed_features[sparse_ft_name]
      dense_tensor = tf.sparse.to_dense(sparse_tensor, default_value=0)
      # Pad multi-hot features to a fixed size and cast to int32.
      padded_tensor = tf.pad(
          dense_tensor,
          [[0, self._multi_hot_sizes[i] - tf.shape(dense_tensor)[0]]],
      )
      sparse_features_map[str(i)] = tf.cast(padded_tensor, dtype=tf.int32)

    return {
        "clicked": labels,
        "dense_features": dense_features,
        "sparse_features": sparse_features_map,
    }

  def _create_dataset(self) -> tf.data.Dataset:
    """Creates and configures the TensorFlow dataset."""
    batch_size = self._params.global_batch_size // jax.process_count()
    
    files = tf.data.Dataset.list_files(self._file_pattern, shuffle=self._params.is_training)
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, buffer_size=32 * 1024 * 1024),
        cycle_length=PARALELLISM,
        num_parallel_calls=PARALELLISM,
        deterministic=False,
    )
    
    if self._params.is_training:
        dataset = dataset.shuffle(self._shuffle_buffer)
        dataset = dataset.repeat()

    dataset = dataset.map(self._parse_example, num_parallel_calls=PARALELLISM)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=PARALELLISM)
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    return dataset

  def get_iterator(self):
    """Returns an iterator over the dataset that provides NumPy arrays."""
    dataset = self._create_dataset()
    def _convert_to_numpy(batch):
      # Squeeze the label dimension after batching
      return {
          'clicked': np.squeeze(batch['clicked'].numpy()),
          'dense_features': batch['dense_features'].numpy(),
          'sparse_features': {
              k: v.numpy() for k, v in batch['sparse_features'].items()
          },
      }
    return map(_convert_to_numpy, iter(dataset))
