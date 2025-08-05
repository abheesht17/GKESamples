# Copyright 2024 RecML authors <recommendations-ml@google.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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
from typing import Any, Dict, List

import jax
import numpy as np
import tensorflow as tf

dataclass = dataclasses.dataclass
PARALLELISM = tf.data.AUTOTUNE


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
      embedding_threshold: int,
      shuffle_buffer: int = 256,
      prefetch_size: int = 256,
  ):
    self._file_pattern = file_pattern
    print(f' file_pattern: {file_pattern}')
    self._params = params
    self._num_dense_features = num_dense_features
    self._vocab_sizes = vocab_sizes
    self._multi_hot_sizes = multi_hot_sizes
    self._embedding_threshold = embedding_threshold
    self._shuffle_buffer = shuffle_buffer
    self._prefetch_size = prefetch_size

    self.label_feature_name = 'label'
    self.dense_feature_names = [f'dense-feature-{x}' for x in range(1, 14)]
    self.sparse_feature_names = [f'sparse-feature-{x}' for x in range(14, 40)]

  def _get_feature_spec(self) -> Dict[str, Any]:
    """Creates the feature specification for parsing single TFRecord examples."""
    feature_spec = {
        self.label_feature_name: tf.io.FixedLenFeature(
            [1], dtype=tf.int64, default_value=0
        )
    }
    for feat_name in self.dense_feature_names:
      feature_spec[feat_name] = tf.io.FixedLenFeature(
          [1], dtype=tf.float32, default_value=0.0
      )
    for feat_name in self.sparse_feature_names:
      feature_spec[feat_name] = tf.io.VarLenFeature(dtype=tf.int64)
    return feature_spec

  def _parse_example(
      self, serialized_example: tf.Tensor
  ) -> Dict[str, tf.Tensor]:
    """Parses a single serialized TFRecord example."""
    feature_spec = self._get_feature_spec()
    parsed_features = tf.io.parse_single_example(
        serialized_example, feature_spec
    )

    labels = tf.cast(parsed_features[self.label_feature_name], tf.int32)

    dense_features_list = [
        parsed_features[name] for name in self.dense_feature_names
    ]
    dense_features = tf.concat(dense_features_list, axis=-1)

    sparse_features = {}
    for i, sparse_ft_name in enumerate(self.sparse_feature_names):
      sparse_tensor = parsed_features[sparse_ft_name]
      dense_tensor = tf.sparse.to_dense(sparse_tensor, default_value=0)
      pad_size = self._multi_hot_sizes[i] - tf.shape(dense_tensor)[0]
      paddings = [[0, tf.maximum(0, pad_size)]]
      padded_tensor = tf.pad(dense_tensor, paddings, 'CONSTANT')
      final_tensor = tf.slice(padded_tensor, [0], [self._multi_hot_sizes[i]])
      sparse_features[str(i)] = final_tensor

    return {
        'clicked': labels,
        'dense_features': dense_features,
        'sparse_features': sparse_features,
    }

  def _create_dataset(self) -> tf.data.Dataset:
    """Creates and configures the TensorFlow dataset."""
    dataset = tf.data.Dataset.list_files(
        self._file_pattern, shuffle=self._params.is_training
    )
    dataset = dataset.shard(jax.process_count(), jax.process_index())

    if self._params.is_training:
      dataset = dataset.repeat()

    dataset = tf.data.TFRecordDataset(
        dataset,
        buffer_size=32 * 1024 * 1024,
        num_parallel_reads=PARALLELISM,
    )

    if self._params.is_training:
      dataset = dataset.shuffle(self._shuffle_buffer)

    dataset = dataset.map(
        self._parse_example,
        num_parallel_calls=PARALLELISM,
    )

    per_process_batch_size = (
        self._params.global_batch_size // jax.process_count()
    )
    dataset = dataset.batch(per_process_batch_size, drop_remainder=True)

    dataset = dataset.prefetch(self._prefetch_size)
    options = tf.data.Options()
    options.deterministic = False
    options.threading.private_threadpool_size = 48
    dataset = dataset.with_options(options)
    return dataset

  def get_iterator(self):
    """Returns an iterator over the dataset that provides NumPy arrays."""
    dataset = self._create_dataset()
    return dataset.as_numpy_iterator()

