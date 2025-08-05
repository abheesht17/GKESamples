# jobset/dataloader.py (Corrected and Final Version)

import tensorflow as tf
from typing import Dict, List

class DataConfig:
    def __init__(self, global_batch_size: int, is_training: bool, use_cached_data: bool):
        self.global_batch_size = global_batch_size
        self.is_training = is_training
        self.use_cached_data = use_cached_data

class CriteoDataLoader:
    """Data loader for Criteo dataset, corrected to follow TF best practices."""

    def __init__(
        self,
        file_pattern: str,
        params: DataConfig,
        num_dense_features: int,
        vocab_sizes: List[int],
        multi_hot_sizes: List[int],
        embedding_threshold: int = 21000,
        shuffle_buffer: int = 1024,
        prefetch_size: int = tf.data.AUTOTUNE,
    ):
        self._file_pattern = file_pattern
        self._params = params
        self._num_dense_features = num_dense_features
        self._vocab_sizes = vocab_sizes
        self._multi_hot_sizes = multi_hot_sizes
        self._embedding_threshold = embedding_threshold
        self._shuffle_buffer = shuffle_buffer
        self._prefetch_size = prefetch_size

        self.label_features = 'clicked'
        self.dense_features = [f'int-feature-{x}' for x in range(1, 14)]
        self.sparse_features = [f'categorical-feature-{x}' for x in range(14, 40)]

    # --- THE FIX IS HERE: Removed the incorrect return type hint ---
    def _get_feature_spec(self):
        """Gets the feature specification for parsing a SINGLE TFRecord example."""
        feature_spec = {
            self.label_features: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=0)
        }
        for dense_ft in self.dense_features:
            feature_spec[dense_ft] = tf.io.FixedLenFeature(shape=[1], dtype=tf.float32, default_value=0.0)
        for sparse_ft in self.sparse_features:
            feature_spec[sparse_ft] = tf.io.VarLenFeature(dtype=tf.int64)
        return feature_spec

    def _parse_example(
            self, serialized_example: tf.Tensor, batch_size: int
        ) -> Dict[str, tf.Tensor]:
        """Parses a single serialized TFRecord example into features."""
        feature_spec = self._get_feature_spec()
        parsed_features = tf.io.parse_single_example(serialized_example, feature_spec)

        labels = parsed_features[self.label_features]

        dense_features_map = {}
        for i, dense_ft_name in enumerate(self.dense_features):
            dense_features_map[str(i+1)] = parsed_features[dense_ft_name]

        sparse_features_map = {}
        for i, sparse_ft_name in enumerate(self.sparse_features):
            sparse_tensor = parsed_features[sparse_ft_name]
            dense_tensor = tf.sparse.to_dense(sparse_tensor, default_value=0)
            reshaped_tensor = tf.reshape(dense_tensor, [-1, self._multi_hot_sizes[i]])
            sparse_features_map[str(i)] = reshaped_tensor

        return {
            'clicked': labels,
            'dense_features': dense_features_map,
            'sparse_features': sparse_features_map,
        }

    def _create_dataset(self) -> tf.data.Dataset:
        """Creates the dataset pipeline: ListFiles -> Interleave -> Parse -> Batch -> Prefetch."""
        files = tf.data.Dataset.list_files(self._file_pattern, shuffle=self._params.is_training)
        
        dataset = files.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not self._params.is_training
        )

        if self._params.is_training:
            dataset = dataset.shuffle(self._shuffle_buffer)

        dataset = dataset.map(self._parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self._params.global_batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._prefetch_size)

        if self._params.use_cached_data:
            dataset = dataset.cache()
            
        return dataset

    def get_iterator(self) -> tf.data.Iterator:
        """Returns an iterator for the dataset."""
        return iter(self._create_dataset())
