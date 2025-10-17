# Copyright 2024 The TensorFlow Recommenders Authors.
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

"""An experimental ranking model."""

from typing import Dict, Optional, List

import tensorflow as tf

from tensorflow_recommenders import models
from tensorflow_recommenders.layers.embedding import tpu_embedding_layer
from tensorflow_recommenders.layers.feature_interaction import dcn
from tensorflow_recommenders.layers.feature_interaction import dot_interaction


class Ranking(models.Model):
  """A ranking model.

  This model is composed of three parts:
  - A bottom stack, which consists of a dense network that processes dense
  features and an embedding layer that processes sparse features.
  - A feature interaction layer, which can be one of DCN or DotInteraction.
  - A top stack, which is a dense network that combines the outputs of bottom
  stack and feature interaction layer and computes the final prediction.
  """

  def __init__(
      self,
      embedding_layer: tf.keras.layers.Layer,
      bottom_stack: tf.keras.layers.Layer,
      feature_interaction: tf.keras.layers.Layer,
      top_stack: tf.keras.layers.Layer,
      loss: Optional[tf.keras.losses.Loss] = None,
      metrics: Optional[List[tf.keras.metrics.Metric]] = None,
      optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
  ) -> None:
    """Initializes the model.

    Args:
      embedding_layer: A Keras layer that encodes sparse features.
      bottom_stack: A Keras layer that represents the bottom dense stack.
      feature_interaction: A Keras layer that represents the feature
        interaction.
      top_stack: A Keras layer that represents the top dense stack.
      loss: A Keras loss.
      metrics: A list of Keras metrics.
      optimizer: A Keras optimizer.
    """
    super().__init__()
    self._embedding_layer = embedding_layer
    self._bottom_stack = bottom_stack
    self._feature_interaction = feature_interaction
    self._top_stack = top_stack

    self._loss = loss if loss is not None else tf.keras.losses.BinaryCrossentropy(
        from_logits=True
    )

    self._metrics = metrics if metrics is not None else []

    self._optimizer = optimizer if optimizer is not None else tf.keras.optimizers.Adagrad(
        learning_rate=0.01
    )

  def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Executes forward pass.

    Args:
      inputs: A dictionary of tensors. The keys are feature names and values are
        feature tensors.

    Returns:
      A tensor that represents the ranking score.
    """
    dense_features, sparse_features = self._split_features(inputs)
    bottom_output = self._bottom_stack(dense_features)
    sparse_embeddings = self._embedding_layer(sparse_features)
    interaction_output = self._feature_interaction(
        [bottom_output] + sparse_embeddings
    )
    top_input = tf.concat([bottom_output, interaction_output], axis=1)
    return self._top_stack(top_input)

  def compute_loss(
      self,
      inputs: Dict[str, tf.Tensor],
      training: bool = False,
  ) -> tf.Tensor:
    """Computes the loss.

    Args:
      inputs: A dictionary of tensors. The keys are feature names and values are
        feature tensors. This dictionary must contain a "label" key.
      training: A boolean indicating if the model is in training mode.

    Returns:
      A tensor that represents the loss.
    """
    labels = inputs.pop("label")
    scores = self(inputs, training=training)
    return self._loss(labels, scores)

  def _split_features(
      self, inputs: Dict[str, tf.Tensor]
  ) -> (Dict[str, tf.Tensor], Dict[str, tf.Tensor]):
    """Splits features into dense and sparse features.

    Args:
      inputs: A dictionary of tensors. The keys are feature names and values are
        feature tensors.

    Returns:
      A tuple of two dictionaries, one for dense features and one for sparse
      features.
    """
    dense_features, sparse_features = {}, {}
    for key, value in inputs.items():
      if isinstance(value, tf.Tensor):
        dense_features[key] = value
      elif isinstance(value, tf.SparseTensor):
        sparse_features[key] = value
      else:
        raise ValueError(f"Unsupported feature type: {type(value)}")
    return dense_features, sparse_features
