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
"""A simple DLRM model implementation in Flax."""

from typing import Any, Callable, List, Mapping, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.flax import embed
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec

# Type definition for the initializer function
Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


def uniform_init(bound: float) -> Initializer:
  """Creates a uniform initializer with a given bound."""

  def init_fn(
      key: jnp.ndarray, shape: Sequence[int], dtype: jnp.dtype = jnp.float32
  ) -> jnp.ndarray:
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

  return init_fn


class DenseBlock(nn.Module):
  """A block of dense layers with optional batch normalization and activation."""

  width: int
  use_batchnorm: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
    y = nn.Dense(
        features=self.width,
        kernel_init=nn.initializers.glorot_uniform(),
        bias_init=nn.initializers.normal(stddev=jnp.sqrt(1.0 / self.width)),
    )(x)
    if self.use_batchnorm:
      y = nn.BatchNorm(use_running_average=not is_training)(y)
    return nn.relu(y)


class MLP(nn.Module):
  """A multi-layer perceptron (MLP) with configurable dimensions."""

  dims: Sequence[int]
  use_batchnorm: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
    for width in self.dims[:-1]:
      x = DenseBlock(width=width, use_batchnorm=self.use_batchnorm)(
          x, is_training
      )
    return nn.Dense(features=self.dims[-1])(x)


class DLRMInteraction(nn.Module):
  """DLRM feature interaction layer.

  This layer performs dot-product interactions between feature embeddings.
  """

  @nn.compact
  def __call__(self, bottom_mlp_output: jnp.ndarray,
               embedding_outputs: List[jnp.ndarray]) -> jnp.ndarray:
    """Computes feature interactions.

    Args:
      bottom_mlp_output: The output from the bottom MLP.
      embedding_outputs: A list of embedding vectors.

    Returns:
      The concatenated tensor of interaction results and the bottom MLP output.
    """
    # Concatenate all features to form the interaction tensor
    # Shape: (batch_size, num_features, embedding_dim)
    all_features = jnp.stack([bottom_mlp_output] + embedding_outputs, axis=1)

    # Perform dot product interactions
    # 1. Take the transpose to get (batch_size, embedding_dim, num_features)
    # 2. Perform batch matrix multiplication: (num_features x emb_dim) @ (emb_dim x num_features)
    # This results in a (batch_size, num_features, num_features) interaction matrix.
    interactions = jnp.matmul(all_features, all_features.transpose((0, 2, 1)))

    # Get the lower triangular part of the interaction matrix (including diagonal)
    # and flatten it to get the interaction features.
    batch_size, num_features, _ = interactions.shape
    indices = jnp.tril_indices(num_features)
    interaction_flat = interactions[:, indices[0], indices[1]]

    # Concatenate the dense features (from bottom MLP) with the interaction results
    return jnp.concatenate([bottom_mlp_output, interaction_flat], axis=1)


class DLRM(nn.Module):
  """A standard DLRM model implementation.

  This model consists of a bottom MLP for dense features, embedding tables for
  sparse features, a dot-product interaction layer, and a top MLP.
  """

  feature_specs: Mapping[str, embedding_spec.FeatureSpec]
  mesh: Any
  sharding_axis: str
  global_batch_size: int
  embedding_size: int
  bottom_mlp_dims: Sequence[int]
  top_mlp_dims: Sequence[int]
  vocab_sizes: Sequence[int]
  use_batchnorm_for_bottom_mlp: bool = False
  use_batchnorm_for_top_mlp: bool = False

  @nn.compact
  def __call__(
      self,
      dense_features: jnp.ndarray,
      dense_lookups: Mapping[str, jnp.ndarray],
      embedding_lookups: embed.EmbeddingLookupInput,
      is_training: bool = True,
  ) -> jnp.ndarray:
    # --- Bottom MLP for Dense Features ---
    bottom_mlp = MLP(
        dims=self.bottom_mlp_dims,
        use_batchnorm=self.use_batchnorm_for_bottom_mlp,
        name="bottom_mlp",
    )
    dense_emb = bottom_mlp(dense_features, is_training)

    # --- Embedding Lookups ---
    all_embeddings = []

    # SparseCore Embeddings (for large tables)
    sc_embeddings = embed.SparseCoreEmbedding(
        feature_specs=self.feature_specs,
        mesh=self.mesh,
        sharding_axis=self.sharding_axis,
        name="sc_embedding",
    )(embedding_lookups)
    all_embeddings.extend(list(sc_embeddings.values()))

    # TensorCore Embeddings (for small tables)
    for i, vocab_size in enumerate(self.vocab_sizes):
      if vocab_size <= 21000:  # Using the threshold from dlrm_main.py
        tc_emb_table = self.param(
            f"tc_emb_{i}",
            nn.initializers.uniform(scale=1 / jnp.sqrt(vocab_size)),
            (vocab_size, self.embedding_size),
        )
        all_embeddings.append(jnp.take(tc_emb_table, dense_lookups[str(i)], axis=0).squeeze())

    # --- Feature Interaction ---
    interaction_output = DLRMInteraction(name="interaction")(
        dense_emb, all_embeddings
    )

    # --- Top MLP ---
    top_mlp = MLP(
        dims=self.top_mlp_dims,
        use_batchnorm=self.use_batchnorm_for_top_mlp,
        name="top_mlp",
    )
    logits = top_mlp(interaction_output, is_training)

    return logits.squeeze()

