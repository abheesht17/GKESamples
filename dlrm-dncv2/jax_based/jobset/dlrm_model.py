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
"""Flax modules for DLRM DCNv2 model."""

from typing import Any, List, Mapping, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax_tpu_embedding.sparsecore.lib.flax import embed
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec


def uniform_init(bound: float):
  """Uniform initializer."""

  def init(key, shape, dtype=jnp.float64):
    return jax.random.uniform(
        key, shape=shape, dtype=dtype, minval=-bound, maxval=bound
    )

  return init


class MLP(nn.Module):
  """A multi-layer perceptron module."""

  dims: Sequence[int]
  sharding_axis: str = "x"

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    kernel_init = jax.nn.initializers.glorot_uniform()
    bias_init = jax.nn.initializers.normal(stddev=jnp.sqrt(1.0 / self.dims[-1]))

    for i, dim in enumerate(self.dims):
      x = nn.Dense(
          features=dim,
          kernel_init=kernel_init,
          bias_init=bias_init,
      )(x)
      if i < len(self.dims) - 1:
        x = nn.relu(x)
    return x


class DLRMDCNV2(nn.Module):
  """DLRM DCNv2 model."""

  feature_specs: Mapping[str, embedding_spec.FeatureSpec]
  mesh: Any
  sharding_axis: str
  global_batch_size: int
  embedding_size: int
  bottom_mlp_dims: List[int]
  vocab_sizes: List[int]
  num_dense_features: int = 13

  def setup(self):
    self.bottom_mlp = MLP(
        dims=self.bottom_mlp_dims, sharding_axis=self.sharding_axis
    )
    self.top_mlp = MLP(
        dims=[1024, 1024, 512, 256, 1], sharding_axis=self.sharding_axis
    )

    self.dense_embeddings = [
        nn.Embed(
            num_embeddings=vocab_size,
            features=self.embedding_size,
            embedding_init=jax.nn.initializers.normal(),
        )
        if vocab_size <= 21000
        else None
        for vocab_size in self.vocab_sizes
    ]

    self.sparse_embedder = embed.SparseCoreEmbed(
        feature_specs=self.feature_specs,
        mesh=self.mesh,
        sharding_axis=self.sharding_axis,
    )

  def __call__(
      self, dense_features, dense_lookups, embedding_lookups
  ) -> jnp.ndarray:
    dense_bot_mlp = self.bottom_mlp(dense_features)
    dense_bot_mlp = jnp.expand_dims(dense_bot_mlp, axis=1)

    sparse_embeddings_dict = self.sparse_embedder(embedding_lookups)

    sparse_embeddings = list(sparse_embeddings_dict.values())

    dense_embed_lookups = []
    for i, vocab_size in enumerate(self.vocab_sizes):
      if vocab_size <= 21000:
        dense_embed_lookups.append(
            jnp.expand_dims(self.dense_embeddings[i](dense_lookups[str(i)]), 1)
        )

    all_embeddings = [dense_bot_mlp] + sparse_embeddings + dense_embed_lookups
    x = jnp.concatenate(all_embeddings, axis=1)
    z = self.top_mlp(jnp.reshape(x, (self.global_batch_size, -1)))
    return jnp.squeeze(z, -1)

