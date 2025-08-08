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
"""DLRM DCN v2 model."""

from typing import List, Dict

from flax import linen as nn
import jax
import jax.numpy as jnp


def uniform_init(bound: float):
  def init(key, shape, dtype=jnp.float_):
    return jax.random.uniform(
        key,
        shape=shape,
        dtype=dtype,
        minval=-bound,
        maxval=bound
    )
  return init


class DLRMDCNV2(nn.Module):
  """DLRM DCN v2 model."""
  vocab_sizes: List[int]
  embedding_size: int
  bottom_mlp_dims: List[int]
  global_batch_size: int
  top_mlp_dims = [1024, 1024, 512, 256, 1]
  dcn_layers: int = 3
  projection_dim: int = 512

  def setup(self):
    self.embedding_layers = [
        nn.Embed(vocab_size, self.embedding_size)
        for vocab_size in self.vocab_sizes
    ]

  @nn.remat
  def bottom_mlp(self, x):
    for dim in self.bottom_mlp_dims:
      previous_dim = x.shape[-1]
      bound = jnp.sqrt(1.0 / previous_dim)
      x = nn.Dense(
          dim,
          kernel_init=uniform_init(bound),
          bias_init=uniform_init(bound),
      )(x)
      x = nn.relu(x)
    return x

  @nn.remat
  def top_mlp(self, x):
    previous_dim = x.shape[-1]
    for dim in self.top_mlp_dims[:-1]:
      bound = jnp.sqrt(1.0 / previous_dim)
      x = nn.Dense(
          dim,
          kernel_init=uniform_init(bound),
          bias_init=uniform_init(bound),
      )(x)
      x = nn.relu(x)
      previous_dim = dim

    bound = jnp.sqrt(1.0 / previous_dim)
    x = nn.Dense(
        self.top_mlp_dims[-1],
        kernel_init=uniform_init(bound),
        bias_init=uniform_init(bound),
    )(x)
    x = nn.sigmoid(x)
    return x

  @nn.remat
  def dcn_layer(self, x0):
    xl = x0
    input_dim = x0.shape[-1]

    for i in range(self.dcn_layers):
      u_kernel = self.param(
          f'u_kernel_{i}',
          nn.initializers.xavier_normal(),
          (input_dim, self.projection_dim),
      )
      v_kernel = self.param(
          f'v_kernel_{i}',
          nn.initializers.xavier_normal(),
          (self.projection_dim, input_dim),
      )
      bias = self.param(f'bias_{i}', nn.initializers.zeros, (input_dim,))

      u_output = jnp.matmul(xl, u_kernel)
      v_output = jnp.matmul(u_output, v_kernel)
      v_output += bias

      xl = x0 * v_output + xl

    return xl

  @nn.compact
  def __call__(
      self, dense_features: jax.Array, sparse_features: Dict[str, jax.Array]
  ):
    dense_outputs = self.bottom_mlp(dense_features)

    embedding_outputs = []
    for i, (key, value) in enumerate(sparse_features.items()):
        embeddings = self.embedding_layers[i](value)
        embeddings = jnp.sum(embeddings, axis=-2)
        embedding_outputs.append(embeddings)

    stacked_embeddings = jnp.stack(embedding_outputs, axis=1)

    interaction_args = jax.lax.concatenate(
        [
            dense_outputs.reshape(
                (self.global_batch_size, 1, self.embedding_size)
            ),
            stacked_embeddings,
        ],
        dimension=1,
    )
    interaction_args = interaction_args.reshape((self.global_batch_size, -1))
    interaction_outputs = self.dcn_layer(interaction_args)
    predictions = self.top_mlp(interaction_outputs)
    predictions = jnp.reshape(predictions, (-1,))

    return predictions

