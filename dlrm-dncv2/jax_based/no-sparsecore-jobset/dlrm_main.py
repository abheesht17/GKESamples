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
"""DLRM DCN v2 model training and evaluation."""

import collections
import functools
import os
import threading
import time
from typing import Any, Callable, List, Mapping

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from clu import metrics as clu_metrics
from dataloader import CriteoDataLoader
from dataloader import DataConfig
from dlrm_model import DLRMDCNV2
import flax
import jax
import jax.numpy as jnp
import jax.profiler
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import metrax
import numpy as np
import optax
import orbax.checkpoint as ocp


jax.distributed.initialize()
jax.profiler.start_server(9999)
partial = functools.partial
info = logging.info

FLAGS = flags.FLAGS

# --- Data and Model Flags ---
VOCAB_SIZES = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000, 40000000,
    40000000, 590152, 12973, 108, 36
]
MULTI_HOT_SIZES = [
    3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10,
    3, 1, 1
]
_NUM_DENSE_FEATURES = flags.DEFINE_integer(
    "num_dense_features", 13, "Number of dense features."
)
_EMBEDDING_SIZE = flags.DEFINE_integer("embedding_size", 16, "Embedding size.")

# --- Mode Flag ---
_MODE = flags.DEFINE_enum(
    "mode", "train", ["train", "eval"],
    "Mode to run: 'train' for training, 'eval' for evaluation-only."
)

# --- Training Flags ---
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 8192, "Batch size.")
_FILE_PATTERN = flags.DEFINE_string(
    "file_pattern", None, "File pattern for the training data."
)
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.0034, "Learning rate.")
_NUM_STEPS = flags.DEFINE_integer(
    "num_steps", 28000, "Number of steps to train for."
)
_LOGGING_INTERVAL = flags.DEFINE_integer(
    "logging_interval", 6000, "Frequency of logging training metrics."
)

# --- Evaluation Flags ---
_EVAL_FILE_PATTERN = flags.DEFINE_string(
    "eval_file_pattern", None, "File pattern for the evaluation data."
)
_EVAL_INTERVAL = flags.DEFINE_integer(
    "eval_interval", 5000, "Run evaluation every N steps."
)
_EVAL_STEPS = flags.DEFINE_integer(
    "eval_steps", 0, "Number of steps for each eval. 0 for all."
)

# --- Checkpointing and Misc Flags ---
_MODEL_DIR = flags.DEFINE_string(
    "model_dir", "/tmp/dlrm_jax", "Model working directory."
)
_SAVE_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    "save_checkpoint_interval", 5000, "Frequency of saving checkpoints."
)
_RESTORE_CHECKPOINT = flags.DEFINE_bool(
    "restore_checkpoint", False, "Restore from the latest checkpoint."
)


@flax.struct.dataclass
class TrainMetrics(clu_metrics.Collection):
  """Metrics for the training loop."""
  loss: metrax.Average
  accuracy: metrax.Accuracy


@flax.struct.dataclass
class EvalMetrics(clu_metrics.Collection):
  """Metrics for the evaluation loop."""
  loss: metrax.Average
  accuracy: metrax.Accuracy


class DLRMDataLoader:
  """Parallel data producer for the DLRM model."""

  def __init__(
      self,
      file_pattern: str,
      batch_size,
      is_training: bool,
      num_workers=4,
      buffer_size=128,
      global_sharding=None,
  ):
    """Initialize the producer."""
    self.data_config = DataConfig(
        global_batch_size=batch_size,
        is_training=is_training,
        use_cached_data=file_pattern is None,
    )
    self._dataloader = CriteoDataLoader(
        file_pattern=file_pattern,
        params=self.data_config,
        num_dense_features=_NUM_DENSE_FEATURES.value,
        vocab_sizes=VOCAB_SIZES,
        multi_hot_sizes=MULTI_HOT_SIZES,
    )
    self._iterator = self._dataloader.get_iterator()
    self.global_sharding = global_sharding

    self.buffer = collections.deque(maxlen=buffer_size)
    self._sync = threading.Condition()
    self._workers = []

    for _ in range(num_workers):
      worker = threading.Thread(target=self._worker_loop, daemon=True)
      worker.start()
      self._workers.append(worker)

  def process_inputs(self, feature_batch):
    """Process input features into the required format."""
    labels = feature_batch["clicked"]
    dense_features = feature_batch["dense_features"]
    sparse_features = feature_batch["sparse_features"]

    make_global_view = lambda x: jax.tree.map(
        lambda y: jax.make_array_from_process_local_data(
            self.global_sharding, y
        ),
        x,
    )
    labels = make_global_view(labels)
    dense_features = make_global_view(dense_features)
    sparse_features = make_global_view(sparse_features)
    return [labels, dense_features, sparse_features]

  def _worker_loop(self):
    """Worker thread that continuously generates and processes batches."""
    while True:
      try:
        batch = next(self._iterator)
        processed_batch = self.process_inputs(batch)
        with self._sync:
          self._sync.wait_for(lambda: len(self.buffer) < self.buffer.maxlen)
          self.buffer.append(processed_batch)
          self._sync.notify_all()
      except (StopIteration, AttributeError):
        with self._sync:
          self.buffer.append(None)
          self._sync.notify_all()
        return

  def __iter__(self):
    return self

  def __next__(self):
    """Get next batch from the buffer."""
    with self._sync:
      self._sync.wait_for(lambda: self.buffer)
      item = self.buffer.popleft()
      self._sync.notify_all()
      if item is None:
        raise StopIteration
      return item

  def stop(self):
    """Stop all worker threads and clear the buffer."""
    if hasattr(self, '_iterator'):
      del self._iterator


def eval_loop(
    eval_producer: DLRMDataLoader,
    eval_step_fn: Callable,
    params: Any,
    apply_fn: Callable,
    max_steps: int = 0,
):
  """Runs the evaluation loop."""
  info("Starting evaluation...")
  eval_metrics_collection = EvalMetrics.empty()
  step_count = 0
  for batch in eval_producer:
    labels, dense_features, sparse_features = batch
    eval_metrics_collection = eval_step_fn(
        apply_fn,
        params,
        labels,
        dense_features,
        sparse_features,
        eval_metrics_collection,
    )
    step_count += 1
    if max_steps > 0 and step_count >= max_steps:
      info("Reached max evaluation steps (%d).", max_steps)
      break
  
  info("Finished evaluation after %d steps.", step_count)
  metrics_on_host = jax.device_get(eval_metrics_collection)
  loss_val = metrics_on_host.loss.compute()
  accuracy_val = metrics_on_host.accuracy.compute()
  info(
      "Evaluation results: loss=%.5f, accuracy=%.5f",
      loss_val,
      accuracy_val,
  )


@partial(jax.jit, static_argnums=0)
def eval_step(
    apply_fn: Callable,
    params: Any,
    labels: jax.Array,
    dense_features: jax.Array,
    sparse_features: Mapping[str, jax.Array],
    metrics_collection,
):
  logits = apply_fn(params, dense_features, sparse_features)
  loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
  preds = jax.nn.sigmoid(logits)
  binarized_preds = (preds > 0.5).astype(jnp.int32)
  metric_updates = EvalMetrics.empty().replace(
      loss=metrax.Average.from_model_output(values=loss),
      accuracy=metrax.Accuracy.from_model_output(binarized_preds, labels),
  )
  return metrics_collection.merge(metric_updates)


def train_loop(
    model: DLRMDCNV2,
    global_sharding=None,
):
  """Main training and evaluation loop."""
  producer = DLRMDataLoader(
      file_pattern=_FILE_PATTERN.value,
      batch_size=_BATCH_SIZE.value,
      is_training=True,
      num_workers=16,
      buffer_size=256,
      global_sharding=global_sharding
  )

  _, dense_features, sparse_features = next(producer)
  params = model.init(
      jax.random.key(42), dense_features, sparse_features
  )
  tx = optax.adagrad(learning_rate=_LEARNING_RATE.value)
  opt_state = tx.init(params)
  
  checkpoint_dir = os.path.join(_MODEL_DIR.value, "checkpoints")
  checkpointer = ocp.CheckpointManager(checkpoint_dir)

  initial_step = 0
  if _RESTORE_CHECKPOINT.value:
    latest_step = checkpointer.latest_step()
    if latest_step is not None:
      info("Found checkpoint at step %d. Restoring...", latest_step)
      ckpt_target = {"params": params, "opt_state": opt_state, "step": 0}
      restored = checkpointer.restore(
          latest_step, args=ocp.args.PyTreeRestore(ckpt_target)
      )
      params = restored["params"]
      opt_state = restored["opt_state"]
      initial_step = restored["step"]
      info("Restored state from step %d.", initial_step)
    else:
      info(
          "No checkpoint found to restore from. Starting from scratch."
      )

  @partial(jax.jit, donate_argnums=(0, 4))
  def train_step_fn(
      params: Any,
      labels: jax.Array,
      dense_features: jax.Array,
      sparse_features: Mapping[str, jax.Array],
      opt_state,
      metrics_collection,
  ):
    def forward_pass(p, lbl, dense, sparse):
      logits = model.apply(p, dense, sparse)
      xentropy = optax.sigmoid_binary_cross_entropy(logits, lbl)
      return jnp.mean(xentropy), logits

    train_fn = jax.value_and_grad(forward_pass, has_aux=True)
    (loss_val, logits), grads = train_fn(
        params, labels, dense_features, sparse_features
    )
    preds = jax.nn.sigmoid(logits)
    binarized_preds = (preds > 0.5).astype(jnp.int32)
    metric_updates = TrainMetrics.empty().replace(
        loss=metrax.Average.from_model_output(values=loss_val),
        accuracy=metrax.Accuracy.from_model_output(binarized_preds, labels),
    )
    metrics_collection = metrics_collection.merge(metric_updates)
    updates, new_opt_state = tx.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, metrics_collection

  start_time = time.time()
  overall_start_time = time.time()
  train_metrics_collection = TrainMetrics.empty()
  total_eval_time_since_last_log = 0.0

  if _EVAL_FILE_PATTERN.value:
    eval_producer = DLRMDataLoader(
        file_pattern=_EVAL_FILE_PATTERN.value,
        batch_size=_BATCH_SIZE.value,
        is_training=False,
        num_workers=4,
        buffer_size=128,
        global_sharding=global_sharding,
    )

  for step in range(initial_step, _NUM_STEPS.value):
    with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
      labels, dense_features, sparse_features = next(producer)
      params, opt_state, train_metrics_collection = train_step_fn(
          params, labels, dense_features, sparse_features,
          opt_state, train_metrics_collection
      )

    current_step = step + 1
    if current_step % _LOGGING_INTERVAL.value == 0:
      end_time = time.time()
      metrics_on_host = jax.device_get(train_metrics_collection)
      elapsed_time = end_time - start_time
      train_time = elapsed_time - total_eval_time_since_last_log
      throughput = _BATCH_SIZE.value * _LOGGING_INTERVAL.value / train_time
      
      info(
          "Step %d: loss=%.5f, accuracy=%.5f, throughput=%.2f examples/sec",
          current_step, metrics_on_host.loss.compute(),
          metrics_on_host.accuracy.compute(), throughput
      )
      train_metrics_collection = TrainMetrics.empty()
      start_time = time.time()
      total_eval_time_since_last_log = 0.0

    if current_step % _EVAL_INTERVAL.value == 0 and _EVAL_FILE_PATTERN.value:
      eval_start_time = time.time()
      eval_loop(
          eval_producer,
          eval_step,
          params,
          model.apply,
          max_steps=_EVAL_STEPS.value,
      )
      eval_end_time = time.time()
      total_eval_time_since_last_log += (eval_end_time - eval_start_time)

    if current_step % _SAVE_CHECKPOINT_INTERVAL.value == 0:
      ckpt_to_save = {
          "params": params,
          "opt_state": opt_state,
          "step": current_step,
      }
      checkpointer.save(
          current_step, args=ocp.args.PyTreeSave(ckpt_to_save), force=True
      )

  overall_end_time = time.time()
  total_training_time = overall_end_time - overall_start_time
  total_steps_trained = _NUM_STEPS.value - initial_step
  total_examples_processed = total_steps_trained * _BATCH_SIZE.value
  overall_throughput = total_examples_processed / total_training_time
  info(
      "Finished training %d steps in %.2f seconds.",
      total_steps_trained, total_training_time
  )
  info("Overall training throughput: %.2f examples/sec", overall_throughput)
  
  producer.stop()
  if _EVAL_FILE_PATTERN.value:
    eval_producer.stop()
  checkpointer.wait_until_finished()
  checkpointer.close()


def run_evaluation_only(
    model: DLRMDCNV2,
    global_sharding: NamedSharding,
):
  """Runs evaluation on a saved checkpoint."""
  if not _EVAL_FILE_PATTERN.value:
    raise ValueError("--eval_file_pattern must be set in 'eval' mode.")

  checkpoint_dir = os.path.join(_MODEL_DIR.value, "checkpoints")
  checkpointer = ocp.CheckpointManager(checkpoint_dir)
  latest_step = checkpointer.latest_step()

  if latest_step is None:
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir} to eval.")

  info(
      "Found checkpoint at step %d. Restoring for evaluation...", latest_step
  )

  dummy_producer = DLRMDataLoader(
      file_pattern=_EVAL_FILE_PATTERN.value,
      batch_size=_BATCH_SIZE.value,
      is_training=False,
      num_workers=1,
      global_sharding=global_sharding,
  )
  _, dense_features, sparse_features = next(dummy_producer)

  params_structure = jax.eval_shape(
      lambda: model.init(
          jax.random.key(0), dense_features, sparse_features
      )
  )

  dummy_tx = optax.adagrad(learning_rate=0.0)
  opt_state_structure = jax.eval_shape(lambda: dummy_tx.init(params_structure))

  restore_target = {
      "params": params_structure,
      "opt_state": opt_state_structure,
      "step": 0,
  }

  restored_full_dict = checkpointer.restore(
      latest_step, args=ocp.args.PyTreeRestore(restore_target)
  )
  params = restored_full_dict["params"]

  info("Parameters restored. Starting evaluation...")

  eval_producer = DLRMDataLoader(
      file_pattern=_EVAL_FILE_PATTERN.value,
      batch_size=_BATCH_SIZE.value,
      is_training=False,
      num_workers=16,
      buffer_size=128,
      global_sharding=global_sharding,
  )

  eval_loop(
      eval_producer, eval_step, params, model.apply, max_steps=_EVAL_STEPS.value
  )

  eval_producer.stop()
  checkpointer.close()


def main(argv):
  del argv

  info("--- Starting DLRMv2 Training/Evaluation (No SparseCore) ---")
  info("--- Flag Values ---")
  for flag_name in FLAGS:
    info(f"{flag_name}: {FLAGS[flag_name].value}")
  info("--------------------")

  pd = P("x")
  global_devices = jax.devices()
  mesh = jax.sharding.Mesh(global_devices, "x")
  global_sharding = jax.sharding.NamedSharding(mesh, pd)

  model = DLRMDCNV2(
      global_batch_size=_BATCH_SIZE.value,
      embedding_size=_EMBEDDING_SIZE.value,
      bottom_mlp_dims=[512, 256, _EMBEDDING_SIZE.value],
      vocab_sizes=VOCAB_SIZES,
  )

  if _MODE.value == "train":
    train_loop(model, global_sharding)
  elif _MODE.value == "eval":
    run_evaluation_only(model, global_sharding)


if __name__ == "__main__":
  app.run(main)