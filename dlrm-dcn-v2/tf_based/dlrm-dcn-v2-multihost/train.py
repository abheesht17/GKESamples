# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Train and evaluate the Ranking model."""

import time
from typing import Dict, Mapping, Optional

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tf_keras

# --- Imports for Monkey-Patching ---
# This block contains the specific, low-level imports needed for the patch to work.
# CORRECTED: Importing 'restore' instead of 'trackable_reader'.
from tensorflow.python.checkpoint import restore
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
# --- End of Imports for Monkey-Patching ---


from official.common import distribute_utils
from official.core import base_trainer
from official.core import train_lib
from official.core import train_utils
from official.recommendation.ranking import common
from official.recommendation.ranking.task import RankingTask
from official.utils.misc import keras_utils

FLAGS = flags.FLAGS

# --- Start of Monkey-Patching Block for Checkpoint SAVE Timing ---
from tensorflow.python.checkpoint import functional_saver

# 1. Save a reference to the original save function
original_saver_save = functional_saver.MultiDeviceSaver.save

# 2. Define the new function with timing logic
def timed_saver_save(self, file_prefix, options=None):
  """
  A wrapper around the original MultiDeviceSaver.save method to log the
  precise time of the actual file write operation.
  """
  logging.info(
      f"TIMING (SAVE): Preparing to write checkpoint data to prefix '{file_prefix}'..."
  )
  start_time = time.time()

  # Call the original function to perform the actual save
  result = original_saver_save(self, file_prefix, options=options)

  duration = time.time() - start_time
  logging.info(
      f"✅ TIMING (SAVE): Actual I/O write for checkpoint to '{file_prefix}' "
      f"took {duration:.4f} seconds."
  )
  return result

# 3. Apply the patch
functional_saver.MultiDeviceSaver.save = timed_saver_save

logging.info("Monkey-patch for checkpoint SAVE timing has been successfully applied.")
# --- End of Monkey-Patching Block ---

# --- Start of Monkey-Patching Block for Checkpoint Timing ---

# 1. Save a reference to the original function
original_value_tensors = restore.CheckpointPosition.value_tensors

# 2. Define the new function with timing logic
def timed_value_tensors(
    self, shape_and_slices: Optional[str] = None
) -> Mapping[str, tensor.Tensor]:
  """
  A wrapper around the original value_tensors method to log the
  precise time of the io_ops.restore_v2 call.
  """
  value_tensors = {}
  for serialized_tensor in self.object_proto.attributes:
    checkpoint_key = serialized_tensor.checkpoint_key
    io_device = self._checkpoint.options.experimental_io_device or "cpu:0"
    with ops.init_scope():
      with ops.device(io_device):
        if (
            shape_and_slices is not None
            and serialized_tensor.name in shape_and_slices
        ):
          shape_and_slice = shape_and_slices[serialized_tensor.name]
        else:
          shape_and_slice = ""
        checkpoint_keys, full_shape_and_slices = (
            self.callback.update_restore_inputs(
                checkpoint_key, shape_and_slice
            )
        )
        dtypes = []
        for key in checkpoint_keys:
          dtype = self._checkpoint.dtype_map[key]
          dtypes.append(dtype.base_dtype)

        # --- TIMING LOGIC ADDED HERE ---
        logging.info(
            f"TIMING: Preparing to restore tensor '{serialized_tensor.name}' "
            f"with key '{checkpoint_key}'..."
        )
        start_time = time.time()

        restored_values = io_ops.restore_v2(
            prefix=self._checkpoint.save_path_tensor,
            tensor_names=checkpoint_keys,
            shape_and_slices=full_shape_and_slices,
            dtypes=dtypes,
            name="%s_checkpoint_read" % (serialized_tensor.name,),
        )

        duration = time.time() - start_time
        logging.info(
            f"✅ TIMING: Restore of tensor '{serialized_tensor.name}' "
            f"took {duration:.4f} seconds."
        )
        # --- END OF TIMING LOGIC ---

        value = self.callback.reshard(
            restored_values, shape_and_slice
        )
      value_tensors[serialized_tensor.name] = array_ops.identity(value)
  return value_tensors

# 3. Apply the patch: Replace the original function with our new timed version
# CORRECTED: Targeting the 'value_tensors' method of the 'CheckpointPosition' class.
restore.CheckpointPosition.value_tensors = timed_value_tensors

logging.info("Monkey-patch for checkpoint restore timing has been successfully applied.")

# --- End of Monkey-Patching Block ---


class RankingTrainer(base_trainer.Trainer):
  """A trainer for Ranking Model.

  The RankingModel has two optimizers for embedding and non embedding weights.
  Overriding `train_loop_end` method to log learning rates for each optimizer.
  """

  def train_loop_end(self) -> Dict[str, float]:
    """See base class."""
    self.join()
    logs = {}
    for metric in self.train_metrics + [self.train_loss]:
      logs[metric.name] = metric.result()
      metric.reset_states()

    for i, optimizer in enumerate(self.optimizer.optimizers):
      lr_key = f'{type(optimizer).__name__}_{i}_learning_rate'
      if callable(optimizer.learning_rate):
        logs[lr_key] = optimizer.learning_rate(self.global_step)
      else:
        logs[lr_key] = optimizer.learning_rate
    return logs


# --- New class to add timing to the Orbit training path ---
class TimedRankingTrainer(RankingTrainer):
  """A RankingTrainer that measures checkpoint loading and saving time."""

  def initialize(self):
    """Overrides the base trainer's initialize method to add timing."""
    latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)

    if latest_checkpoint:
        logging.info('Starting to load checkpoint from gcs/gcsfuse (Orbit path): %s',
                     latest_checkpoint)
        start_time = time.time()
        super().initialize() # Perform the actual restoration.
        duration = time.time() - start_time
        logging.info('✅ Checkpoint loaded successfully.')
        logging.info(
            '⏰ Time to load checkpoint from gcs/gcsfuse (Orbit path): %.4f seconds',
            duration)
    else:
        logging.info('No checkpoint found, initializing from scratch (Orbit path).')
        super().initialize()

  def save_checkpoint(self):
    """Overrides the base trainer's save_checkpoint method to add timing."""
    logging.info('Starting to save checkpoint (Orbit path)...')
    start_time = time.time()
    super().save_checkpoint() # Perform the actual save.
    duration = time.time() - start_time
    logging.info(
        '⏰ Time to save checkpoint (Orbit path): %.4f seconds', duration)


# --- New class to add step-based checkpoint save timing to the compile/fit path ---
class TimedStepCheckpoint(tf_keras.callbacks.Callback):
  """A Keras Callback that saves checkpoints at step intervals and logs the time."""

  def __init__(self, checkpoint_manager: tf.train.CheckpointManager):
    """Initializes the callback.

    Args:
      checkpoint_manager: A `tf.train.CheckpointManager` instance.
    """
    super().__init__()
    self._checkpoint_manager = checkpoint_manager

  def on_train_batch_end(self, batch, logs=None):
    """Overrides on_train_batch_end to save a checkpoint at the correct step."""
    # Get the current step from the checkpoint manager's tracked step counter.
    step = self._checkpoint_manager._step_counter.numpy()
    interval = self._checkpoint_manager.checkpoint_interval
    # Check if a checkpoint should be saved at this step.
    if interval and step > 0 and step % interval == 0:
      logging.info('Starting to save checkpoint (Compile/Fit path) at step %d...',
                   step)
      start_time = time.time()
      self._checkpoint_manager.save()
      duration = time.time() - start_time
      logging.info(
          '⏰ Time to save checkpoint (Compile/Fit path): %.4f seconds',
          duration)


def main(_) -> None:
  """Train and evaluate the Ranking model."""
  params = train_utils.parse_configuration(FLAGS)
  mode = FLAGS.mode
  model_dir = FLAGS.model_dir
  if 'train' in FLAGS.mode:
    train_utils.serialize_config(params, model_dir)

  if FLAGS.seed is not None:
    logging.info('Setting tf seed.')
    tf.random.set_seed(FLAGS.seed)

  task = RankingTask(
      params=params.task,
      trainer_config=params.trainer,
      logging_dir=model_dir,
      steps_per_execution=params.trainer.steps_per_loop,
      name='RankingTask')

  enable_tensorboard = params.trainer.callbacks.enable_tensorboard

  strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu)

  with strategy.scope():
    model = task.build_model()

  def get_dataset_fn(params):
    return lambda input_context: task.build_inputs(params, input_context)

  train_dataset = None
  if 'train' in mode:
    train_dataset = strategy.distribute_datasets_from_function(
        get_dataset_fn(params.task.train_data),
        options=tf.distribute.InputOptions(experimental_fetch_to_device=False))

  validation_dataset = None
  if 'eval' in mode:
    validation_dataset = strategy.distribute_datasets_from_function(
        get_dataset_fn(params.task.validation_data),
        options=tf.distribute.InputOptions(experimental_fetch_to_device=False))

  if params.trainer.use_orbit:
    with strategy.scope():
      checkpoint_exporter = train_utils.maybe_create_best_ckpt_exporter(
          params, model_dir)
      trainer = TimedRankingTrainer(
          config=params,
          task=task,
          model=model,
          optimizer=model.optimizer,
          train='train' in mode,
          evaluate='eval' in mode,
          train_dataset=train_dataset,
          validation_dataset=validation_dataset,
          checkpoint_exporter=checkpoint_exporter)

    train_lib.run_experiment(
        distribution_strategy=strategy,
        task=task,
        mode=mode,
        params=params,
        model_dir=model_dir,
        trainer=trainer)

  else:  # Compile/fit
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)

    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    if latest_checkpoint:
      logging.info('Starting to load checkpoint from GCS: %s', latest_checkpoint)
      start_time = time.time()
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      duration = time.time() - start_time
      logging.info(
          '⏰ Time to load checkpoint from GCS (Compile/Fit path): %.4f seconds',
          duration)
    else:
      logging.info('No checkpoint found to restore, initializing from scratch.')


    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=model_dir,
        max_to_keep=params.trainer.max_to_keep,
        step_counter=model.optimizer.iterations,
        checkpoint_interval=params.trainer.checkpoint_interval)

    checkpoint_callback = TimedStepCheckpoint(checkpoint_manager)

    time_callback = keras_utils.TimeHistory(
        params.task.train_data.global_batch_size,
        params.trainer.time_history.log_steps,
        logdir=model_dir if enable_tensorboard else None)
    callbacks = [checkpoint_callback, time_callback]

    if enable_tensorboard:
      tensorboard_callback = tf_keras.callbacks.TensorBoard(
          log_dir=model_dir,
          update_freq=min(1000, params.trainer.validation_interval),
          profile_batch=FLAGS.profile_steps)
      callbacks.append(tensorboard_callback)

    num_epochs = (params.trainer.train_steps //
                  params.trainer.validation_interval)
    current_step = model.optimizer.iterations.numpy()
    initial_epoch = current_step // params.trainer.validation_interval

    eval_steps = params.trainer.validation_steps if 'eval' in mode else None

    if mode in ['train', 'train_and_eval']:
      logging.info('Training started')
      history = model.fit(
          train_dataset,
          initial_epoch=initial_epoch,
          epochs=num_epochs,
          steps_per_epoch=params.trainer.validation_interval,
          validation_data=validation_dataset,
          validation_steps=eval_steps,
          callbacks=callbacks,
      )
      model.summary()
      logging.info('Train history: %s', history.history)
    elif mode == 'eval':
      logging.info('Evaluation started')
      validation_output = model.evaluate(validation_dataset, steps=eval_steps)
      logging.info('Evaluation output: %s', validation_output)
    else:
      raise NotImplementedError('The mode is not implemented: %s' % mode)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  common.define_flags()
  app.run(main)

