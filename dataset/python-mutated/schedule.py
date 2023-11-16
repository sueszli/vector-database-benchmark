"""Abstract training on a step or epoch basis."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import tensorflow as tf
(_TRAIN, _EVAL) = (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL)
NUM_EXAMPLES = {tf.estimator.ModeKeys.TRAIN: 4572160, tf.estimator.ModeKeys.EVAL: 3000}

class Manager(object):
    """Container for convenience functions to abstract step or epoch basis.
  Transformer allows users to specify an epoch basis (generally recommended for
  full training) or a number of steps basis (convenient since epochs are rather
  large). TPUs furthermore require a step basis; however epochs are the norm in
  the machine learning community and it is desirable to allow users to specify
  epochs even when running with TPUS which requires behind the scenes
  conversions.
  This container simply groups what are largely mundane checks and conversions
  rather than interspersing them throughout the run loop code.
  """

    def __init__(self, train_steps, steps_between_evals, train_epochs, epochs_between_evals, default_train_epochs, batch_size, max_length, use_tpu=False, num_tpu_shards=8):
        if False:
            print('Hello World!')
        if train_steps and train_epochs:
            raise ValueError('Both train_steps or train_epochs were be defined.')
        if train_steps:
            self.train_eval_iterations = train_steps // steps_between_evals
            self._single_iteration_train_steps = steps_between_evals
            self._single_iteration_train_epochs = None
        else:
            train_epochs = train_epochs or default_train_epochs
            self.train_eval_iterations = train_epochs // epochs_between_evals
            self._single_iteration_train_steps = None
            self._single_iteration_train_epochs = epochs_between_evals
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_tpu = use_tpu
        self.num_tpu_shards = num_tpu_shards
        if self.use_tpu:
            assert self.batch_size // self.max_length % self.num_tpu_shards == 0

    @property
    def single_iteration_train_steps(self):
        if False:
            for i in range(10):
                print('nop')
        if self._single_iteration_train_steps or not self.use_tpu:
            return self._single_iteration_train_steps
        return self.epochs_to_steps(num_epochs=self._single_iteration_train_epochs, mode=_TRAIN)

    @property
    def single_iteration_eval_steps(self):
        if False:
            print('Hello World!')
        if not self.use_tpu:
            return None
        return self.epochs_to_steps(num_epochs=1, mode=_EVAL)

    @property
    def train_increment_str(self):
        if False:
            for i in range(10):
                print('nop')
        if self._single_iteration_train_steps:
            return '{} steps.'.format(self._single_iteration_train_steps)
        if not self.use_tpu:
            return '{} epochs.'.format(self._single_iteration_train_epochs)
        return '~{} epochs. ({} steps)'.format(self._single_iteration_train_epochs, self.single_iteration_train_steps)

    @property
    def repeat_dataset(self):
        if False:
            i = 10
            return i + 15
        if self._single_iteration_train_epochs is None and self._single_iteration_train_steps > NUM_EXAMPLES[_TRAIN]:
            return math.ceil(self._single_iteration_train_steps / NUM_EXAMPLES[_TRAIN])
        return self._single_iteration_train_epochs

    def epochs_to_steps(self, num_epochs, mode):
        if False:
            for i in range(10):
                print('nop')
        'Converts a number of epochs to a number of training steps.\n\n    TPU only: This function assumes that static_batch is True.\n\n      TPU can not tolerate an OutOfRange error from a dataset. As a result the\n    number of examples to be processed must be known ahead of time. TPUs also\n    do not allow partial batches, so this function rounds down.\n\n    Args:\n      num_epochs: An integer of the number of epochs to convert to steps.\n      mode: The estimator ModeKey of the computation\n\n    Returns:\n      An integer of the number of equivalent steps rounded down.\n    '
        assert self.use_tpu, 'epochs_to_steps should only be reached when using TPU'
        total_num_tokens = NUM_EXAMPLES[mode] * self.max_length * num_epochs
        return total_num_tokens // self.batch_size