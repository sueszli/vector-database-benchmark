"""A trainable optimizer that learns a learning rate schedule."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from learned_optimizer.optimizer import trainable_optimizer

class LearningRateSchedule(trainable_optimizer.TrainableOptimizer):
    """Learns a learning rate schedule over a fixed number of iterations."""

    def __init__(self, initial_rate=0.0, n_steps=1000, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the learning rates.'
        self.max_index = tf.constant(n_steps - 1, dtype=tf.int32)
        with tf.variable_scope(trainable_optimizer.OPTIMIZER_SCOPE):
            initializer = tf.constant_initializer(initial_rate)
            self.learning_rates = tf.get_variable('learning_rates', shape=[n_steps], initializer=initializer)
        super(LearningRateSchedule, self).__init__('LRS', ['itr'], **kwargs)

    def _initialize_state(self, var):
        if False:
            for i in range(10):
                print('nop')
        'Return a dictionary mapping names of state variables to their values.'
        return {'itr': tf.constant(0, dtype=tf.int32)}

    def _compute_update(self, param, grad, state):
        if False:
            print('Hello World!')
        'Compute updates of parameters.'
        index = tf.minimum(state['itr'], self.max_index)
        learning_rate = tf.gather(self.learning_rates, index)
        updated_param = param - tf.scalar_mul(learning_rate, grad)
        return (updated_param, {'itr': state['itr'] + 1})