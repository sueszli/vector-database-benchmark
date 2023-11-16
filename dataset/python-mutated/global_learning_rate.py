"""A trainable optimizer that learns a single global learning rate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from learned_optimizer.optimizer import trainable_optimizer

class GlobalLearningRate(trainable_optimizer.TrainableOptimizer):
    """Optimizes for a single global learning rate."""

    def __init__(self, initial_rate=0.001, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the global learning rate.'
        with tf.variable_scope(trainable_optimizer.OPTIMIZER_SCOPE):
            initializer = tf.constant_initializer(initial_rate)
            self.learning_rate = tf.get_variable('global_learning_rate', shape=(), initializer=initializer)
        super(GlobalLearningRate, self).__init__('GLR', [], **kwargs)

    def _compute_update(self, param, grad, state):
        if False:
            while True:
                i = 10
        return (param - tf.scalar_mul(self.learning_rate, grad), state)