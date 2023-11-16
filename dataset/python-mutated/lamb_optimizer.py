"""Functions and classes related to optimization (weight updates)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import six
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops

class LAMBOptimizer(tf.train.Optimizer):
    """LAMB (Layer-wise Adaptive Moments optimizer for Batch training)."""

    def __init__(self, learning_rate, weight_decay_rate=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-06, exclude_from_weight_decay=None, exclude_from_layer_adaptation=None, name='LAMBOptimizer'):
        if False:
            print('Hello World!')
        'Constructs a LAMBOptimizer.'
        super(LAMBOptimizer, self).__init__(False, name)
        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        if False:
            return 10
        'See base class.'
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue
            param_name = self._get_variable_name(param.name)
            m = tf.get_variable(name=six.ensure_str(param_name) + '/adam_m', shape=param.shape.as_list(), dtype=tf.float32, trainable=False, initializer=tf.zeros_initializer())
            v = tf.get_variable(name=six.ensure_str(param_name) + '/adam_v', shape=param.shape.as_list(), dtype=tf.float32, trainable=False, initializer=tf.zeros_initializer())
            next_m = tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad)
            next_v = tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2, tf.square(grad))
            update = next_m / (tf.sqrt(next_v) + self.epsilon)
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param
            ratio = 1.0
            if self._do_layer_adaptation(param_name):
                w_norm = linalg_ops.norm(param, ord=2)
                g_norm = linalg_ops.norm(update, ord=2)
                ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(math_ops.greater(g_norm, 0), w_norm / g_norm, 1.0), 1.0)
            update_with_lr = ratio * self.learning_rate * update
            next_param = param - update_with_lr
            assignments.extend([param.assign(next_param), m.assign(next_m), v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        if False:
            print('Hello World!')
        'Whether to use L2 weight decay for `param_name`.'
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        if False:
            i = 10
            return i + 15
        'Whether to do layer-wise learning rate adaptation for `param_name`.'
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        if False:
            while True:
                i = 10
        'Get the variable name from the tensor name.'
        m = re.match('^(.*):\\d+$', six.ensure_str(param_name))
        if m is not None:
            param_name = m.group(1)
        return param_name