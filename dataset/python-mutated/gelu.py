"""Gaussian error linear unit."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='Text')
def gelu(x):
    if False:
        while True:
            i = 10
    'Gaussian Error Linear Unit.\n\n  This is a smoother version of the RELU.\n  Original paper: https://arxiv.org/abs/1606.08415\n  Args:\n    x: float Tensor to perform activation.\n\n  Returns:\n    `x` with the GELU activation applied.\n  '
    cdf = 0.5 * (1.0 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))
    return x * cdf