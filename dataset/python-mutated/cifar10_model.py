"""Model class for Cifar10 Dataset."""
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import model_base

class ResNetCifar10(model_base.ResNet):
    """Cifar10 model with ResNetV1 and basic residual block."""

    def __init__(self, num_layers, is_training, batch_norm_decay, batch_norm_epsilon, data_format='channels_first'):
        if False:
            i = 10
            return i + 15
        super(ResNetCifar10, self).__init__(is_training, data_format, batch_norm_decay, batch_norm_epsilon)
        self.n = (num_layers - 2) // 6
        self.num_classes = 10 + 1
        self.filters = [16, 16, 32, 64]
        self.strides = [1, 2, 2]

    def forward_pass(self, x, input_data_format='channels_last'):
        if False:
            for i in range(10):
                print('nop')
        'Build the core model within the graph.'
        if self._data_format != input_data_format:
            if input_data_format == 'channels_last':
                x = tf.transpose(x, [0, 3, 1, 2])
            else:
                x = tf.transpose(x, [0, 2, 3, 1])
        x = x / 128 - 1
        x = self._conv(x, 3, 16, 1)
        x = self._batch_norm(x)
        x = self._relu(x)
        res_func = self._residual_v1
        for i in range(3):
            with tf.name_scope('stage'):
                for j in range(self.n):
                    if j == 0:
                        x = res_func(x, 3, self.filters[i], self.filters[i + 1], self.strides[i])
                    else:
                        x = res_func(x, 3, self.filters[i + 1], self.filters[i + 1], 1)
        x = self._global_avg_pool(x)
        x = self._fully_connected(x, self.num_classes)
        return x