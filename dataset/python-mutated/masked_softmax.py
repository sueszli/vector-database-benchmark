"""Keras-based softmax layer with optional masking."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='Text')
class MaskedSoftmax(tf.keras.layers.Layer):
    """Performs a softmax with optional masking on a tensor.

  Attributes:
    mask_expansion_axes: Any axes that should be padded on the mask tensor.
  """

    def __init__(self, mask_expansion_axes=None, **kwargs):
        if False:
            return 10
        self._mask_expansion_axes = mask_expansion_axes
        super(MaskedSoftmax, self).__init__(**kwargs)

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        if isinstance(inputs, list) and len(inputs) == 2:
            (scores, mask) = inputs
        else:
            (scores, mask) = (inputs, None)
        if mask is not None:
            if self._mask_expansion_axes is not None:
                mask = tf.expand_dims(mask, axis=self._mask_expansion_axes)
            adder = (1.0 - tf.cast(mask, scores.dtype)) * -10000.0
            scores += adder
        return tf.nn.softmax(scores)

    def get_config(self):
        if False:
            while True:
                i = 10
        config = {'mask_expansion_axes': self._mask_expansion_axes}
        base_config = super(MaskedSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))