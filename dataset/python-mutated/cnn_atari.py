"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
from typing import Optional
from ray.rllib.algorithms.dreamerv3.utils import get_cnn_multiplier
from ray.rllib.utils.framework import try_import_tf
(_, tf, _) = try_import_tf()

class CNNAtari(tf.keras.Model):
    """An image encoder mapping 64x64 RGB images via 4 CNN layers into a 1D space."""

    def __init__(self, *, model_size: Optional[str]='XS', cnn_multiplier: Optional[int]=None):
        if False:
            i = 10
            return i + 15
        'Initializes a CNNAtari instance.\n\n        Args:\n            model_size: The "Model Size" used according to [1] Appendinx B.\n                Use None for manually setting the `cnn_multiplier`.\n            cnn_multiplier: Optional override for the additional factor used to multiply\n                the number of filters with each CNN layer. Starting with\n                1 * `cnn_multiplier` filters in the first CNN layer, the number of\n                filters then increases via `2*cnn_multiplier`, `4*cnn_multiplier`, till\n                `8*cnn_multiplier`.\n        '
        super().__init__(name='image_encoder')
        cnn_multiplier = get_cnn_multiplier(model_size, override=cnn_multiplier)
        self.conv_layers = [tf.keras.layers.Conv2D(filters=1 * cnn_multiplier, kernel_size=4, strides=(2, 2), padding='same', activation=None, use_bias=False), tf.keras.layers.Conv2D(filters=2 * cnn_multiplier, kernel_size=4, strides=(2, 2), padding='same', activation=None, use_bias=False), tf.keras.layers.Conv2D(filters=4 * cnn_multiplier, kernel_size=4, strides=(2, 2), padding='same', activation=None, use_bias=False), tf.keras.layers.Conv2D(filters=8 * cnn_multiplier, kernel_size=4, strides=(2, 2), padding='same', activation=None, use_bias=False)]
        self.layer_normalizations = []
        for _ in range(len(self.conv_layers)):
            self.layer_normalizations.append(tf.keras.layers.LayerNormalization())
        self.flatten_layer = tf.keras.layers.Flatten(data_format='channels_last')

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 64, 64, 3], dtype=tf.keras.mixed_precision.global_policy().compute_dtype or tf.float32)])
    def call(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        'Performs a forward pass through the CNN Atari encoder.\n\n        Args:\n            inputs: The image inputs of shape (B, 64, 64, 3).\n        '
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, -1)
        out = inputs
        for (conv_2d, layer_norm) in zip(self.conv_layers, self.layer_normalizations):
            out = tf.nn.silu(layer_norm(inputs=conv_2d(out)))
        assert out.shape[1] == 4 and out.shape[2] == 4
        return self.flatten_layer(out)