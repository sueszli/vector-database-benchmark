"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
from typing import Optional
import gymnasium as gym
from ray.rllib.algorithms.dreamerv3.tf.models.components.mlp import MLP
from ray.rllib.algorithms.dreamerv3.utils import get_gru_units, get_num_z_categoricals, get_num_z_classes
from ray.rllib.utils.framework import try_import_tf
(_, tf, _) = try_import_tf()

class VectorDecoder(tf.keras.Model):
    """A simple vector decoder to reproduce non-image (1D vector) observations.

    Wraps an MLP for mean parameter computations and a Gaussian distribution,
    from which we then sample using these mean values and a fixed stddev of 1.0.
    """

    def __init__(self, *, model_size: Optional[str]='XS', observation_space: gym.Space):
        if False:
            while True:
                i = 10
        'Initializes a VectorDecoder instance.\n\n        Args:\n            model_size: The "Model Size" used according to [1] Appendinx B.\n                Determines the exact size of the underlying MLP.\n            observation_space: The observation space to decode back into. This must\n                be a Box of shape (d,), where d >= 1.\n        '
        super().__init__(name='vector_decoder')
        self.model_size = model_size
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1
        self.mlp = MLP(model_size=model_size, output_layer_size=observation_space.shape[0])
        dl_type = tf.keras.mixed_precision.global_policy().compute_dtype or tf.float32
        self.call = tf.function(input_signature=[tf.TensorSpec(shape=[None, get_gru_units(model_size)], dtype=dl_type), tf.TensorSpec(shape=[None, get_num_z_categoricals(model_size), get_num_z_classes(model_size)], dtype=dl_type)])(self.call)

    def call(self, h, z):
        if False:
            for i in range(10):
                print('nop')
        'Performs a forward pass through the vector encoder.\n\n        Args:\n            h: The deterministic hidden state of the sequence model. [B, dim(h)].\n            z: The stochastic discrete representations of the original\n                observation input. [B, num_categoricals, num_classes].\n        '
        assert len(z.shape) == 3
        z_shape = tf.shape(z)
        z = tf.reshape(z, shape=(z_shape[0], -1))
        assert len(z.shape) == 2
        out = tf.concat([h, z], axis=-1)
        out.set_shape([None, get_num_z_categoricals(self.model_size) * get_num_z_classes(self.model_size) + get_gru_units(self.model_size)])
        loc = self.mlp(out)
        return loc