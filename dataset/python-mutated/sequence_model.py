"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
from typing import Optional
import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.dreamerv3.tf.models.components.mlp import MLP
from ray.rllib.algorithms.dreamerv3.utils import get_gru_units, get_num_z_classes, get_num_z_categoricals
from ray.rllib.utils.framework import try_import_tf
(_, tf, _) = try_import_tf()

class SequenceModel(tf.keras.Model):
    """The "sequence model" of the RSSM, computing ht+1 given (ht, zt, at).

    Note: The "internal state" always consists of:
    The actions `a` (initially, this is a zeroed-out action), `h`-states (deterministic,
    continuous), and `z`-states (stochastic, discrete).
    There are two versions of z-states: "posterior" for world model training and "prior"
    for creating the dream data.

    Initial internal state values (`a`, `h`, and `z`) are used where ever a new episode
    starts within a batch row OR at the beginning of each train batch's B rows,
    regardless of whether there was an actual episode boundary or not. Thus, internal
    states are not required to be stored in or retrieved from the replay buffer AND
    retrieved batches from the buffer must not be zero padded.

    Initial `a` is the zero "one hot" action, e.g. [0.0, 0.0] for Discrete(2), initial
    `h` is a separate learned variable, and initial `z` are computed by the "dynamics"
    (or "prior") net, using only the initial-h state as input.

    The GRU in this SequenceModel always produces the next h-state, then.
    """

    def __init__(self, *, model_size: Optional[str]='XS', action_space: gym.Space, num_gru_units: Optional[int]=None):
        if False:
            while True:
                i = 10
        'Initializes a SequenceModel instance.\n\n        Args:\n            model_size: The "Model Size" used according to [1] Appendinx B.\n                Use None for manually setting the number of GRU units used.\n            action_space: The action space the our environment used.\n            num_gru_units: Overrides the number of GRU units (dimension of the h-state).\n                If None, use the value given through `model_size`\n                (see [1] Appendix B).\n        '
        super().__init__(name='sequence_model')
        self.model_size = model_size
        self.action_space = action_space
        num_gru_units = get_gru_units(self.model_size, override=num_gru_units)
        self.pre_gru_layer = MLP(num_dense_layers=1, model_size=self.model_size, output_layer_size=None)
        self.gru_unit = tf.keras.layers.GRU(num_gru_units, return_sequences=False, return_state=False, time_major=True)
        dl_type = tf.keras.mixed_precision.global_policy().compute_dtype or tf.float32
        self.call = tf.function(input_signature=[tf.TensorSpec(shape=[None] + ([action_space.n] if isinstance(action_space, gym.spaces.Discrete) else list(action_space.shape)), dtype=dl_type), tf.TensorSpec(shape=[None, num_gru_units], dtype=dl_type), tf.TensorSpec(shape=[None, get_num_z_categoricals(self.model_size), get_num_z_classes(self.model_size)], dtype=dl_type)])(self.call)

    def call(self, a, h, z):
        if False:
            for i in range(10):
                print('nop')
        "\n\n        Args:\n            a: The previous action (already one-hot'd if applicable). (B, ...).\n            h: The previous deterministic hidden state of the sequence model.\n                (B, num_gru_units)\n            z: The previous stochastic discrete representations of the original\n                observation input. (B, num_categoricals, num_classes_per_categorical).\n        "
        z_shape = tf.shape(z)
        z = tf.reshape(z, shape=(z_shape[0], -1))
        out = tf.concat([z, a], axis=-1)
        out.set_shape([None, get_num_z_categoricals(self.model_size) * get_num_z_classes(self.model_size) + (self.action_space.n if isinstance(self.action_space, gym.spaces.Discrete) else int(np.prod(self.action_space.shape)))])
        out = self.pre_gru_layer(out)
        h_next = self.gru_unit(tf.expand_dims(out, axis=0), initial_state=h)
        return h_next