"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
from typing import Optional
from ray.rllib.algorithms.dreamerv3.tf.models.components.mlp import MLP
from ray.rllib.algorithms.dreamerv3.utils import get_gru_units, get_num_z_classes, get_num_z_categoricals
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
(_, tf, _) = try_import_tf()
tfp = try_import_tfp()

class ContinuePredictor(tf.keras.Model):
    """The world-model network sub-component used to predict the `continue` flags .

    Predicted continue flags are used to produce "dream data" to learn the policy in.

    The continue flags are predicted via a linear output used to parameterize a
    Bernoulli distribution, from which simply the mode is used (no stochastic
    sampling!). In other words, if the sigmoid of the output of the linear layer is
    >0.5, we predict a continuation of the episode, otherwise we predict an episode
    terminal.
    """

    def __init__(self, *, model_size: Optional[str]='XS'):
        if False:
            while True:
                i = 10
        'Initializes a ContinuePredictor instance.\n\n        Args:\n            model_size: The "Model Size" used according to [1] Appendinx B.\n                Determines the exact size of the underlying MLP.\n        '
        super().__init__(name='continue_predictor')
        self.model_size = model_size
        self.mlp = MLP(model_size=model_size, output_layer_size=1)
        dl_type = tf.keras.mixed_precision.global_policy().compute_dtype or tf.float32
        self.call = tf.function(input_signature=[tf.TensorSpec(shape=[None, get_gru_units(model_size)], dtype=dl_type), tf.TensorSpec(shape=[None, get_num_z_categoricals(model_size), get_num_z_classes(model_size)], dtype=dl_type)])(self.call)

    def call(self, h, z):
        if False:
            print('Hello World!')
        'Performs a forward pass through the continue predictor.\n\n        Args:\n            h: The deterministic hidden state of the sequence model. [B, dim(h)].\n            z: The stochastic discrete representations of the original\n                observation input. [B, num_categoricals, num_classes].\n        '
        assert len(z.shape) == 3
        z_shape = tf.shape(z)
        z = tf.reshape(z, shape=(z_shape[0], -1))
        assert len(z.shape) == 2
        out = tf.concat([h, z], axis=-1)
        out.set_shape([None, get_num_z_categoricals(self.model_size) * get_num_z_classes(self.model_size) + get_gru_units(self.model_size)])
        out = self.mlp(out)
        logits = tf.cast(tf.squeeze(out, axis=-1), tf.float32)
        bernoulli = tfp.distributions.Bernoulli(logits=logits, dtype=tf.float32)
        continue_ = bernoulli.mode()
        return (continue_, bernoulli)