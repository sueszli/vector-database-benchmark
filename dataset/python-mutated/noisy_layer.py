import numpy as np
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import get_variable, try_import_tf, TensorType, TensorShape
from ray.rllib.utils.deprecation import deprecation_warning
from ray.util import log_once
(tf1, tf, tfv) = try_import_tf()

class NoisyLayer(tf.keras.layers.Layer if tf else object):
    """A Layer that adds learnable Noise to some previous layer's outputs.

    Consists of:
    - a common dense layer: y = w^{T}x + b
    - a noisy layer: y = (w + \\epsilon_w*\\sigma_w)^{T}x +
        (b+\\epsilon_b*\\sigma_b)
    , where \\epsilon are random variables sampled from factorized normal
    distributions and \\sigma are trainable variables which are expected to
    vanish along the training procedure.
    """

    def __init__(self, prefix: str, out_size: int, sigma0: float, activation: str='relu'):
        if False:
            return 10
        'Initializes a NoisyLayer object.\n\n        Args:\n            prefix:\n            out_size: Output size for Noisy Layer\n            sigma0: Initialization value for sigma_b (bias noise)\n            non_linear: Non-linear activation for Noisy Layer\n        '
        super().__init__()
        self.prefix = prefix
        self.out_size = out_size
        self.sigma0 = sigma0
        self.activation = activation
        self.w = None
        self.b = None
        self.sigma_w = None
        self.sigma_b = None
        if log_once('noisy_layer'):
            deprecation_warning(old='rllib.models.tf.layers.NoisyLayer')

    def build(self, input_shape: TensorShape):
        if False:
            for i in range(10):
                print('nop')
        in_size = int(input_shape[1])
        self.sigma_w = get_variable(value=tf.keras.initializers.RandomUniform(minval=-1.0 / np.sqrt(float(in_size)), maxval=1.0 / np.sqrt(float(in_size))), trainable=True, tf_name=self.prefix + '_sigma_w', shape=[in_size, self.out_size], dtype=tf.float32)
        self.sigma_b = get_variable(value=tf.keras.initializers.Constant(self.sigma0 / np.sqrt(float(in_size))), trainable=True, tf_name=self.prefix + '_sigma_b', shape=[self.out_size], dtype=tf.float32)
        self.w = get_variable(value=tf.keras.initializers.GlorotUniform(), tf_name=self.prefix + '_fc_w', trainable=True, shape=[in_size, self.out_size], dtype=tf.float32)
        self.b = get_variable(value=tf.keras.initializers.Zeros(), tf_name=self.prefix + '_fc_b', trainable=True, shape=[self.out_size], dtype=tf.float32)

    def call(self, inputs: TensorType) -> TensorType:
        if False:
            i = 10
            return i + 15
        in_size = int(inputs.shape[1])
        epsilon_in = tf.random.normal(shape=[in_size])
        epsilon_out = tf.random.normal(shape=[self.out_size])
        epsilon_in = self._f_epsilon(epsilon_in)
        epsilon_out = self._f_epsilon(epsilon_out)
        epsilon_w = tf.matmul(a=tf.expand_dims(epsilon_in, -1), b=tf.expand_dims(epsilon_out, 0))
        epsilon_b = epsilon_out
        action_activation = tf.matmul(inputs, self.w + self.sigma_w * epsilon_w) + self.b + self.sigma_b * epsilon_b
        fn = get_activation_fn(self.activation, framework='tf')
        if fn is not None:
            action_activation = fn(action_activation)
        return action_activation

    def _f_epsilon(self, x: TensorType) -> TensorType:
        if False:
            while True:
                i = 10
        return tf.math.sign(x) * tf.math.sqrt(tf.math.abs(x))