"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""
from typing import Optional
from ray.rllib.algorithms.dreamerv3.utils import get_num_z_categoricals, get_num_z_classes
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
(_, tf, _) = try_import_tf()
tfp = try_import_tfp()

class RepresentationLayer(tf.keras.layers.Layer):
    """A representation (z-state) generating layer.

    The value for z is the result of sampling from a categorical distribution with
    shape B x `num_classes`. So a computed z-state consists of `num_categoricals`
    one-hot vectors, each of size `num_classes_per_categorical`.
    """

    def __init__(self, *, model_size: Optional[str]='XS', num_categoricals: Optional[int]=None, num_classes_per_categorical: Optional[int]=None):
        if False:
            i = 10
            return i + 15
        'Initializes a RepresentationLayer instance.\n\n        Args:\n            model_size: The "Model Size" used according to [1] Appendinx B.\n                Use None for manually setting the different parameters.\n            num_categoricals: Overrides the number of categoricals used in the z-states.\n                In [1], 32 is used for any model size.\n            num_classes_per_categorical: Overrides the number of classes within each\n                categorical used for the z-states. In [1], 32 is used for any model\n                dimension.\n        '
        self.num_categoricals = get_num_z_categoricals(model_size, override=num_categoricals)
        self.num_classes_per_categorical = get_num_z_classes(model_size, override=num_classes_per_categorical)
        super().__init__(name=f'z{self.num_categoricals}x{self.num_classes_per_categorical}')
        self.z_generating_layer = tf.keras.layers.Dense(self.num_categoricals * self.num_classes_per_categorical, activation=None)

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        'Produces a discrete, differentiable z-sample from some 1D input tensor.\n\n        Pushes the input_ tensor through our dense layer, which outputs\n        32(B=num categoricals)*32(c=num classes) logits. Logits are used to:\n\n        1) sample stochastically\n        2) compute probs (via softmax)\n        3) make sure the sampling step is differentiable (see [2] Algorithm 1):\n            sample=one_hot(draw(logits))\n            probs=softmax(logits)\n            sample=sample + probs - stop_grad(probs)\n            -> Now sample has the gradients of the probs.\n\n        Args:\n            inputs: The input to our z-generating layer. This might be a) the combined\n                (concatenated) outputs of the (image?) encoder + the last hidden\n                deterministic state, or b) the output of the dynamics predictor MLP\n                network.\n\n        Returns:\n            Tuple consisting of a differentiable z-sample and the probabilities for the\n            categorical distribution (in the shape of [B, num_categoricals,\n            num_classes]) that created this sample.\n        '
        logits = self.z_generating_layer(inputs)
        logits = tf.reshape(logits, shape=(-1, self.num_categoricals, self.num_classes_per_categorical))
        probs = tf.nn.softmax(tf.cast(logits, tf.float32))
        probs = 0.99 * probs + 0.01 * (1.0 / self.num_classes_per_categorical)
        logits = tf.math.log(probs)
        distribution = tfp.distributions.Independent(tfp.distributions.OneHotCategorical(logits=logits), reinterpreted_batch_ndims=1)
        sample = tf.cast(distribution.sample(), tf.float32)
        differentiable_sample = tf.cast(tf.stop_gradient(sample) + probs - tf.stop_gradient(probs), tf.keras.mixed_precision.global_policy().compute_dtype or tf.float32)
        return (differentiable_sample, probs)