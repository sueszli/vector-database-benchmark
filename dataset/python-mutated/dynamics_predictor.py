"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
from typing import Optional
from ray.rllib.algorithms.dreamerv3.tf.models.components.mlp import MLP
from ray.rllib.algorithms.dreamerv3.tf.models.components.representation_layer import RepresentationLayer
from ray.rllib.algorithms.dreamerv3.utils import get_gru_units
from ray.rllib.utils.framework import try_import_tf
(_, tf, _) = try_import_tf()

class DynamicsPredictor(tf.keras.Model):
    """The dynamics (or "prior") network described in [1], producing prior z-states.

    The dynamics net is used to:
    - compute the initial z-state (from the tanh'd initial h-state variable) at the
    beginning of a sequence.
    - compute prior-z-states during dream data generation. Note that during dreaming,
    no actual observations are available and thus no posterior z-states can be computed.
    """

    def __init__(self, *, model_size: Optional[str]='XS', num_categoricals: Optional[int]=None, num_classes_per_categorical: Optional[int]=None):
        if False:
            while True:
                i = 10
        'Initializes a DynamicsPredictor instance.\n\n        Args:\n            model_size: The "Model Size" used according to [1] Appendinx B.\n                Use None for manually setting the different parameters.\n            num_categoricals: Overrides the number of categoricals used in the z-states.\n                In [1], 32 is used for any model size.\n            num_classes_per_categorical: Overrides the number of classes within each\n                categorical used for the z-states. In [1], 32 is used for any model\n                dimension.\n        '
        super().__init__(name='dynamics_predictor')
        self.mlp = MLP(num_dense_layers=1, model_size=model_size, output_layer_size=None)
        self.representation_layer = RepresentationLayer(model_size=model_size, num_categoricals=num_categoricals, num_classes_per_categorical=num_classes_per_categorical)
        dl_type = tf.keras.mixed_precision.global_policy().compute_dtype or tf.float32
        self.call = tf.function(input_signature=[tf.TensorSpec(shape=[None, get_gru_units(model_size)], dtype=dl_type)])(self.call)

    def call(self, h):
        if False:
            while True:
                i = 10
        'Performs a forward pass through the dynamics (or "prior") network.\n\n        Args:\n            h: The deterministic hidden state of the sequence model.\n\n        Returns:\n            Tuple consisting of a differentiable z-sample and the probabilities for the\n            categorical distribution (in the shape of [B, num_categoricals,\n            num_classes]) that created this sample.\n        '
        out = self.mlp(h)
        return self.representation_layer(out)