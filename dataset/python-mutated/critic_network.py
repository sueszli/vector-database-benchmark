"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
from typing import Optional
from ray.rllib.algorithms.dreamerv3.tf.models.components.mlp import MLP
from ray.rllib.algorithms.dreamerv3.tf.models.components.reward_predictor_layer import RewardPredictorLayer
from ray.rllib.algorithms.dreamerv3.utils import get_gru_units, get_num_z_categoricals, get_num_z_classes
from ray.rllib.utils.framework import try_import_tf
(_, tf, _) = try_import_tf()

class CriticNetwork(tf.keras.Model):
    """The critic network described in [1], predicting values for policy learning.

    Contains a copy of itself (EMA net) for weight regularization.
    The EMA net is updated after each train step via EMA (using the `ema_decay`
    parameter and the actual critic's weights). The EMA net is NOT used for target
    computations (we use the actual critic for that), its only purpose is to compute a
    weights regularizer term for the critic's loss such that the actual critic does not
    move too quickly.
    """

    def __init__(self, *, model_size: Optional[str]='XS', num_buckets: int=255, lower_bound: float=-20.0, upper_bound: float=20.0, ema_decay: float=0.98):
        if False:
            i = 10
            return i + 15
        'Initializes a CriticNetwork instance.\n\n        Args:\n            model_size: The "Model Size" used according to [1] Appendinx B.\n               Use None for manually setting the different network sizes.\n            num_buckets: The number of buckets to create. Note that the number of\n                possible symlog\'d outcomes from the used distribution is\n                `num_buckets` + 1:\n                lower_bound --bucket-- o[1] --bucket-- o[2] ... --bucket-- upper_bound\n                o=outcomes\n                lower_bound=o[0]\n                upper_bound=o[num_buckets]\n            lower_bound: The symlog\'d lower bound for a possible reward value.\n                Note that a value of -20.0 here already allows individual (actual env)\n                rewards to be as low as -400M. Buckets will be created between\n                `lower_bound` and `upper_bound`.\n            upper_bound: The symlog\'d upper bound for a possible reward value.\n                Note that a value of +20.0 here already allows individual (actual env)\n                rewards to be as high as 400M. Buckets will be created between\n                `lower_bound` and `upper_bound`.\n            ema_decay: The weight to use for updating the weights of the critic\'s copy\n                vs the actual critic. After each training update, the EMA copy of the\n                critic gets updated according to:\n                ema_net=(`ema_decay`*ema_net) + (1.0-`ema_decay`)*critic_net\n                The EMA copy of the critic is used inside the critic loss function only\n                to produce a regularizer term against the current critic\'s weights, NOT\n                to compute any target values.\n        '
        super().__init__(name='critic')
        self.model_size = model_size
        self.ema_decay = ema_decay
        self.mlp = MLP(model_size=self.model_size, output_layer_size=None)
        self.return_layer = RewardPredictorLayer(num_buckets=num_buckets, lower_bound=lower_bound, upper_bound=upper_bound)
        self.mlp_ema = MLP(model_size=self.model_size, output_layer_size=None, trainable=False)
        self.return_layer_ema = RewardPredictorLayer(num_buckets=num_buckets, lower_bound=lower_bound, upper_bound=upper_bound, trainable=False)
        dl_type = tf.keras.mixed_precision.global_policy().compute_dtype or tf.float32
        self.call = tf.function(input_signature=[tf.TensorSpec(shape=[None, get_gru_units(model_size)], dtype=dl_type), tf.TensorSpec(shape=[None, get_num_z_categoricals(model_size), get_num_z_classes(model_size)], dtype=dl_type), tf.TensorSpec(shape=[], dtype=tf.bool)])(self.call)

    def call(self, h, z, use_ema):
        if False:
            return 10
        'Performs a forward pass through the critic network.\n\n        Args:\n            h: The deterministic hidden state of the sequence model. [B, dim(h)].\n            z: The stochastic discrete representations of the original\n                observation input. [B, num_categoricals, num_classes].\n            use_ema: Whether to use the EMA-copy of the critic instead of the actual\n                critic to perform this computation.\n        '
        assert len(z.shape) == 3
        z_shape = tf.shape(z)
        z = tf.reshape(z, shape=(z_shape[0], -1))
        assert len(z.shape) == 2
        out = tf.concat([h, z], axis=-1)
        out.set_shape([None, get_num_z_categoricals(self.model_size) * get_num_z_classes(self.model_size) + get_gru_units(self.model_size)])
        if not use_ema:
            out = self.mlp(out)
            return self.return_layer(out)
        else:
            out = self.mlp_ema(out)
            return self.return_layer_ema(out)

    def init_ema(self) -> None:
        if False:
            return 10
        "Initializes the EMA-copy of the critic from the critic's weights.\n\n        After calling this method, the two networks have identical weights.\n        "
        vars = self.mlp.trainable_variables + self.return_layer.trainable_variables
        vars_ema = self.mlp_ema.variables + self.return_layer_ema.variables
        assert len(vars) == len(vars_ema) and len(vars) > 0
        for (var, var_ema) in zip(vars, vars_ema):
            assert var is not var_ema
            var_ema.assign(var)

    def update_ema(self) -> None:
        if False:
            i = 10
            return i + 15
        'Updates the EMA-copy of the critic according to the update formula:\n\n        ema_net=(`ema_decay`*ema_net) + (1.0-`ema_decay`)*critic_net\n        '
        vars = self.mlp.trainable_variables + self.return_layer.trainable_variables
        vars_ema = self.mlp_ema.variables + self.return_layer_ema.variables
        assert len(vars) == len(vars_ema) and len(vars) > 0
        for (var, var_ema) in zip(vars, vars_ema):
            var_ema.assign(self.ema_decay * var_ema + (1.0 - self.ema_decay) * var)