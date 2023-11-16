import numpy as np
import gymnasium as gym
from typing import Dict
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import TensorType, List, ModelConfigDict
from ray.rllib.utils.annotations import DeveloperAPI
(tf1, tf, tfv) = try_import_tf()

@DeveloperAPI
class FullyConnectedNetwork(TFModelV2):
    """Generic fully connected network implemented in ModelV2 API."""

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        if False:
            for i in range(10):
                print('nop')
        super(FullyConnectedNetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        hiddens = list(model_config.get('fcnet_hiddens', [])) + list(model_config.get('post_fcnet_hiddens', []))
        activation = model_config.get('fcnet_activation')
        if not model_config.get('fcnet_hiddens', []):
            activation = model_config.get('post_fcnet_activation')
        activation = get_activation_fn(activation)
        no_final_linear = model_config.get('no_final_linear')
        vf_share_layers = model_config.get('vf_share_layers')
        free_log_std = model_config.get('free_log_std')
        if free_log_std:
            assert num_outputs % 2 == 0, ('num_outputs must be divisible by two', num_outputs)
            num_outputs = num_outputs // 2
            self.log_std_var = tf.Variable([0.0] * num_outputs, dtype=tf.float32, name='log_std')
        inputs = tf.keras.layers.Input(shape=(int(np.product(obs_space.shape)),), name='observations')
        last_layer = inputs
        logits_out = None
        i = 1
        for size in hiddens[:-1]:
            last_layer = tf.keras.layers.Dense(size, name='fc_{}'.format(i), activation=activation, kernel_initializer=normc_initializer(1.0))(last_layer)
            i += 1
        if no_final_linear and num_outputs:
            logits_out = tf.keras.layers.Dense(num_outputs, name='fc_out', activation=activation, kernel_initializer=normc_initializer(1.0))(last_layer)
        else:
            if len(hiddens) > 0:
                last_layer = tf.keras.layers.Dense(hiddens[-1], name='fc_{}'.format(i), activation=activation, kernel_initializer=normc_initializer(1.0))(last_layer)
            if num_outputs:
                logits_out = tf.keras.layers.Dense(num_outputs, name='fc_out', activation=None, kernel_initializer=normc_initializer(0.01))(last_layer)
            else:
                self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[-1]
        if free_log_std and logits_out is not None:

            def tiled_log_std(x):
                if False:
                    i = 10
                    return i + 15
                return tf.tile(tf.expand_dims(self.log_std_var, 0), [tf.shape(x)[0], 1])
            log_std_out = tf.keras.layers.Lambda(tiled_log_std)(inputs)
            logits_out = tf.keras.layers.Concatenate(axis=1)([logits_out, log_std_out])
        last_vf_layer = None
        if not vf_share_layers:
            last_vf_layer = inputs
            i = 1
            for size in hiddens:
                last_vf_layer = tf.keras.layers.Dense(size, name='fc_value_{}'.format(i), activation=activation, kernel_initializer=normc_initializer(1.0))(last_vf_layer)
                i += 1
        value_out = tf.keras.layers.Dense(1, name='value_out', activation=None, kernel_initializer=normc_initializer(0.01))(last_vf_layer if last_vf_layer is not None else last_layer)
        self.base_model = tf.keras.Model(inputs, [logits_out if logits_out is not None else last_layer, value_out])

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        if False:
            for i in range(10):
                print('nop')
        (model_out, self._value_out) = self.base_model(input_dict['obs_flat'])
        return (model_out, state)

    def value_function(self) -> TensorType:
        if False:
            i = 10
            return i + 15
        return tf.reshape(self._value_out, [-1])