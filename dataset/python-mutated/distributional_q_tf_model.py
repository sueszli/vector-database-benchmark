"""Tensorflow model for DQN"""
from typing import List
import gymnasium as gym
from ray.rllib.models.tf.layers import NoisyLayer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict, TensorType
(tf1, tf, tfv) = try_import_tf()

class DistributionalQTFModel(TFModelV2):
    """Extension of standard TFModel to provide distributional Q values.

    It also supports options for noisy nets and parameter space noise.

    Data flow:
        obs -> forward() -> model_out
        model_out -> get_q_value_distributions() -> Q(s, a) atoms
        model_out -> get_state_value() -> V(s)

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass."""

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str, q_hiddens=(256,), dueling: bool=False, num_atoms: int=1, use_noisy: bool=False, v_min: float=-10.0, v_max: float=10.0, sigma0: float=0.5, add_layer_norm: bool=False):
        if False:
            while True:
                i = 10
        'Initialize variables of this model.\n\n        Extra model kwargs:\n            q_hiddens (List[int]): List of layer-sizes after(!) the\n                Advantages(A)/Value(V)-split. Hence, each of the A- and V-\n                branches will have this structure of Dense layers. To define\n                the NN before this A/V-split, use - as always -\n                config["model"]["fcnet_hiddens"].\n            dueling: Whether to build the advantage(A)/value(V) heads\n                for DDQN. If True, Q-values are calculated as:\n                Q = (A - mean[A]) + V. If False, raw NN output is interpreted\n                as Q-values.\n            num_atoms: If >1, enables distributional DQN.\n            use_noisy: Use noisy nets.\n            v_min: Min value support for distributional DQN.\n            v_max: Max value support for distributional DQN.\n            sigma0 (float): Initial value of noisy layers.\n            add_layer_norm: Enable layer norm (for param noise).\n\n        Note that the core layers for forward() are not defined here, this\n        only defines the layers for the Q head. Those layers for forward()\n        should be defined in subclasses of DistributionalQModel.\n        '
        super(DistributionalQTFModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.model_out = tf.keras.layers.Input(shape=(num_outputs,), name='model_out')

        def build_action_value(prefix: str, model_out: TensorType) -> List[TensorType]:
            if False:
                i = 10
                return i + 15
            if q_hiddens:
                action_out = model_out
                for i in range(len(q_hiddens)):
                    if use_noisy:
                        action_out = NoisyLayer('{}hidden_{}'.format(prefix, i), q_hiddens[i], sigma0)(action_out)
                    elif add_layer_norm:
                        action_out = tf.keras.layers.Dense(units=q_hiddens[i], activation=tf.nn.relu)(action_out)
                        action_out = tf.keras.layers.LayerNormalization()(action_out)
                    else:
                        action_out = tf.keras.layers.Dense(units=q_hiddens[i], activation=tf.nn.relu, name='hidden_%d' % i)(action_out)
            else:
                action_out = model_out
            if use_noisy:
                action_scores = NoisyLayer('{}output'.format(prefix), self.action_space.n * num_atoms, sigma0, activation=None)(action_out)
            elif q_hiddens:
                action_scores = tf.keras.layers.Dense(units=self.action_space.n * num_atoms, activation=None)(action_out)
            else:
                action_scores = model_out
            if num_atoms > 1:
                z = tf.range(num_atoms, dtype=tf.float32)
                z = v_min + z * (v_max - v_min) / float(num_atoms - 1)

                def _layer(x):
                    if False:
                        return 10
                    support_logits_per_action = tf.reshape(tensor=x, shape=(-1, self.action_space.n, num_atoms))
                    support_prob_per_action = tf.nn.softmax(logits=support_logits_per_action)
                    x = tf.reduce_sum(input_tensor=z * support_prob_per_action, axis=-1)
                    logits = support_logits_per_action
                    dist = support_prob_per_action
                    return [x, z, support_logits_per_action, logits, dist]
                return tf.keras.layers.Lambda(_layer)(action_scores)
            else:
                logits = tf.expand_dims(tf.ones_like(action_scores), -1)
                dist = tf.expand_dims(tf.ones_like(action_scores), -1)
                return [action_scores, logits, dist]

        def build_state_score(prefix: str, model_out: TensorType) -> TensorType:
            if False:
                i = 10
                return i + 15
            state_out = model_out
            for i in range(len(q_hiddens)):
                if use_noisy:
                    state_out = NoisyLayer('{}dueling_hidden_{}'.format(prefix, i), q_hiddens[i], sigma0)(state_out)
                else:
                    state_out = tf.keras.layers.Dense(units=q_hiddens[i], activation=tf.nn.relu)(state_out)
                    if add_layer_norm:
                        state_out = tf.keras.layers.LayerNormalization()(state_out)
            if use_noisy:
                state_score = NoisyLayer('{}dueling_output'.format(prefix), num_atoms, sigma0, activation=None)(state_out)
            else:
                state_score = tf.keras.layers.Dense(units=num_atoms, activation=None)(state_out)
            return state_score
        q_out = build_action_value(name + '/action_value/', self.model_out)
        self.q_value_head = tf.keras.Model(self.model_out, q_out)
        if dueling:
            state_out = build_state_score(name + '/state_value/', self.model_out)
            self.state_value_head = tf.keras.Model(self.model_out, state_out)

    def get_q_value_distributions(self, model_out: TensorType) -> List[TensorType]:
        if False:
            while True:
                i = 10
        'Returns distributional values for Q(s, a) given a state embedding.\n\n        Override this in your custom model to customize the Q output head.\n\n        Args:\n            model_out: embedding from the model layers\n\n        Returns:\n            (action_scores, logits, dist) if num_atoms == 1, otherwise\n            (action_scores, z, support_logits_per_action, logits, dist)\n        '
        return self.q_value_head(model_out)

    def get_state_value(self, model_out: TensorType) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        'Returns the state value prediction for the given state embedding.'
        return self.state_value_head(model_out)