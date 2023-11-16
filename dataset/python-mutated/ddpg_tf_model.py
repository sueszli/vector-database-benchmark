from typing import List, Optional
import gymnasium as gym
import numpy as np
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict, TensorType
(tf1, tf, tfv) = try_import_tf()

class DDPGTFModel(TFModelV2):
    """Extension of standard TFModel to provide DDPG action- and q-outputs.

    Data flow:
        obs -> forward() -> model_out
        model_out -> get_policy_output() -> deterministic actions
        model_out, actions -> get_q_values() -> Q(s, a)
        model_out, actions -> get_twin_q_values() -> Q_twin(s, a)

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass."""

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str, actor_hiddens: Optional[List[int]]=None, actor_hidden_activation: str='relu', critic_hiddens: Optional[List[int]]=None, critic_hidden_activation: str='relu', twin_q: bool=False, add_layer_norm: bool=False):
        if False:
            print('Hello World!')
        'Initialize variables of this model.\n\n        Extra model kwargs:\n            actor_hiddens: Defines size of hidden layers for the DDPG\n                policy head.\n                These will be used to postprocess the model output for the\n                purposes of computing deterministic actions.\n\n        Note that the core layers for forward() are not defined here, this\n        only defines the layers for the DDPG head. Those layers for forward()\n        should be defined in subclasses of DDPGActionModel.\n        '
        if actor_hiddens is None:
            actor_hiddens = [256, 256]
        if critic_hiddens is None:
            critic_hiddens = [256, 256]
        super(DDPGTFModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        actor_hidden_activation = getattr(tf.nn, actor_hidden_activation, tf.nn.relu)
        critic_hidden_activation = getattr(tf.nn, critic_hidden_activation, tf.nn.relu)
        self.model_out = tf.keras.layers.Input(shape=(num_outputs,), name='model_out')
        self.bounded = np.logical_and(action_space.bounded_above, action_space.bounded_below).any()
        self.action_dim = action_space.shape[0]
        if actor_hiddens:
            last_layer = self.model_out
            for (i, n) in enumerate(actor_hiddens):
                last_layer = tf.keras.layers.Dense(n, name='actor_hidden_{}'.format(i), activation=actor_hidden_activation)(last_layer)
                if add_layer_norm:
                    last_layer = tf.keras.layers.LayerNormalization(name='LayerNorm_{}'.format(i))(last_layer)
            actor_out = tf.keras.layers.Dense(self.action_dim, activation=None, name='actor_out')(last_layer)
        else:
            actor_out = self.model_out

        def lambda_(x):
            if False:
                print('Hello World!')
            action_range = (action_space.high - action_space.low)[None]
            low_action = action_space.low[None]
            sigmoid_out = tf.nn.sigmoid(2 * x)
            squashed = action_range * sigmoid_out + low_action
            return squashed
        if self.bounded:
            actor_out = tf.keras.layers.Lambda(lambda_)(actor_out)
        self.policy_model = tf.keras.Model(self.model_out, actor_out)
        self.actions_input = tf.keras.layers.Input(shape=(self.action_dim,), name='actions')

        def build_q_net(name, observations, actions):
            if False:
                return 10
            q_net = tf.keras.Sequential([tf.keras.layers.Concatenate(axis=1)] + [tf.keras.layers.Dense(units=units, activation=critic_hidden_activation, name='{}_hidden_{}'.format(name, i)) for (i, units) in enumerate(critic_hiddens)] + [tf.keras.layers.Dense(units=1, activation=None, name='{}_out'.format(name))])
            q_net = tf.keras.Model([observations, actions], q_net([observations, actions]))
            return q_net
        self.q_model = build_q_net('q', self.model_out, self.actions_input)
        if twin_q:
            self.twin_q_model = build_q_net('twin_q', self.model_out, self.actions_input)
        else:
            self.twin_q_model = None

    def get_q_values(self, model_out: TensorType, actions: TensorType) -> TensorType:
        if False:
            return 10
        'Return the Q estimates for the most recent forward pass.\n\n        This implements Q(s, a).\n\n        Args:\n            model_out: obs embeddings from the model layers, of shape\n                [BATCH_SIZE, num_outputs].\n            actions: Actions to return the Q-values for.\n                Shape: [BATCH_SIZE, action_dim].\n\n        Returns:\n            tensor of shape [BATCH_SIZE].\n        '
        if actions is not None:
            return self.q_model([model_out, actions])
        else:
            return self.q_model(model_out)

    def get_twin_q_values(self, model_out: TensorType, actions: TensorType) -> TensorType:
        if False:
            i = 10
            return i + 15
        'Same as get_q_values but using the twin Q net.\n\n        This implements the twin Q(s, a).\n\n        Args:\n            model_out: obs embeddings from the model layers, of shape\n                [BATCH_SIZE, num_outputs].\n            actions: Actions to return the Q-values for.\n                Shape: [BATCH_SIZE, action_dim].\n\n        Returns:\n            tensor of shape [BATCH_SIZE].\n        '
        if actions is not None:
            return self.twin_q_model([model_out, actions])
        else:
            return self.twin_q_model(model_out)

    def get_policy_output(self, model_out: TensorType) -> TensorType:
        if False:
            return 10
        'Return the action output for the most recent forward pass.\n\n        This outputs the support for pi(s). For continuous action spaces, this\n        is the action directly.\n\n        Args:\n            model_out: obs embeddings from the model layers, of shape\n                [BATCH_SIZE, num_outputs].\n\n        Returns:\n            tensor of shape [BATCH_SIZE, action_out_size]\n        '
        return self.policy_model(model_out)

    def policy_variables(self) -> List[TensorType]:
        if False:
            for i in range(10):
                print('nop')
        'Return the list of variables for the policy net.'
        return list(self.policy_model.variables)

    def q_variables(self) -> List[TensorType]:
        if False:
            for i in range(10):
                print('nop')
        'Return the list of variables for Q / twin Q nets.'
        return self.q_model.variables + (self.twin_q_model.variables if self.twin_q_model else [])