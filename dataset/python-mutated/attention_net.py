"""
[1] - Attention Is All You Need - Vaswani, Jones, Shazeer, Parmar,
      Uszkoreit, Gomez, Kaiser - Google Brain/Research, U Toronto - 2017.
      https://arxiv.org/pdf/1706.03762.pdf
[2] - Stabilizing Transformers for Reinforcement Learning - E. Parisotto
      et al. - DeepMind - 2019. https://arxiv.org/pdf/1910.06764.pdf
[3] - Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context.
      Z. Dai, Z. Yang, et al. - Carnegie Mellon U - 2019.
      https://www.aclweb.org/anthology/P19-1285.pdf
"""
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np
import tree
from typing import Any, Dict, Optional, Union
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.layers import GRUGate, RelativeMultiHeadAttention, SkipConnection
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.tf_utils import flatten_inputs_to_1d_tensor, one_hot
from ray.rllib.utils.typing import ModelConfigDict, TensorType, List
from ray.rllib.utils.deprecation import deprecation_warning
from ray.util import log_once
(tf1, tf, tfv) = try_import_tf()

class PositionwiseFeedforward(tf.keras.layers.Layer if tf else object):
    """A 2x linear layer with ReLU activation in between described in [1].

    Each timestep coming from the attention head will be passed through this
    layer separately.
    """

    def __init__(self, out_dim: int, hidden_dim: int, output_activation: Optional[Any]=None, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self._hidden_layer = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu)
        self._output_layer = tf.keras.layers.Dense(out_dim, activation=output_activation)
        if log_once('positionwise_feedforward_tf'):
            deprecation_warning(old='rllib.models.tf.attention_net.PositionwiseFeedforward')

    def call(self, inputs: TensorType, **kwargs) -> TensorType:
        if False:
            while True:
                i = 10
        del kwargs
        output = self._hidden_layer(inputs)
        return self._output_layer(output)

class TrXLNet(RecurrentNetwork):
    """A TrXL net Model described in [1]."""

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str, num_transformer_units: int, attention_dim: int, num_heads: int, head_dim: int, position_wise_mlp_dim: int):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a TrXLNet object.\n\n        Args:\n            num_transformer_units: The number of Transformer repeats to\n                use (denoted L in [2]).\n            attention_dim: The input and output dimensions of one\n                Transformer unit.\n            num_heads: The number of attention heads to use in parallel.\n                Denoted as `H` in [3].\n            head_dim: The dimension of a single(!) attention head within\n                a multi-head attention unit. Denoted as `d` in [3].\n            position_wise_mlp_dim: The dimension of the hidden layer\n                within the position-wise MLP (after the multi-head attention\n                block within one Transformer unit). This is the size of the\n                first of the two layers within the PositionwiseFeedforward. The\n                second layer always has size=`attention_dim`.\n        '
        if log_once('trxl_net_tf'):
            deprecation_warning(old='rllib.models.tf.attention_net.TrXLNet')
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        self.num_transformer_units = num_transformer_units
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = model_config['max_seq_len']
        self.obs_dim = observation_space.shape[0]
        inputs = tf.keras.layers.Input(shape=(self.max_seq_len, self.obs_dim), name='inputs')
        E_out = tf.keras.layers.Dense(attention_dim)(inputs)
        for _ in range(self.num_transformer_units):
            MHA_out = SkipConnection(RelativeMultiHeadAttention(out_dim=attention_dim, num_heads=num_heads, head_dim=head_dim, input_layernorm=False, output_activation=None), fan_in_layer=None)(E_out)
            E_out = SkipConnection(PositionwiseFeedforward(attention_dim, position_wise_mlp_dim))(MHA_out)
            E_out = tf.keras.layers.LayerNormalization(axis=-1)(E_out)
        logits = tf.keras.layers.Dense(self.num_outputs, activation=tf.keras.activations.linear, name='logits')(E_out)
        self.base_model = tf.keras.models.Model([inputs], [logits])

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        if False:
            return 10
        observations = state[0]
        observations = tf.concat((observations, inputs), axis=1)[:, -self.max_seq_len:]
        logits = self.base_model([observations])
        T = tf.shape(inputs)[1]
        logits = logits[:, -T:]
        return (logits, [observations])

    @override(RecurrentNetwork)
    def get_initial_state(self) -> List[np.ndarray]:
        if False:
            i = 10
            return i + 15
        return [np.zeros((self.max_seq_len, self.obs_dim), np.float32)]

class GTrXLNet(RecurrentNetwork):
    """A GTrXL net Model described in [2].

    This is still in an experimental phase.
    Can be used as a drop-in replacement for LSTMs in PPO and IMPALA.
    For an example script, see: `ray/rllib/examples/attention_net.py`.

    To use this network as a replacement for an RNN, configure your Algorithm
    as follows:

    Examples:
        >> config["model"]["custom_model"] = GTrXLNet
        >> config["model"]["max_seq_len"] = 10
        >> config["model"]["custom_model_config"] = {
        >>     num_transformer_units=1,
        >>     attention_dim=32,
        >>     num_heads=2,
        >>     memory_inference=100,
        >>     memory_training=50,
        >>     etc..
        >> }
    """

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: Optional[int], model_config: ModelConfigDict, name: str, *, num_transformer_units: int=1, attention_dim: int=64, num_heads: int=2, memory_inference: int=50, memory_training: int=50, head_dim: int=32, position_wise_mlp_dim: int=32, init_gru_gate_bias: float=2.0):
        if False:
            i = 10
            return i + 15
        'Initializes a GTrXLNet instance.\n\n        Args:\n            num_transformer_units: The number of Transformer repeats to\n                use (denoted L in [2]).\n            attention_dim: The input and output dimensions of one\n                Transformer unit.\n            num_heads: The number of attention heads to use in parallel.\n                Denoted as `H` in [3].\n            memory_inference: The number of timesteps to concat (time\n                axis) and feed into the next transformer unit as inference\n                input. The first transformer unit will receive this number of\n                past observations (plus the current one), instead.\n            memory_training: The number of timesteps to concat (time\n                axis) and feed into the next transformer unit as training\n                input (plus the actual input sequence of len=max_seq_len).\n                The first transformer unit will receive this number of\n                past observations (plus the input sequence), instead.\n            head_dim: The dimension of a single(!) attention head within\n                a multi-head attention unit. Denoted as `d` in [3].\n            position_wise_mlp_dim: The dimension of the hidden layer\n                within the position-wise MLP (after the multi-head attention\n                block within one Transformer unit). This is the size of the\n                first of the two layers within the PositionwiseFeedforward. The\n                second layer always has size=`attention_dim`.\n            init_gru_gate_bias: Initial bias values for the GRU gates\n                (two GRUs per Transformer unit, one after the MHA, one after\n                the position-wise MLP).\n        '
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        self.num_transformer_units = num_transformer_units
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.memory_inference = memory_inference
        self.memory_training = memory_training
        self.head_dim = head_dim
        self.max_seq_len = model_config['max_seq_len']
        self.obs_dim = observation_space.shape[0]
        input_layer = tf.keras.layers.Input(shape=(None, self.obs_dim), name='inputs')
        memory_ins = [tf.keras.layers.Input(shape=(None, self.attention_dim), dtype=tf.float32, name='memory_in_{}'.format(i)) for i in range(self.num_transformer_units)]
        E_out = tf.keras.layers.Dense(self.attention_dim)(input_layer)
        memory_outs = [E_out]
        for i in range(self.num_transformer_units):
            MHA_out = SkipConnection(RelativeMultiHeadAttention(out_dim=self.attention_dim, num_heads=num_heads, head_dim=head_dim, input_layernorm=True, output_activation=tf.nn.relu), fan_in_layer=GRUGate(init_gru_gate_bias), name='mha_{}'.format(i + 1))(E_out, memory=memory_ins[i])
            E_out = SkipConnection(tf.keras.Sequential((tf.keras.layers.LayerNormalization(axis=-1), PositionwiseFeedforward(out_dim=self.attention_dim, hidden_dim=position_wise_mlp_dim, output_activation=tf.nn.relu))), fan_in_layer=GRUGate(init_gru_gate_bias), name='pos_wise_mlp_{}'.format(i + 1))(MHA_out)
            memory_outs.append(E_out)
        self._logits = None
        self._value_out = None
        if num_outputs is not None:
            self._logits = tf.keras.layers.Dense(self.num_outputs, activation=None, name='logits')(E_out)
            values_out = tf.keras.layers.Dense(1, activation=None, name='values')(E_out)
            outs = [self._logits, values_out]
        else:
            outs = [E_out]
            self.num_outputs = self.attention_dim
        self.trxl_model = tf.keras.Model(inputs=[input_layer] + memory_ins, outputs=outs + memory_outs[:-1])
        self.trxl_model.summary()
        for i in range(self.num_transformer_units):
            space = Box(-1.0, 1.0, shape=(self.attention_dim,))
            self.view_requirements['state_in_{}'.format(i)] = ViewRequirement('state_out_{}'.format(i), shift='-{}:-1'.format(self.memory_inference), batch_repeat_value=self.max_seq_len, space=space)
            self.view_requirements['state_out_{}'.format(i)] = ViewRequirement(space=space, used_for_training=False)

    @override(ModelV2)
    def forward(self, input_dict, state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        if False:
            return 10
        assert seq_lens is not None
        B = tf.shape(seq_lens)[0]
        observations = input_dict[SampleBatch.OBS]
        shape = tf.shape(observations)
        T = shape[0] // B
        observations = tf.reshape(observations, tf.concat([[-1, T], shape[1:]], axis=0))
        all_out = self.trxl_model([observations] + state)
        if self._logits is not None:
            out = tf.reshape(all_out[0], [-1, self.num_outputs])
            self._value_out = all_out[1]
            memory_outs = all_out[2:]
        else:
            out = tf.reshape(all_out[0], [-1, self.attention_dim])
            memory_outs = all_out[1:]
        return (out, [tf.reshape(m, [-1, self.attention_dim]) for m in memory_outs])

    @override(RecurrentNetwork)
    def get_initial_state(self) -> List[np.ndarray]:
        if False:
            print('Hello World!')
        return [tf.zeros(self.view_requirements['state_in_{}'.format(i)].space.shape) for i in range(self.num_transformer_units)]

    @override(ModelV2)
    def value_function(self) -> TensorType:
        if False:
            i = 10
            return i + 15
        return tf.reshape(self._value_out, [-1])

class AttentionWrapper(TFModelV2):
    """GTrXL wrapper serving as interface for ModelV2s that set use_attention."""

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        if False:
            while True:
                i = 10
        if log_once('attention_wrapper_tf_deprecation'):
            deprecation_warning(old='ray.rllib.models.tf.attention_net.AttentionWrapper')
        super().__init__(obs_space, action_space, None, model_config, name)
        self.use_n_prev_actions = model_config['attention_use_n_prev_actions']
        self.use_n_prev_rewards = model_config['attention_use_n_prev_rewards']
        self.action_space_struct = get_base_struct_from_space(self.action_space)
        self.action_dim = 0
        for space in tree.flatten(self.action_space_struct):
            if isinstance(space, Discrete):
                self.action_dim += space.n
            elif isinstance(space, MultiDiscrete):
                self.action_dim += np.sum(space.nvec)
            elif space.shape is not None:
                self.action_dim += int(np.product(space.shape))
            else:
                self.action_dim += int(len(space))
        if self.use_n_prev_actions:
            self.num_outputs += self.use_n_prev_actions * self.action_dim
        if self.use_n_prev_rewards:
            self.num_outputs += self.use_n_prev_rewards
        cfg = model_config
        self.attention_dim = cfg['attention_dim']
        if self.num_outputs is not None:
            in_space = gym.spaces.Box(float('-inf'), float('inf'), shape=(self.num_outputs,), dtype=np.float32)
        else:
            in_space = obs_space
        self.gtrxl = GTrXLNet(in_space, action_space, None, model_config, 'gtrxl', num_transformer_units=cfg['attention_num_transformer_units'], attention_dim=self.attention_dim, num_heads=cfg['attention_num_heads'], head_dim=cfg['attention_head_dim'], memory_inference=cfg['attention_memory_inference'], memory_training=cfg['attention_memory_training'], position_wise_mlp_dim=cfg['attention_position_wise_mlp_dim'], init_gru_gate_bias=cfg['attention_init_gru_gate_bias'])
        input_ = tf.keras.layers.Input(shape=(self.gtrxl.num_outputs,))
        self.num_outputs = num_outputs
        out = tf.keras.layers.Dense(self.num_outputs, activation=None)(input_)
        self._logits_branch = tf.keras.models.Model([input_], [out])
        out = tf.keras.layers.Dense(1, activation=None)(input_)
        self._value_branch = tf.keras.models.Model([input_], [out])
        self.view_requirements = self.gtrxl.view_requirements
        self.view_requirements['obs'].space = self.obs_space
        if self.use_n_prev_actions:
            self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(SampleBatch.ACTIONS, space=self.action_space, shift='-{}:-1'.format(self.use_n_prev_actions))
        if self.use_n_prev_rewards:
            self.view_requirements[SampleBatch.PREV_REWARDS] = ViewRequirement(SampleBatch.REWARDS, shift='-{}:-1'.format(self.use_n_prev_rewards))

    @override(RecurrentNetwork)
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        if False:
            print('Hello World!')
        assert seq_lens is not None
        (wrapped_out, _) = self._wrapped_forward(input_dict, [], None)
        prev_a_r = []
        if self.use_n_prev_actions:
            prev_n_actions = input_dict[SampleBatch.PREV_ACTIONS]
            if self.model_config['_disable_action_flattening']:
                flat = flatten_inputs_to_1d_tensor(prev_n_actions, spaces_struct=self.action_space_struct, time_axis=True)
                flat = tf.reshape(flat, [tf.shape(flat)[0], -1])
                prev_a_r.append(flat)
            elif isinstance(self.action_space, Discrete):
                for i in range(self.use_n_prev_actions):
                    prev_a_r.append(one_hot(prev_n_actions[:, i], self.action_space))
            elif isinstance(self.action_space, MultiDiscrete):
                for i in range(0, self.use_n_prev_actions, self.action_space.shape[0]):
                    prev_a_r.append(one_hot(tf.cast(prev_n_actions[:, i:i + self.action_space.shape[0]], tf.float32), space=self.action_space))
            else:
                prev_a_r.append(tf.reshape(tf.cast(prev_n_actions, tf.float32), [-1, self.use_n_prev_actions * self.action_dim]))
        if self.use_n_prev_rewards:
            prev_a_r.append(tf.reshape(tf.cast(input_dict[SampleBatch.PREV_REWARDS], tf.float32), [-1, self.use_n_prev_rewards]))
        if prev_a_r:
            wrapped_out = tf.concat([wrapped_out] + prev_a_r, axis=1)
        input_dict['obs_flat'] = input_dict['obs'] = wrapped_out
        (self._features, memory_outs) = self.gtrxl(input_dict, state, seq_lens)
        model_out = self._logits_branch(self._features)
        return (model_out, memory_outs)

    @override(ModelV2)
    def value_function(self) -> TensorType:
        if False:
            return 10
        assert self._features is not None, 'Must call forward() first!'
        return tf.reshape(self._value_branch(self._features), [-1])

    @override(ModelV2)
    def get_initial_state(self) -> Union[List[np.ndarray], List[TensorType]]:
        if False:
            print('Hello World!')
        return [np.zeros(self.gtrxl.view_requirements['state_in_{}'.format(i)].space.shape) for i in range(self.gtrxl.num_transformer_units)]