from gymnasium.spaces import Box
from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN
(tf1, tf, tfv) = try_import_tf()
(torch, nn) = try_import_torch()

class ParametricActionsModel(DistributionalQTFModel):
    """Parametric action model that handles the dot product and masking.

    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, true_obs_shape=(4,), action_embed_size=2, **kw):
        if False:
            print('Hello World!')
        super(ParametricActionsModel, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        self.action_embed_model = FullyConnectedNetwork(Box(-1, 1, shape=true_obs_shape), action_space, action_embed_size, model_config, name + '_action_embed')

    def forward(self, input_dict, state, seq_lens):
        if False:
            while True:
                i = 10
        avail_actions = input_dict['obs']['avail_actions']
        action_mask = input_dict['obs']['action_mask']
        (action_embed, _) = self.action_embed_model({'obs': input_dict['obs']['cart']})
        intent_vector = tf.expand_dims(action_embed, 1)
        action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=2)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return (action_logits + inf_mask, state)

    def value_function(self):
        if False:
            while True:
                i = 10
        return self.action_embed_model.value_function()

class TorchParametricActionsModel(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, true_obs_shape=(4,), action_embed_size=2, **kw):
        if False:
            print('Hello World!')
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kw)
        self.action_embed_model = TorchFC(Box(-1, 1, shape=true_obs_shape), action_space, action_embed_size, model_config, name + '_action_embed')

    def forward(self, input_dict, state, seq_lens):
        if False:
            return 10
        avail_actions = input_dict['obs']['avail_actions']
        action_mask = input_dict['obs']['action_mask']
        (action_embed, _) = self.action_embed_model({'obs': input_dict['obs']['cart']})
        intent_vector = torch.unsqueeze(action_embed, 1)
        action_logits = torch.sum(avail_actions * intent_vector, dim=2)
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        return (action_logits + inf_mask, state)

    def value_function(self):
        if False:
            i = 10
            return i + 15
        return self.action_embed_model.value_function()

class ParametricActionsModelThatLearnsEmbeddings(DistributionalQTFModel):
    """Same as the above ParametricActionsModel.

    However, this version also learns the action embeddings.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, true_obs_shape=(4,), action_embed_size=2, **kw):
        if False:
            print('Hello World!')
        super(ParametricActionsModelThatLearnsEmbeddings, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        action_ids_shifted = tf.constant(list(range(1, num_outputs + 1)), dtype=tf.float32)
        obs_cart = tf.keras.layers.Input(shape=true_obs_shape, name='obs_cart')
        valid_avail_actions_mask = tf.keras.layers.Input(shape=(num_outputs,), name='valid_avail_actions_mask')
        self.pred_action_embed_model = FullyConnectedNetwork(Box(-1, 1, shape=true_obs_shape), action_space, action_embed_size, model_config, name + '_pred_action_embed')
        (pred_action_embed, _) = self.pred_action_embed_model({'obs': obs_cart})
        _value_out = self.pred_action_embed_model.value_function()
        intent_vector = tf.expand_dims(pred_action_embed, 1)
        valid_avail_actions = action_ids_shifted * valid_avail_actions_mask
        valid_avail_actions_embed = tf.keras.layers.Embedding(input_dim=num_outputs + 1, output_dim=action_embed_size, name='action_embed_matrix')(valid_avail_actions)
        action_logits = tf.reduce_sum(valid_avail_actions_embed * intent_vector, axis=2)
        inf_mask = tf.maximum(tf.math.log(valid_avail_actions_mask), tf.float32.min)
        action_logits = action_logits + inf_mask
        self.param_actions_model = tf.keras.Model(inputs=[obs_cart, valid_avail_actions_mask], outputs=[action_logits, _value_out])
        self.param_actions_model.summary()

    def forward(self, input_dict, state, seq_lens):
        if False:
            print('Hello World!')
        valid_avail_actions_mask = input_dict['obs']['valid_avail_actions_mask']
        (action_logits, self._value_out) = self.param_actions_model([input_dict['obs']['cart'], valid_avail_actions_mask])
        return (action_logits, state)

    def value_function(self):
        if False:
            for i in range(10):
                print('nop')
        return self._value_out