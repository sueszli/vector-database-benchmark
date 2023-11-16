from gymnasium.spaces import Discrete, Tuple
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.misc import normc_initializer as normc_init_torch
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
(tf1, tf, tfv) = try_import_tf()
(torch, nn) = try_import_torch()

class AutoregressiveActionModel(TFModelV2):
    """Implements the `.action_model` branch required above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        if False:
            for i in range(10):
                print('nop')
        super(AutoregressiveActionModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        if action_space != Tuple([Discrete(2), Discrete(2)]):
            raise ValueError('This model only supports the [2, 2] action space')
        obs_input = tf.keras.layers.Input(shape=obs_space.shape, name='obs_input')
        a1_input = tf.keras.layers.Input(shape=(1,), name='a1_input')
        ctx_input = tf.keras.layers.Input(shape=(num_outputs,), name='ctx_input')
        context = tf.keras.layers.Dense(num_outputs, name='hidden', activation=tf.nn.tanh, kernel_initializer=normc_initializer(1.0))(obs_input)
        value_out = tf.keras.layers.Dense(1, name='value_out', activation=None, kernel_initializer=normc_initializer(0.01))(context)
        a1_logits = tf.keras.layers.Dense(2, name='a1_logits', activation=None, kernel_initializer=normc_initializer(0.01))(ctx_input)
        a2_context = a1_input
        a2_hidden = tf.keras.layers.Dense(16, name='a2_hidden', activation=tf.nn.tanh, kernel_initializer=normc_initializer(1.0))(a2_context)
        a2_logits = tf.keras.layers.Dense(2, name='a2_logits', activation=None, kernel_initializer=normc_initializer(0.01))(a2_hidden)
        self.base_model = tf.keras.Model(obs_input, [context, value_out])
        self.base_model.summary()
        self.action_model = tf.keras.Model([ctx_input, a1_input], [a1_logits, a2_logits])
        self.action_model.summary()

    def forward(self, input_dict, state, seq_lens):
        if False:
            i = 10
            return i + 15
        (context, self._value_out) = self.base_model(input_dict['obs'])
        return (context, state)

    def value_function(self):
        if False:
            print('Hello World!')
        return tf.reshape(self._value_out, [-1])

class TorchAutoregressiveActionModel(TorchModelV2, nn.Module):
    """PyTorch version of the AutoregressiveActionModel above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        if False:
            for i in range(10):
                print('nop')
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        if action_space != Tuple([Discrete(2), Discrete(2)]):
            raise ValueError('This model only supports the [2, 2] action space')
        self.context_layer = SlimFC(in_size=obs_space.shape[0], out_size=num_outputs, initializer=normc_init_torch(1.0), activation_fn=nn.Tanh)
        self.value_branch = SlimFC(in_size=num_outputs, out_size=1, initializer=normc_init_torch(0.01), activation_fn=None)
        self.a1_logits = SlimFC(in_size=num_outputs, out_size=2, activation_fn=None, initializer=normc_init_torch(0.01))

        class _ActionModel(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                nn.Module.__init__(self)
                self.a2_hidden = SlimFC(in_size=1, out_size=16, activation_fn=nn.Tanh, initializer=normc_init_torch(1.0))
                self.a2_logits = SlimFC(in_size=16, out_size=2, activation_fn=None, initializer=normc_init_torch(0.01))

            def forward(self_, ctx_input, a1_input):
                if False:
                    i = 10
                    return i + 15
                a1_logits = self.a1_logits(ctx_input)
                a2_logits = self_.a2_logits(self_.a2_hidden(a1_input))
                return (a1_logits, a2_logits)
        self.action_module = _ActionModel()
        self._context = None

    def forward(self, input_dict, state, seq_lens):
        if False:
            while True:
                i = 10
        self._context = self.context_layer(input_dict['obs'])
        return (self._context, state)

    def value_function(self):
        if False:
            for i in range(10):
                print('nop')
        return torch.reshape(self.value_branch(self._context), [-1])