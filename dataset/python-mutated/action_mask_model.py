from gymnasium.spaces import Dict
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
(tf1, tf, tfv) = try_import_tf()
(torch, nn) = try_import_torch()

class ActionMaskModel(TFModelV2):
    """Model that handles simple discrete action masking.

    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        if False:
            print('Hello World!')
        orig_space = getattr(obs_space, 'original_space', obs_space)
        assert isinstance(orig_space, Dict) and 'action_mask' in orig_space.spaces and ('observations' in orig_space.spaces)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.internal_model = FullyConnectedNetwork(orig_space['observations'], action_space, num_outputs, model_config, name + '_internal')
        self.no_masking = model_config['custom_model_config'].get('no_masking', False)

    def forward(self, input_dict, state, seq_lens):
        if False:
            return 10
        action_mask = input_dict['obs']['action_mask']
        (logits, _) = self.internal_model({'obs': input_dict['obs']['observations']})
        if self.no_masking:
            return (logits, state)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask
        return (masked_logits, state)

    def value_function(self):
        if False:
            return 10
        return self.internal_model.value_function()

class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        orig_space = getattr(obs_space, 'original_space', obs_space)
        assert isinstance(orig_space, Dict) and 'action_mask' in orig_space.spaces and ('observations' in orig_space.spaces)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)
        self.internal_model = TorchFC(orig_space['observations'], action_space, num_outputs, model_config, name + '_internal')
        self.no_masking = False
        if 'no_masking' in model_config['custom_model_config']:
            self.no_masking = model_config['custom_model_config']['no_masking']

    def forward(self, input_dict, state, seq_lens):
        if False:
            return 10
        action_mask = input_dict['obs']['action_mask']
        (logits, _) = self.internal_model({'obs': input_dict['obs']['observations']})
        if self.no_masking:
            return (logits, state)
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask
        return (masked_logits, state)

    def value_function(self):
        if False:
            print('Hello World!')
        return self.internal_model.value_function()