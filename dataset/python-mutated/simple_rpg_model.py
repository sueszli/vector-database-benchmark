from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork as TFFCNet
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet
from ray.rllib.utils.framework import try_import_tf, try_import_torch
(tf1, tf, tfv) = try_import_tf()
(torch, nn) = try_import_torch()

class CustomTorchRPGModel(TorchModelV2, nn.Module):
    """Example of interpreting repeated observations."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        if False:
            i = 10
            return i + 15
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.model = TorchFCNet(obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        if False:
            while True:
                i = 10
        print('The unpacked input tensors:', input_dict['obs'])
        print()
        print('Unbatched repeat dim', input_dict['obs'].unbatch_repeat_dim())
        print()
        print('Fully unbatched', input_dict['obs'].unbatch_all())
        print()
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        if False:
            for i in range(10):
                print('nop')
        return self.model.value_function()

class CustomTFRPGModel(TFModelV2):
    """Example of interpreting repeated observations."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        if False:
            i = 10
            return i + 15
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.model = TFFCNet(obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        if False:
            print('Hello World!')
        print('The unpacked input tensors:', input_dict['obs'])
        print()
        print('Unbatched repeat dim', input_dict['obs'].unbatch_repeat_dim())
        print()
        if tf.executing_eagerly():
            print('Fully unbatched', input_dict['obs'].unbatch_all())
            print()
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        if False:
            for i in range(10):
                print('nop')
        return self.model.value_function()