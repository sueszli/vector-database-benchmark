import numpy as np
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.offline import JsonReader
(tf1, tf, tfv) = try_import_tf()
(torch, nn) = try_import_torch()

class CustomLossModel(TFModelV2):
    """Custom model that adds an imitation loss on top of the policy loss."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        if False:
            i = 10
            return i + 15
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.fcnet = FullyConnectedNetwork(self.obs_space, self.action_space, num_outputs, model_config, name='fcnet')

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if False:
            for i in range(10):
                print('nop')
        return self.fcnet(input_dict, state, seq_lens)

    @override(ModelV2)
    def value_function(self):
        if False:
            while True:
                i = 10
        return self.fcnet.value_function()

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        if False:
            for i in range(10):
                print('nop')
        reader = JsonReader(self.model_config['custom_model_config']['input_files'])
        input_ops = reader.tf_input_ops()
        obs = restore_original_dimensions(tf.cast(input_ops['obs'], tf.float32), self.obs_space)
        (logits, _) = self.forward({'obs': obs}, [], None)
        action_dist = Categorical(logits, self.model_config)
        self.policy_loss = policy_loss
        self.imitation_loss = tf.reduce_mean(-action_dist.logp(input_ops['actions']))
        return policy_loss + 10 * self.imitation_loss

    def metrics(self):
        if False:
            print('Hello World!')
        return {'policy_loss': self.policy_loss, 'imitation_loss': self.imitation_loss}

class TorchCustomLossModel(TorchModelV2, nn.Module):
    """PyTorch version of the CustomLossModel above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, input_files):
        if False:
            print('Hello World!')
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.input_files = input_files
        self.reader = JsonReader(self.input_files)
        self.fcnet = TorchFC(self.obs_space, self.action_space, num_outputs, model_config, name='fcnet')

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if False:
            for i in range(10):
                print('nop')
        return self.fcnet(input_dict, state, seq_lens)

    @override(ModelV2)
    def value_function(self):
        if False:
            i = 10
            return i + 15
        return self.fcnet.value_function()

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        if False:
            while True:
                i = 10
        'Calculates a custom loss on top of the given policy_loss(es).\n\n        Args:\n            policy_loss (List[TensorType]): The list of already calculated\n                policy losses (as many as there are optimizers).\n            loss_inputs: Struct of np.ndarrays holding the\n                entire train batch.\n\n        Returns:\n            List[TensorType]: The altered list of policy losses. In case the\n                custom loss should have its own optimizer, make sure the\n                returned list is one larger than the incoming policy_loss list.\n                In case you simply want to mix in the custom loss into the\n                already calculated policy losses, return a list of altered\n                policy losses (as done in this example below).\n        '
        batch = self.reader.next()
        obs = restore_original_dimensions(torch.from_numpy(batch['obs']).float().to(policy_loss[0].device), self.obs_space, tensorlib='torch')
        (logits, _) = self.forward({'obs': obs}, [], None)
        action_dist = TorchCategorical(logits, self.model_config)
        imitation_loss = torch.mean(-action_dist.logp(torch.from_numpy(batch['actions']).to(policy_loss[0].device)))
        self.imitation_loss_metric = imitation_loss.item()
        self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])
        return [loss_ + 10 * imitation_loss for loss_ in policy_loss]

    def metrics(self):
        if False:
            return 10
        return {'policy_loss': self.policy_loss_metric, 'imitation_loss': self.imitation_loss_metric}