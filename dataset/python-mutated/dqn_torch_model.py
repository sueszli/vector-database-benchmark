"""PyTorch model for DQN"""
from typing import Sequence
import gymnasium as gym
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.modules.noisy_layer import NoisyLayer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict
(torch, nn) = try_import_torch()

class DQNTorchModel(TorchModelV2, nn.Module):
    """Extension of standard TorchModelV2 to provide dueling-Q functionality."""

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str, *, q_hiddens: Sequence[int]=(256,), dueling: bool=False, dueling_activation: str='relu', num_atoms: int=1, use_noisy: bool=False, v_min: float=-10.0, v_max: float=10.0, sigma0: float=0.5, add_layer_norm: bool=False):
        if False:
            i = 10
            return i + 15
        'Initialize variables of this model.\n\n        Extra model kwargs:\n            q_hiddens (Sequence[int]): List of layer-sizes after(!) the\n                Advantages(A)/Value(V)-split. Hence, each of the A- and V-\n                branches will have this structure of Dense layers. To define\n                the NN before this A/V-split, use - as always -\n                config["model"]["fcnet_hiddens"].\n            dueling: Whether to build the advantage(A)/value(V) heads\n                for DDQN. If True, Q-values are calculated as:\n                Q = (A - mean[A]) + V. If False, raw NN output is interpreted\n                as Q-values.\n            dueling_activation: The activation to use for all dueling\n                layers (A- and V-branch). One of "relu", "tanh", "linear".\n            num_atoms: If >1, enables distributional DQN.\n            use_noisy: Use noisy layers.\n            v_min: Min value support for distributional DQN.\n            v_max: Max value support for distributional DQN.\n            sigma0 (float): Initial value of noisy layers.\n            add_layer_norm: Enable layer norm (for param noise).\n        '
        nn.Module.__init__(self)
        super(DQNTorchModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.dueling = dueling
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.sigma0 = sigma0
        ins = num_outputs
        advantage_module = nn.Sequential()
        value_module = nn.Sequential()
        for (i, n) in enumerate(q_hiddens):
            if use_noisy:
                advantage_module.add_module('dueling_A_{}'.format(i), NoisyLayer(ins, n, sigma0=self.sigma0, activation=dueling_activation))
                value_module.add_module('dueling_V_{}'.format(i), NoisyLayer(ins, n, sigma0=self.sigma0, activation=dueling_activation))
            else:
                advantage_module.add_module('dueling_A_{}'.format(i), SlimFC(ins, n, activation_fn=dueling_activation))
                value_module.add_module('dueling_V_{}'.format(i), SlimFC(ins, n, activation_fn=dueling_activation))
                if add_layer_norm:
                    advantage_module.add_module('LayerNorm_A_{}'.format(i), nn.LayerNorm(n))
                    value_module.add_module('LayerNorm_V_{}'.format(i), nn.LayerNorm(n))
            ins = n
        if use_noisy:
            advantage_module.add_module('A', NoisyLayer(ins, self.action_space.n * self.num_atoms, sigma0, activation=None))
        elif q_hiddens:
            advantage_module.add_module('A', SlimFC(ins, action_space.n * self.num_atoms, activation_fn=None))
        self.advantage_module = advantage_module
        if self.dueling:
            if use_noisy:
                value_module.add_module('V', NoisyLayer(ins, self.num_atoms, sigma0, activation=None))
            elif q_hiddens:
                value_module.add_module('V', SlimFC(ins, self.num_atoms, activation_fn=None))
            self.value_module = value_module

    def get_q_value_distributions(self, model_out):
        if False:
            return 10
        'Returns distributional values for Q(s, a) given a state embedding.\n\n        Override this in your custom model to customize the Q output head.\n\n        Args:\n            model_out: Embedding from the model layers.\n\n        Returns:\n            (action_scores, logits, dist) if num_atoms == 1, otherwise\n            (action_scores, z, support_logits_per_action, logits, dist)\n        '
        action_scores = self.advantage_module(model_out)
        if self.num_atoms > 1:
            z = torch.arange(0.0, self.num_atoms, dtype=torch.float32).to(action_scores.device)
            z = self.v_min + z * (self.v_max - self.v_min) / float(self.num_atoms - 1)
            support_logits_per_action = torch.reshape(action_scores, shape=(-1, self.action_space.n, self.num_atoms))
            support_prob_per_action = nn.functional.softmax(support_logits_per_action, dim=-1)
            action_scores = torch.sum(z * support_prob_per_action, dim=-1)
            logits = support_logits_per_action
            probs = support_prob_per_action
            return (action_scores, z, support_logits_per_action, logits, probs)
        else:
            logits = torch.unsqueeze(torch.ones_like(action_scores), -1)
            return (action_scores, logits, logits)

    def get_state_value(self, model_out):
        if False:
            while True:
                i = 10
        'Returns the state value prediction for the given state embedding.'
        return self.value_module(model_out)