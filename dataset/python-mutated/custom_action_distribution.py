"""
This example shows two modifications:
1. How to write a custom action distribution
2. How to inject a custom action distribution into a Catalog
"""
import torch
import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.models.distributions import Distribution
from ray.rllib.models.torch.torch_distributions import TorchDeterministic

class CustomTorchCategorical(Distribution):

    def __init__(self, logits):
        if False:
            print('Hello World!')
        self.torch_dist = torch.distributions.categorical.Categorical(logits=logits)

    def sample(self, sample_shape=torch.Size(), **kwargs):
        if False:
            print('Hello World!')
        return self.torch_dist.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size(), **kwargs):
        if False:
            return 10
        return self._dist.rsample(sample_shape)

    def logp(self, value, **kwargs):
        if False:
            while True:
                i = 10
        return self.torch_dist.log_prob(value)

    def entropy(self):
        if False:
            for i in range(10):
                print('nop')
        return self.torch_dist.entropy()

    def kl(self, other, **kwargs):
        if False:
            while True:
                i = 10
        return torch.distributions.kl.kl_divergence(self.torch_dist, other.torch_dist)

    @staticmethod
    def required_input_dim(space, **kwargs):
        if False:
            while True:
                i = 10
        return int(space.n)

    @classmethod
    def from_logits(cls, logits):
        if False:
            while True:
                i = 10
        return CustomTorchCategorical(logits=logits)

    def to_deterministic(self):
        if False:
            print('Hello World!')
        return TorchDeterministic(loc=torch.argmax(self.logits, dim=-1))
env = gym.make('CartPole-v1')
dummy_logits = torch.randn([env.action_space.n])
dummy_dist = CustomTorchCategorical.from_logits(dummy_logits)
action = dummy_dist.sample()
env = gym.make('CartPole-v1')
env.reset()
env.step(action.numpy())

class CustomPPOCatalog(PPOCatalog):

    def get_action_dist_cls(self, framework):
        if False:
            while True:
                i = 10
        assert framework == 'torch'
        return CustomTorchCategorical
algo = PPOConfig().environment('CartPole-v1').rl_module(rl_module_spec=SingleAgentRLModuleSpec(catalog_class=CustomPPOCatalog)).build()
results = algo.train()
print(results)