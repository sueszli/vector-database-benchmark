"""
This file holds several examples for the Catalogs API that are used in the catalog
guide.
"""
import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
env = gym.make('CartPole-v1')
catalog = PPOCatalog(env.observation_space, env.action_space, model_config_dict={})
encoder = catalog.build_actor_critic_encoder(framework='torch')
policy_head = catalog.build_pi_head(framework='torch')
action_dist_class = catalog.get_action_dist_cls(framework='torch')
import gymnasium as gym
import torch
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.policy.sample_batch import SampleBatch
env = gym.make('CartPole-v1')
catalog = Catalog(env.observation_space, env.action_space, model_config_dict={})
action_dist_class = catalog.get_action_dist_cls(framework='torch')
encoder = catalog.build_encoder(framework='torch')
head = torch.nn.Linear(catalog.latent_dims[0], env.action_space.n)
(obs, info) = env.reset()
input_batch = {SampleBatch.OBS: torch.Tensor([obs])}
encoding = encoder(input_batch)[ENCODER_OUT]
action_dist_inputs = head(encoding)
action_dist = action_dist_class.from_logits(action_dist_inputs)
actions = action_dist.sample().numpy()
env.step(actions[0])
import gymnasium as gym
import torch
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.base import STATE_IN, ENCODER_OUT, ACTOR
from ray.rllib.policy.sample_batch import SampleBatch
env = gym.make('CartPole-v1')
catalog = PPOCatalog(env.observation_space, env.action_space, model_config_dict={})
encoder = catalog.build_actor_critic_encoder(framework='torch')
policy_head = catalog.build_pi_head(framework='torch')
action_dist_class = catalog.get_action_dist_cls(framework='torch')
(obs, info) = env.reset()
input_batch = {SampleBatch.OBS: torch.Tensor([obs])}
encoding = encoder(input_batch)[ENCODER_OUT][ACTOR]
action_dist_inputs = policy_head(encoding)
action_dist = action_dist_class.from_logits(action_dist_inputs)
actions = action_dist.sample().numpy()
env.step(actions[0])
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec

class MyPPOCatalog(PPOCatalog):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        print('Hi from within PPORLModule!')
        super().__init__(*args, **kwargs)
config = PPOConfig().experimental(_enable_new_api_stack=True).environment('CartPole-v1').framework('torch')
config = config.rl_module(rl_module_spec=SingleAgentRLModuleSpec(catalog_class=MyPPOCatalog))
ppo = config.build()