"""
This example shows two modifications:
- How to write a custom Encoder (using MobileNet v2)
- How to enhance Catalogs with this custom Encoder

With the pattern shown in this example, we can enhance Catalogs such that they extend
to new observation- or action spaces while retaining their original functionality.
"""
import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.examples.models.mobilenet_v2_encoder import MobileNetV2EncoderConfig, MOBILENET_INPUT_SHAPE
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.examples.env.random_env import RandomEnv

class MobileNetEnhancedPPOCatalog(PPOCatalog):

    @classmethod
    def _get_encoder_config(cls, observation_space: gym.Space, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(observation_space, gym.spaces.Box) and observation_space.shape == MOBILENET_INPUT_SHAPE:
            return MobileNetV2EncoderConfig()
        else:
            return super()._get_encoder_config(observation_space, **kwargs)
ppo_config = PPOConfig().experimental(_enable_new_api_stack=True).rl_module(rl_module_spec=SingleAgentRLModuleSpec(catalog_class=MobileNetEnhancedPPOCatalog)).rollouts(num_rollout_workers=0).training(train_batch_size=32, sgd_minibatch_size=16, num_sgd_iter=1)
ppo_config.environment('CartPole-v1')
results = ppo_config.build().train()
print(results)
ppo_config.environment(RandomEnv, env_config={'action_space': gym.spaces.Discrete(2), 'observation_space': gym.spaces.Box(0.0, 1.0, shape=MOBILENET_INPUT_SHAPE, dtype=np.float32)})
results = ppo_config.build().train()
print(results)