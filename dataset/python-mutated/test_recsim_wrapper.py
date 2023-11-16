import gymnasium as gym
import unittest
from ray.rllib.examples.env.recommender_system_envs_with_recsim import InterestEvolutionRecSimEnv
from ray.rllib.env.wrappers.recsim import MultiDiscreteToDiscreteActionWrapper
from ray.rllib.utils.error import UnsupportedSpaceException

class TestRecSimWrapper(unittest.TestCase):

    def test_observation_space(self):
        if False:
            for i in range(10):
                print('nop')
        env = InterestEvolutionRecSimEnv()
        (obs, info) = env.reset()
        self.assertTrue(env.observation_space.contains(obs), f"{env.observation_space} doesn't contain {obs}")
        (new_obs, _, _, _, _) = env.step(env.action_space.sample())
        self.assertTrue(env.observation_space.contains(new_obs))

    def test_action_space_conversion(self):
        if False:
            while True:
                i = 10
        env = InterestEvolutionRecSimEnv({'convert_to_discrete_action_space': True})
        self.assertIsInstance(env.action_space, gym.spaces.Discrete)
        env.reset()
        action = env.action_space.sample()
        self.assertTrue(env.action_space.contains(action))
        (new_obs, _, _, _, _) = env.step(action)
        self.assertTrue(env.observation_space.contains(new_obs))

    def test_bandits_observation_space_conversion(self):
        if False:
            return 10
        env = InterestEvolutionRecSimEnv({'wrap_for_bandits': True})
        self.assertIsInstance(env.observation_space['item'], gym.spaces.Box)

    def test_double_action_space_conversion_raises_exception(self):
        if False:
            return 10
        env = InterestEvolutionRecSimEnv({'convert_to_discrete_action_space': True})
        with self.assertRaises(UnsupportedSpaceException):
            env = MultiDiscreteToDiscreteActionWrapper(env)
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', __file__]))