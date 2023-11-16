import gymnasium as gym
import numpy as np
import unittest
import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.utils.test_utils import framework_iterator
from ray.tune.registry import register_env

class TestReproducibility(unittest.TestCase):

    def test_reproducing_trajectory(self):
        if False:
            while True:
                i = 10

        class PickLargest(gym.Env):

            def __init__(self):
                if False:
                    print('Hello World!')
                self.observation_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(4,))
                self.action_space = gym.spaces.Discrete(4)

            def reset(self, *, seed=None, options=None):
                if False:
                    i = 10
                    return i + 15
                self.obs = np.random.randn(4)
                return (self.obs, {})

            def step(self, action):
                if False:
                    i = 10
                    return i + 15
                reward = self.obs[action]
                return (self.obs, reward, True, False, {})

        def env_creator(env_config):
            if False:
                for i in range(10):
                    print('nop')
            return PickLargest()
        for fw in framework_iterator(frameworks=('tf', 'torch')):
            trajs = list()
            for trial in range(3):
                ray.init()
                register_env('PickLargest', env_creator)
                config = DQNConfig().environment('PickLargest').debugging(seed=666 if trial in [0, 1] else 999).reporting(min_time_s_per_iteration=0, min_sample_timesteps_per_iteration=100).framework(fw)
                algo = config.build()
                trajectory = list()
                for _ in range(8):
                    r = algo.train()
                    trajectory.append(r['episode_reward_max'])
                    trajectory.append(r['episode_reward_min'])
                trajs.append(trajectory)
                algo.stop()
                ray.shutdown()
            all_same = True
            for (v0, v1) in zip(trajs[0], trajs[1]):
                if v0 != v1:
                    all_same = False
            self.assertTrue(all_same)
            diff_cnt = 0
            for (v1, v2) in zip(trajs[1], trajs[2]):
                if v1 != v2:
                    diff_cnt += 1
            self.assertTrue(diff_cnt > 8)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))