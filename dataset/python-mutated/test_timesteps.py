import numpy as np
import unittest
import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.utils.test_utils import check, framework_iterator

class TestTimeSteps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        ray.init()

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        ray.shutdown()

    def test_timesteps(self):
        if False:
            return 10
        'Test whether PG can be built with both frameworks.'
        config = ppo.PPOConfig().experimental(_disable_preprocessor_api=True).environment(RandomEnv).rollouts(num_rollout_workers=0).training(model={'fcnet_hiddens': [1], 'fcnet_activation': None})
        obs = np.array(1)
        obs_batch = np.array([1])
        for _ in framework_iterator(config):
            algo = config.build()
            policy = algo.get_policy()
            for i in range(1, 21):
                algo.compute_single_action(obs)
                check(int(policy.global_timestep), i)
            for i in range(1, 21):
                policy.compute_actions(obs_batch)
                check(int(policy.global_timestep), i + 20)
            crazy_timesteps = int(100000000000.0)
            policy.on_global_var_update({'timestep': crazy_timesteps})
            for i in range(1, 11):
                policy.compute_actions(obs_batch)
                check(int(policy.global_timestep), i + crazy_timesteps)
            algo.train()
            algo.stop()
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))