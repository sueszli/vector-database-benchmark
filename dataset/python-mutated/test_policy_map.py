import gymnasium as gym
import numpy as np
import time
import unittest
import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPOTF2Policy
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.utils.test_utils import check
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary

class TestPolicyMap(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            return 10
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            return 10
        ray.shutdown()

    def test_policy_map(self):
        if False:
            for i in range(10):
                print('nop')
        config = PPOConfig().framework('tf2')
        obs_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        dummy_obs = obs_space.sample()
        act_space = gym.spaces.Discrete(10000)
        num_policies = 6
        capacity = 2
        cls = get_tf_eager_cls_if_necessary(PPOTF2Policy, config)
        for use_swapping in [False, True]:
            policy_map = PolicyMap(capacity=capacity, policy_states_are_swappable=use_swapping)
            for i in range(num_policies):
                config.training(lr=(i + 1) * 1e-05)
                policy = cls(observation_space=obs_space, action_space=act_space, config=config.to_dict())
                policy_map[f'pol{i}'] = policy
                expected = [f'pol{j}' for j in range(max(i - 1, 0), i + 1)]
                self.assertEqual(list(policy_map._deque), expected)
                self.assertEqual(list(policy_map.cache.keys()), expected)
                self.assertEqual(policy_map._valid_keys, {f'pol{j}' for j in range(i + 1)})
            actions = {pid: p.compute_single_action(dummy_obs, explore=False)[0] for (pid, p) in policy_map.items()}
            start = time.time()
            for i in range(50):
                pid = f'pol{i % num_policies}'
                print(f'{i}) Testing `compute_single_action()` resulting in same outputs for stashed/recovered policy ({pid}) ...')
                pol = policy_map[pid]
                self.assertTrue(policy_map._deque[-1] == pid)
                self.assertTrue(len(policy_map._deque) == 2)
                self.assertTrue(len(policy_map.cache) == 2)
                self.assertTrue(pid in policy_map.cache)
                check(pol.compute_single_action(dummy_obs, explore=False)[0], actions[pid])
            time_total = time.time() - start
            print(f'Random access (swapping={use_swapping} took {time_total}sec.')
        policy_id = next(iter(policy_map._deque))
        del policy_map[policy_id]
        self.assertEqual(len(policy_map._deque), capacity - 1)
        self.assertTrue(policy_id not in policy_map._deque)
        self.assertEqual(len(policy_map.cache), capacity - 1)
        self.assertTrue(policy_id not in policy_map._deque)
        self.assertEqual(len(policy_map._valid_keys), num_policies - 1)
        self.assertTrue(policy_id not in policy_map._deque)
        config.training(lr=(i + 1) * 1e-05)
        policy = cls(observation_space=obs_space, action_space=act_space, config=config.to_dict())
        policy_id = f'pol{num_policies + 1}'
        policy_map[policy_id] = policy
        self.assertEqual(len(policy_map._deque), capacity)
        self.assertTrue(policy_id in policy_map._deque)
        self.assertEqual(len(policy_map.cache), capacity)
        self.assertTrue(policy_id in policy_map._deque)
        self.assertEqual(len(policy_map._valid_keys), num_policies)
        self.assertTrue(policy_id in policy_map._deque)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))