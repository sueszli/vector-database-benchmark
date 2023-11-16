import gymnasium as gym
import numpy as np
import tree
import unittest
import ray
from ray.rllib.algorithms.appo import APPOConfig, APPOTF1Policy, APPOTF2Policy, APPOTorchPolicy
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check, framework_iterator
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
(tf1, tf, tfv) = try_import_tf()

class TestPolicyStateSwapping(unittest.TestCase):
    """Tests, whether Policies' states can be swapped out via their state on a GPU."""

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            i = 10
            return i + 15
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            i = 10
            return i + 15
        ray.shutdown()

    def test_policy_swap_gpu(self):
        if False:
            print('Hello World!')
        config = APPOConfig().resources(num_gpus=1).framework('tf2')
        obs_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        dummy_obs = obs_space.sample()
        act_space = gym.spaces.Discrete(100)
        num_policies = 2
        capacity = 1
        for fw in framework_iterator(config):
            cls = get_tf_eager_cls_if_necessary(APPOTF2Policy if fw == 'tf2' else APPOTF1Policy if fw == 'tf' else APPOTorchPolicy, config)
            policy_map = PolicyMap(capacity=capacity, policy_states_are_swappable=True)
            for i in range(num_policies):
                config.training(lr=(i + 1) * 0.01)
                with tf1.variable_scope(f'Policy{i}'):
                    policy = cls(observation_space=obs_space, action_space=act_space, config=config.to_dict())
                policy_map[f'pol{i}'] = policy
            dummy_batch_ones = tree.map_structure(lambda s: np.ones_like(s), policy_map['pol0']._dummy_batch)
            dummy_batch_twos = tree.map_structure(lambda s: np.full_like(s, 2.0), policy_map['pol0']._dummy_batch)
            logits = {pid: p.compute_single_action(dummy_obs)[2]['action_dist_inputs'] for (pid, p) in policy_map.items()}
            check(logits['pol0'], logits['pol1'], atol=1e-07, false=True)
            for i in range(50):
                pid = f'pol{i % num_policies}'
                print(i)
                pol = policy_map[pid]
                self.assertTrue(pol.config['lr'] == (i % num_policies + 1) * 0.01)
                self.assertTrue(policy_map._deque[-1] == pid)
                self.assertTrue(len(policy_map._deque) == capacity)
                self.assertTrue(len(policy_map.cache) == capacity)
                self.assertTrue(pid in policy_map.cache)
                check(pol.compute_single_action(dummy_obs)[2]['action_dist_inputs'], logits[pid])
            for i in range(num_policies):
                pid = f'pol{i % num_policies}'
                pol = policy_map[pid]
                if i == 0:
                    pol.learn_on_batch(dummy_batch_ones)
                else:
                    assert i == 1
                    pol.learn_on_batch(dummy_batch_twos)
                old_logits = logits[pid]
                logits[pid] = pol.compute_single_action(dummy_obs)[2]['action_dist_inputs']
                check(logits[pid], old_logits, atol=1e-07, false=True)
            check(logits['pol0'], logits['pol1'], atol=1e-07, false=True)
            for i in range(50):
                pid = f'pol{i % num_policies}'
                pol = policy_map[pid]
                check(pol.compute_single_action(dummy_obs)[2]['action_dist_inputs'], logits[pid])
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))