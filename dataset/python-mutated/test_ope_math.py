import unittest
import time
import gymnasium as gym
import torch
import numpy as np
from ray.rllib.offline.estimators import DirectMethod, DoublyRobust, ImportanceSampling, WeightedImportanceSampling
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.test_utils import check
import ray

class FakePolicy(TorchPolicyV2):
    """A fake policy used in test ope math to emulate a target policy that is better
    and worse than the random behavioral policy.

    In case of an improved policy, we assign higher probs to those actions that
    attained a higher reward and lower probs to those actions that attained a lower
    reward. We do the reverse in case of a worse policy.
    """

    def __init__(self, observation_space, action_space, sample_batch, improved=True):
        if False:
            for i in range(10):
                print('nop')
        self.sample_batch = sample_batch
        self.improved = improved
        self.config = {}
        self.model = ...
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = 'cpu'

    def action_distribution_fn(self, model, obs_batch=None, **kwargs):
        if False:
            i = 10
            return i + 15
        dist_class = TorchCategorical
        inds = obs_batch[SampleBatch.OBS][:, 0]
        old_rewards = self.sample_batch[SampleBatch.REWARDS][inds]
        old_actions = self.sample_batch[SampleBatch.ACTIONS][inds]
        dist_inputs = torch.ones((len(inds), self.action_space.n), dtype=torch.float32)
        delta = old_rewards - 1.5
        if not self.improved:
            delta = -delta
        dist_inputs[torch.arange(len(inds)), old_actions] = (dist_inputs[torch.arange(len(inds)), old_actions] + delta).float()
        return (dist_inputs, dist_class, None)

    def compute_log_likelihoods(self, actions, obs_batch, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        inds = obs_batch[:, 0]
        old_probs = self.sample_batch[SampleBatch.ACTION_PROB][inds]
        old_rewards = self.sample_batch[SampleBatch.REWARDS][inds]
        if self.improved:
            new_probs = (old_rewards == 2) * 1.5 * old_probs + (old_rewards == 1) * 0.5 * old_probs
        else:
            new_probs = (old_rewards == 2) * 0.5 * old_probs + (old_rewards == 1) * 1.5 * old_probs
        return np.log(new_probs)

class TestOPEMath(unittest.TestCase):
    """Tests some sanity checks that should pass based on the math of ope methods."""

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        ray.init()
        bsize = 1024
        action_dim = 2
        observation_space = gym.spaces.Box(-float('inf'), float('inf'), (1,))
        action_space = gym.spaces.Discrete(action_dim)
        cls.sample_batch = SampleBatch({SampleBatch.OBS: np.arange(bsize).reshape(-1, 1), SampleBatch.NEXT_OBS: np.arange(bsize).reshape(-1, 1) + 1, SampleBatch.ACTIONS: np.random.randint(0, action_dim, size=bsize), SampleBatch.REWARDS: np.random.randint(1, 3, size=bsize), SampleBatch.TERMINATEDS: np.ones(bsize), SampleBatch.TRUNCATEDS: np.zeros(bsize), SampleBatch.EPS_ID: np.arange(bsize), SampleBatch.ACTION_PROB: np.ones(bsize) / action_dim})
        cls.policies = {'good': FakePolicy(observation_space, action_space, cls.sample_batch, improved=True), 'bad': FakePolicy(observation_space, action_space, cls.sample_batch, improved=False)}

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        ray.shutdown()

    def test_is_and_wis_math(self):
        if False:
            print('Hello World!')
        'Tests that the importance sampling methods.\n\n        It checks whether is and wis methods outputs are consistent when\n        split_batch_by_episode is True or False (RL vs. Bandits)\n        '
        ope_classes = [ImportanceSampling, WeightedImportanceSampling]
        for class_module in ope_classes:
            for policy_tag in ['good', 'bad']:
                target_policy = self.policies[policy_tag]
                estimator = class_module(target_policy, gamma=0)
                s = time.time()
                estimate_1 = estimator.estimate(self.sample_batch, split_batch_by_episode=True)
                dt1 = time.time() - s
                s = time.time()
                estimate_2 = estimator.estimate(self.sample_batch, split_batch_by_episode=False)
                dt2 = time.time() - s
                if policy_tag == 'good':
                    self.assertGreater(estimate_1['v_gain'], 1)
                else:
                    self.assertLess(estimate_1['v_gain'], 1)
                check(estimate_1, estimate_2)
                self.assertGreater(dt1, dt2, f'in bandits split_by_episode = False should improve performance, dt_wo_split={dt2}, dt_with_split={dt1}')

    def test_dm_dr_math(self):
        if False:
            return 10
        'Tests the Direct Method and Doubly Robust methods.\n\n        It checks whether DM and DR methods outputs are consistent when\n        split_batch_by_episode is True or False (RL vs. Bandits)\n        '
        ope_classes = [DirectMethod, DoublyRobust]
        for class_module in ope_classes:
            target_policy = self.policies['good']
            estimator = class_module(target_policy, gamma=0)
            s = time.time()
            estimate_1 = estimator.estimate(self.sample_batch, split_batch_by_episode=True)
            dt1 = time.time() - s
            s = time.time()
            estimate_2 = estimator.estimate(self.sample_batch, split_batch_by_episode=False)
            dt2 = time.time() - s
            check(estimate_1, estimate_2)
            self.assertGreater(dt1, dt2, f'in bandits split_by_episode = False should improve performance, dt_wo_split={dt2}, dt_with_split={dt1}')
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))