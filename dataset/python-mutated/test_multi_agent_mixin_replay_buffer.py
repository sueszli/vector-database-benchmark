import numpy as np
import unittest
from ray.rllib.utils.replay_buffers.multi_agent_mixin_replay_buffer import MultiAgentMixInReplayBuffer
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, MultiAgentBatch

class TestMixInMultiAgentReplayBuffer(unittest.TestCase):
    batch_id = 0
    capacity = 10

    def _generate_episodes(self):
        if False:
            for i in range(10):
                print('nop')
        return SampleBatch({SampleBatch.T: [1, 0, 1], SampleBatch.ACTIONS: 3 * [np.random.choice([0, 1])], SampleBatch.REWARDS: 3 * [np.random.rand()], SampleBatch.OBS: 3 * [np.random.random((4,))], SampleBatch.NEXT_OBS: 3 * [np.random.random((4,))], SampleBatch.TERMINATEDS: [True, False, True], SampleBatch.TRUNCATEDS: [False, False, False], SampleBatch.SEQ_LENS: [1, 2], SampleBatch.EPS_ID: [-1, self.batch_id, self.batch_id], SampleBatch.AGENT_INDEX: 3 * [0]})

    def _generate_single_timesteps(self):
        if False:
            return 10
        return SampleBatch({SampleBatch.T: [0], SampleBatch.ACTIONS: [np.random.choice([0, 1])], SampleBatch.REWARDS: [np.random.rand()], SampleBatch.OBS: [np.random.random((4,))], SampleBatch.NEXT_OBS: [np.random.random((4,))], SampleBatch.TERMINATEDS: [True], SampleBatch.TRUNCATEDS: [False], SampleBatch.EPS_ID: [self.batch_id], SampleBatch.AGENT_INDEX: [0]})

    def test_mixin_sampling_episodes(self):
        if False:
            while True:
                i = 10
        'Test sampling of episodes.'
        buffer = MultiAgentMixInReplayBuffer(capacity=self.capacity, storage_unit='episodes', replay_ratio=0.5)
        results = []
        batch = self._generate_episodes()
        for _ in range(20):
            buffer.add(batch)
            sample = buffer.sample(2)
            assert type(sample) == MultiAgentBatch
            results.append(len(sample.policy_batches[DEFAULT_POLICY_ID]))
        self.assertAlmostEqual(np.mean(results), 2 * (len(batch) - 1))

    def test_mixin_sampling_sequences(self):
        if False:
            print('Hello World!')
        'Test sampling of sequences.'
        buffer = MultiAgentMixInReplayBuffer(capacity=100, storage_unit='sequences', replay_ratio=0.5, replay_sequence_length=2, replay_sequence_override=True)
        results = []
        batch = self._generate_episodes()
        for _ in range(400):
            buffer.add(batch)
            sample = buffer.sample(10)
            assert type(sample) == MultiAgentBatch
            results.append(len(sample.policy_batches[DEFAULT_POLICY_ID]))
        self.assertAlmostEqual(np.mean(results), 2 * len(batch), delta=0.1)

    def test_mixin_sampling_timesteps(self):
        if False:
            print('Hello World!')
        'Test different mixin ratios with timesteps.'
        buffer = MultiAgentMixInReplayBuffer(capacity=self.capacity, storage_unit='timesteps', replay_ratio=0.333)
        sample = buffer.sample(10)
        assert len(sample.policy_batches) == 0
        batch = self._generate_single_timesteps()
        results = []
        for _ in range(100):
            buffer.add(batch)
            buffer.add(batch)
            sample = buffer.sample(3)
            assert type(sample) == MultiAgentBatch
            results.append(len(sample.policy_batches[DEFAULT_POLICY_ID]))
        self.assertAlmostEqual(np.mean(results), 3.0, delta=0.2)
        results = []
        for _ in range(100):
            buffer.add(batch)
            sample = buffer.sample(5)
            assert type(sample) == MultiAgentBatch
            results.append(len(sample.policy_batches[DEFAULT_POLICY_ID]))
        self.assertAlmostEqual(np.mean(results), 1.5, delta=0.2)
        buffer = MultiAgentMixInReplayBuffer(capacity=self.capacity, replay_ratio=0.9)
        results = []
        for _ in range(100):
            buffer.add(batch)
            sample = buffer.sample(10)
            assert type(sample) == MultiAgentBatch
            results.append(len(sample.policy_batches[DEFAULT_POLICY_ID]))
        self.assertAlmostEqual(np.mean(results), 10.0, delta=0.2)
        buffer = MultiAgentMixInReplayBuffer(capacity=self.capacity, replay_ratio=0.0)
        batch = self._generate_single_timesteps()
        buffer.add(batch)
        sample = buffer.sample(1)
        assert type(sample) == MultiAgentBatch
        self.assertTrue(len(sample) == 1)
        sample = buffer.sample(1)
        assert type(sample) == MultiAgentBatch
        assert len(sample.policy_batches) == 0
        results = []
        for _ in range(100):
            buffer.add(batch)
            sample = buffer.sample(1)
            assert type(sample) == MultiAgentBatch
            results.append(len(sample.policy_batches[DEFAULT_POLICY_ID]))
        self.assertAlmostEqual(np.mean(results), 1.0, delta=0.2)
        buffer = MultiAgentMixInReplayBuffer(capacity=self.capacity, replay_ratio=1.0)
        sample = buffer.sample(1)
        assert len(sample.policy_batches) == 0
        batch = self._generate_single_timesteps()
        buffer.add(batch)
        sample = buffer.sample(1)
        assert type(sample) == MultiAgentBatch
        self.assertTrue(len(sample) == 1)
        sample = buffer.sample(1)
        assert type(sample) == MultiAgentBatch
        self.assertTrue(len(sample) == 1)
        results = []
        for _ in range(100):
            sample = buffer.sample(1)
            assert type(sample) == MultiAgentBatch
            results.append(len(sample.policy_batches[DEFAULT_POLICY_ID]))
        self.assertAlmostEqual(np.mean(results), 1.0)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))