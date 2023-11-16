from collections import Counter
import numpy as np
import unittest
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, concat_samples
from ray.rllib.utils.test_utils import check

class TestPrioritizedReplayBuffer(unittest.TestCase):
    """
    Tests insertion and (weighted) sampling of the PrioritizedReplayBuffer.
    """
    capacity = 10
    alpha = 1.0
    beta = 1.0

    def _generate_data(self):
        if False:
            for i in range(10):
                print('nop')
        return SampleBatch({SampleBatch.T: [np.random.random((4,))], SampleBatch.ACTIONS: [np.random.choice([0, 1])], SampleBatch.REWARDS: [np.random.rand()], SampleBatch.OBS: [np.random.random((4,))], SampleBatch.NEXT_OBS: [np.random.random((4,))], SampleBatch.TERMINATEDS: [np.random.choice([False, True])], SampleBatch.TRUNCATEDS: [np.random.choice([False, False])]})

    def test_multi_agent_batches(self):
        if False:
            while True:
                i = 10
        'Tests buffer with storage of MultiAgentBatches.'
        self.batch_id = 0

        def _add_multi_agent_batch_to_buffer(buffer, num_policies, num_batches=5, seq_lens=False, **kwargs):
            if False:
                print('Hello World!')

            def _generate_data(policy_id):
                if False:
                    for i in range(10):
                        print('nop')
                batch = SampleBatch({SampleBatch.T: [0, 1], SampleBatch.ACTIONS: 2 * [np.random.choice([0, 1])], SampleBatch.REWARDS: 2 * [np.random.rand()], SampleBatch.OBS: 2 * [np.random.random((4,))], SampleBatch.NEXT_OBS: 2 * [np.random.random((4,))], SampleBatch.TERMINATEDS: [False, False], SampleBatch.TRUNCATEDS: [False, True], SampleBatch.EPS_ID: 2 * [self.batch_id], SampleBatch.AGENT_INDEX: 2 * [0], SampleBatch.SEQ_LENS: [2], 'batch_id': 2 * [self.batch_id], 'policy_id': 2 * [policy_id]})
                if not seq_lens:
                    del batch[SampleBatch.SEQ_LENS]
                self.batch_id += 1
                return batch
            for i in range(num_batches):
                policy_batches = {idx: _generate_data(idx) for (idx, _) in enumerate(range(num_policies))}
                batch = MultiAgentBatch(policy_batches, num_batches * 2)
                buffer.add(batch, **kwargs)
        buffer = PrioritizedReplayBuffer(capacity=100, storage_unit='fragments', alpha=0.5)
        _add_multi_agent_batch_to_buffer(buffer, num_policies=2, num_batches=2)
        assert len(buffer) == 2
        assert buffer._num_timesteps_added == 8
        assert buffer._num_timesteps_added_wrap == 8
        assert buffer._next_idx == 2
        assert buffer._eviction_started is False
        buffer.sample(3, beta=0.5)
        assert buffer._num_timesteps_sampled == 12
        _add_multi_agent_batch_to_buffer(buffer, batch_size=100, num_policies=3, num_batches=3)
        assert len(buffer) == 5
        assert buffer._num_timesteps_added == 26
        assert buffer._num_timesteps_added_wrap == 26
        assert buffer._next_idx == 5

    def test_sequence_size(self):
        if False:
            i = 10
            return i + 15
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.1, storage_unit='fragments')
        for _ in range(200):
            buffer.add(self._generate_data())
        assert len(buffer._storage) == 100, len(buffer._storage)
        assert buffer.stats()['added_count'] == 200, buffer.stats()
        state = buffer.get_state()
        new_memory = PrioritizedReplayBuffer(capacity=100, alpha=0.1)
        new_memory.set_state(state)
        assert len(new_memory._storage) == 100, len(new_memory._storage)
        assert new_memory.stats()['added_count'] == 200, new_memory.stats()
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.1, storage_unit='fragments')
        for _ in range(40):
            buffer.add(concat_samples([self._generate_data() for _ in range(5)]))
        assert len(buffer._storage) == 20, len(buffer._storage)
        assert buffer.stats()['added_count'] == 200, buffer.stats()
        state = buffer.get_state()
        new_memory = PrioritizedReplayBuffer(capacity=100, alpha=0.1)
        new_memory.set_state(state)
        assert len(new_memory._storage) == 20, len(new_memory._storage)
        assert new_memory.stats()['added_count'] == 200, new_memory.stats()

    def test_add(self):
        if False:
            return 10
        buffer = PrioritizedReplayBuffer(capacity=2, alpha=self.alpha)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer._next_idx, 0)
        data = self._generate_data()
        buffer.add(data, weight=0.5)
        self.assertTrue(len(buffer) == 1)
        self.assertTrue(buffer._next_idx == 1)
        data = self._generate_data()
        buffer.add(data, weight=0.1)
        self.assertTrue(len(buffer) == 2)
        self.assertTrue(buffer._next_idx == 0)
        data = self._generate_data()
        buffer.add(data, weight=1.0)
        self.assertTrue(len(buffer) == 2)
        self.assertTrue(buffer._next_idx == 1)
        state = buffer.get_state()
        new_memory = PrioritizedReplayBuffer(capacity=2, alpha=self.alpha)
        new_memory.set_state(state)
        self.assertTrue(len(new_memory) == 2)
        self.assertTrue(new_memory._next_idx == 1)

    def test_update_priorities(self):
        if False:
            return 10
        buffer = PrioritizedReplayBuffer(self.capacity, alpha=self.alpha)
        num_records = 5
        for i in range(num_records):
            data = self._generate_data()
            buffer.add(data, weight=1.0)
            self.assertTrue(len(buffer) == i + 1)
            self.assertTrue(buffer._next_idx == i + 1)
        state = buffer.get_state()
        new_memory = PrioritizedReplayBuffer(self.capacity, alpha=self.alpha)
        new_memory.set_state(state)
        self.assertTrue(len(new_memory) == num_records)
        self.assertTrue(new_memory._next_idx == num_records)
        batch = buffer.sample(3, beta=self.beta)
        weights = batch['weights']
        indices = batch['batch_indexes']
        check(weights, np.ones(shape=(3,)))
        self.assertEqual(3, len(indices))
        self.assertTrue(len(buffer) == num_records)
        self.assertTrue(buffer._next_idx == num_records)
        buffer.update_priorities(np.array([0, 2, 3, 4]), np.array([0.01, 0.01, 0.01, 0.01]))
        for _ in range(10):
            batch = buffer.sample(1000, beta=self.beta)
            indices = batch['batch_indexes']
            self.assertTrue(970 < np.sum(indices) < 1100)
        state = buffer.get_state()
        new_memory = PrioritizedReplayBuffer(self.capacity, alpha=self.alpha)
        new_memory.set_state(state)
        batch = new_memory.sample(1000, beta=self.beta)
        indices = batch['batch_indexes']
        self.assertTrue(970 < np.sum(indices) < 1100)
        for _ in range(10):
            rand = np.random.random() + 0.2
            buffer.update_priorities(np.array([0, 1]), np.array([rand, rand]))
            batch = buffer.sample(1000, beta=self.beta)
            indices = batch['batch_indexes']
            self.assertTrue(400 < np.sum(indices) < 800)
        state = buffer.get_state()
        new_memory = PrioritizedReplayBuffer(self.capacity, alpha=self.alpha)
        new_memory.set_state(state)
        batch = new_memory.sample(1000, beta=self.beta)
        indices = batch['batch_indexes']
        self.assertTrue(400 < np.sum(indices) < 800)
        for _ in range(10):
            rand = np.random.random() + 0.2
            buffer.update_priorities(np.array([0, 1]), np.array([rand, rand * 2]))
            batch = buffer.sample(1000, beta=self.beta)
            indices = batch['batch_indexes']
            self.assertTrue(600 < np.sum(indices) < 850)
        state = buffer.get_state()
        new_memory = PrioritizedReplayBuffer(self.capacity, alpha=self.alpha)
        new_memory.set_state(state)
        batch = new_memory.sample(1000, beta=self.beta)
        indices = batch['batch_indexes']
        self.assertTrue(600 < np.sum(indices) < 850)
        for _ in range(10):
            rand = np.random.random() + 0.2
            buffer.update_priorities(np.array([0, 1]), np.array([rand, rand * 4]))
            batch = buffer.sample(1000, beta=self.beta)
            indices = batch['batch_indexes']
            self.assertTrue(750 < np.sum(indices) < 950)
        state = buffer.get_state()
        new_memory = PrioritizedReplayBuffer(self.capacity, alpha=self.alpha)
        new_memory.set_state(state)
        batch = new_memory.sample(1000, beta=self.beta)
        indices = batch['batch_indexes']
        self.assertTrue(750 < np.sum(indices) < 950)
        for _ in range(10):
            rand = np.random.random() + 0.2
            buffer.update_priorities(np.array([0, 1]), np.array([rand, rand * 9]))
            batch = buffer.sample(1000, beta=self.beta)
            indices = batch['batch_indexes']
            self.assertTrue(850 < np.sum(indices) < 1100)
        state = buffer.get_state()
        new_memory = PrioritizedReplayBuffer(self.capacity, alpha=self.alpha)
        new_memory.set_state(state)
        batch = new_memory.sample(1000, beta=self.beta)
        indices = batch['batch_indexes']
        self.assertTrue(850 < np.sum(indices) < 1100)
        num_records = 5
        for i in range(num_records):
            data = self._generate_data()
            buffer.add(data, weight=1.0)
            self.assertTrue(len(buffer) == i + 6)
            self.assertTrue(buffer._next_idx == (i + 6) % self.capacity)
        buffer.update_priorities(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([0.001, 0.1, 2.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]))
        counts = Counter()
        for _ in range(10):
            batch = buffer.sample(np.random.randint(100, 600), beta=self.beta)
            indices = batch['batch_indexes']
            for i in indices:
                counts[i] += 1
        self.assertTrue(counts[9] >= counts[8] >= counts[7] >= counts[6] >= counts[5] >= counts[4] >= counts[3] >= counts[2] >= counts[1] >= counts[0])
        state = buffer.get_state()
        new_memory = PrioritizedReplayBuffer(self.capacity, alpha=self.alpha)
        new_memory.set_state(state)
        counts = Counter()
        for _ in range(10):
            batch = new_memory.sample(np.random.randint(100, 600), beta=self.beta)
            indices = batch['batch_indexes']
            for i in indices:
                counts[i] += 1
        self.assertTrue(counts[9] >= counts[8] >= counts[7] >= counts[6] >= counts[5] >= counts[4] >= counts[3] >= counts[2] >= counts[1] >= counts[0])

    def test_alpha_parameter(self):
        if False:
            for i in range(10):
                print('nop')
        buffer = PrioritizedReplayBuffer(self.capacity, alpha=0.01)
        num_records = 5
        for i in range(num_records):
            data = self._generate_data()
            buffer.add(data, weight=float(np.random.rand()))
            self.assertTrue(len(buffer) == i + 1)
            self.assertTrue(buffer._next_idx == i + 1)
        state = buffer.get_state()
        new_memory = PrioritizedReplayBuffer(self.capacity, alpha=0.01)
        new_memory.set_state(state)
        self.assertTrue(len(new_memory) == num_records)
        self.assertTrue(new_memory._next_idx == num_records)
        batch = buffer.sample(1000, beta=self.beta)
        indices = batch['batch_indexes']
        counts = Counter()
        for i in indices:
            counts[i] += 1
        self.assertTrue(any((100 < i < 300 for i in counts.values())))
        state = buffer.get_state()
        new_memory = PrioritizedReplayBuffer(self.capacity, alpha=0.01)
        new_memory.set_state(state)
        batch = new_memory.sample(1000, beta=self.beta)
        indices = batch['batch_indexes']
        counts = Counter()
        for i in indices:
            counts[i] += 1
        self.assertTrue(any((100 < i < 300 for i in counts.values())))

    def test_sequences_unit(self):
        if False:
            return 10
        'Tests adding, sampling and eviction of sequences.'
        buffer = PrioritizedReplayBuffer(capacity=10, storage_unit='sequences')
        batches = [SampleBatch({SampleBatch.T: i * [np.random.random((4,))], SampleBatch.ACTIONS: i * [np.random.choice([0, 1])], SampleBatch.REWARDS: i * [np.random.rand()], SampleBatch.TERMINATEDS: i * [np.random.choice([False, True])], SampleBatch.TRUNCATEDS: i * [np.random.choice([False, True])], SampleBatch.SEQ_LENS: [i], 'batch_id': i * [i]}) for i in range(1, 4)]
        for batch in batches:
            buffer.add(batch, weight=0.01)
        buffer.add(SampleBatch({SampleBatch.T: 4 * [np.random.random((4,))], SampleBatch.ACTIONS: 4 * [np.random.choice([0, 1])], SampleBatch.REWARDS: 4 * [np.random.rand()], SampleBatch.TERMINATEDS: 4 * [np.random.choice([False, True])], SampleBatch.TRUNCATEDS: 4 * [np.random.choice([False, True])], SampleBatch.SEQ_LENS: [2, 2], 'batch_id': 4 * [4]}), weight=1)
        num_sampled_dict = {_id: 0 for _id in range(1, 5)}
        num_samples = 200
        for i in range(num_samples):
            sample = buffer.sample(1, beta=self.beta)
            _id = sample['batch_id'][0]
            assert len(sample[SampleBatch.SEQ_LENS]) == 1
            num_sampled_dict[_id] += 1
        assert np.allclose(np.array(list(num_sampled_dict.values())) / num_samples, [0.1, 0.1, 0.1, 0.8], atol=0.2)
        buffer.add(SampleBatch({SampleBatch.T: 5 * [np.random.random((4,))], SampleBatch.ACTIONS: 5 * [np.random.choice([0, 1])], SampleBatch.REWARDS: 5 * [np.random.rand()], SampleBatch.TERMINATEDS: 5 * [np.random.choice([False, True])], SampleBatch.TRUNCATEDS: 5 * [np.random.choice([False, True])], SampleBatch.SEQ_LENS: [5], 'batch_id': 5 * [5]}), weight=1)
        assert len(buffer) == 5
        assert buffer._num_timesteps_added == sum(range(1, 6))
        assert buffer._num_timesteps_added_wrap == 5
        assert buffer._next_idx == 1
        assert buffer._eviction_started is True
        num_sampled_dict = {_id: 0 for _id in range(1, 6)}
        num_samples = 200
        for i in range(num_samples):
            sample = buffer.sample(1, beta=self.beta)
            _id = sample['batch_id'][0]
            assert len(sample[SampleBatch.SEQ_LENS]) == 1
            num_sampled_dict[_id] += 1
        assert np.allclose(np.array(list(num_sampled_dict.values())) / num_samples, [0, 0, 0, 0.5, 0.5], atol=0.25)

    def test_episodes_unit(self):
        if False:
            i = 10
            return i + 15
        'Tests adding, sampling, and eviction of episodes.'
        buffer = PrioritizedReplayBuffer(capacity=18, storage_unit='episodes')
        batches = [SampleBatch({SampleBatch.T: [0, 1, 2, 3], SampleBatch.ACTIONS: 4 * [np.random.choice([0, 1])], SampleBatch.REWARDS: 4 * [np.random.rand()], SampleBatch.TERMINATEDS: [False, False, False, True], SampleBatch.TRUNCATEDS: [False, False, False, False], SampleBatch.SEQ_LENS: [4], SampleBatch.EPS_ID: 4 * [i]}) for i in range(3)]
        for batch in batches:
            buffer.add(batch, weight=0.01)
        buffer.add(SampleBatch({SampleBatch.T: [0, 1, 0, 1], SampleBatch.ACTIONS: 4 * [np.random.choice([0, 1])], SampleBatch.REWARDS: 4 * [np.random.rand()], SampleBatch.TERMINATEDS: [False, True, False, True], SampleBatch.TRUNCATEDS: [False, False, False, True], SampleBatch.SEQ_LENS: [2, 2], SampleBatch.EPS_ID: [3, 3, 4, 4]}), weight=1)
        num_sampled_dict = {_id: 0 for _id in range(5)}
        num_samples = 200
        for i in range(num_samples):
            sample = buffer.sample(1, beta=self.beta)
            _id = sample[SampleBatch.EPS_ID][0]
            assert len(sample[SampleBatch.SEQ_LENS]) == 1
            num_sampled_dict[_id] += 1
        assert np.allclose(np.array(list(num_sampled_dict.values())) / num_samples, [0, 0, 0, 0.5, 0.5], atol=0.1)
        buffer.add(SampleBatch({SampleBatch.T: [0, 1, 0, 1], SampleBatch.ACTIONS: 4 * [np.random.choice([0, 1])], SampleBatch.REWARDS: 4 * [np.random.rand()], SampleBatch.TERMINATEDS: [False, True, False, False], SampleBatch.TRUNCATEDS: [False, False, False, False], SampleBatch.SEQ_LENS: [2, 2], SampleBatch.EPS_ID: [5, 5, 6, 6]}), weight=1)
        num_sampled_dict = {_id: 0 for _id in range(7)}
        num_samples = 200
        for i in range(num_samples):
            sample = buffer.sample(1, beta=self.beta)
            _id = sample[SampleBatch.EPS_ID][0]
            assert len(sample[SampleBatch.SEQ_LENS]) == 1
            num_sampled_dict[_id] += 1
        assert np.allclose(np.array(list(num_sampled_dict.values())) / num_samples, [0, 0, 0, 1 / 3, 1 / 3, 1 / 3, 0], atol=0.1)
        buffer.add(SampleBatch({SampleBatch.T: [0, 1, 2, 3], SampleBatch.ACTIONS: 4 * [np.random.choice([0, 1])], SampleBatch.REWARDS: 4 * [np.random.rand()], SampleBatch.TERMINATEDS: [False, False, False, True], SampleBatch.TRUNCATEDS: [False, False, False, True], SampleBatch.SEQ_LENS: [4], SampleBatch.EPS_ID: 4 * [7]}), weight=0.01)
        assert len(buffer) == 6
        assert buffer._num_timesteps_added == 4 * 6 - 2
        assert buffer._num_timesteps_added_wrap == 4
        assert buffer._next_idx == 1
        assert buffer._eviction_started is True
        num_sampled_dict = {_id: 0 for _id in range(8)}
        num_samples = 200
        for i in range(num_samples):
            sample = buffer.sample(1, beta=self.beta)
            _id = sample[SampleBatch.EPS_ID][0]
            assert len(sample[SampleBatch.SEQ_LENS]) == 1
            num_sampled_dict[_id] += 1
        assert np.allclose(np.array(list(num_sampled_dict.values())) / num_samples, [0, 0, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0], atol=0.1)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))