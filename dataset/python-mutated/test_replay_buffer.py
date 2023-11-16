import unittest
import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, concat_samples
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer

class TestReplayBuffer(unittest.TestCase):
    batch_id = 0

    def _add_data_to_buffer(self, _buffer, batch_size, num_batches=5, **kwargs):
        if False:
            print('Hello World!')

        def _generate_data():
            if False:
                return 10
            return SampleBatch({SampleBatch.T: [np.random.random((4,))], SampleBatch.ACTIONS: [np.random.choice([0, 1])], SampleBatch.OBS: [np.random.random((4,))], SampleBatch.NEXT_OBS: [np.random.random((4,))], SampleBatch.REWARDS: [np.random.rand()], SampleBatch.TERMINATEDS: [np.random.choice([False, True])], SampleBatch.TRUNCATEDS: [np.random.choice([False, False])], 'batch_id': [self.batch_id]})
        for i in range(num_batches):
            data = [_generate_data() for _ in range(batch_size)]
            self.batch_id += 1
            batch = concat_samples(data)
            _buffer.add(batch, **kwargs)

    def test_stats(self):
        if False:
            print('Hello World!')
        "Tests stats by adding and sampling few samples and checking the\n        values of the buffer's stats.\n        "
        self.batch_id = 0
        batch_size = 5
        buffer_size = 15
        buffer = ReplayBuffer(capacity=buffer_size, storage_unit='fragments')
        self._add_data_to_buffer(buffer, batch_size=batch_size, num_batches=1)
        assert len(buffer) == 1
        assert buffer._num_timesteps_added == 5
        assert buffer._num_timesteps_added_wrap == 5
        assert buffer._next_idx == 1
        assert buffer._eviction_started is False
        assert buffer.sample(1)['batch_id'][0] == 0
        buffer.sample(2)
        assert buffer._num_timesteps_sampled == 15
        self._add_data_to_buffer(buffer, batch_size=batch_size, num_batches=2)
        assert len(buffer) == 3
        assert buffer._num_timesteps_added == 15
        assert buffer._num_timesteps_added_wrap == 0
        assert buffer._next_idx == 0
        assert buffer._eviction_started is True

    def test_multi_agent_batches(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests buffer with storage of MultiAgentBatches.'
        self.batch_id = 0

        def _add_multi_agent_batch_to_buffer(buffer, num_policies, num_batches=5, seq_lens=False, **kwargs):
            if False:
                return 10

            def _generate_data(policy_id):
                if False:
                    return 10
                batch = SampleBatch({SampleBatch.T: [0, 1], SampleBatch.ACTIONS: 2 * [np.random.choice([0, 1])], SampleBatch.REWARDS: 2 * [np.random.rand()], SampleBatch.OBS: 2 * [np.random.random((4,))], SampleBatch.NEXT_OBS: 2 * [np.random.random((4,))], SampleBatch.TERMINATEDS: [False, True], SampleBatch.TRUNCATEDS: [False, False], SampleBatch.EPS_ID: 2 * [self.batch_id], SampleBatch.AGENT_INDEX: 2 * [0], SampleBatch.SEQ_LENS: [2], 'batch_id': 2 * [self.batch_id], 'policy_id': 2 * [policy_id]})
                if not seq_lens:
                    del batch[SampleBatch.SEQ_LENS]
                self.batch_id += 1
                return batch
            for i in range(num_batches):
                policy_batches = {idx: _generate_data(idx) for (idx, _) in enumerate(range(num_policies))}
                batch = MultiAgentBatch(policy_batches, num_batches * 2)
                buffer.add(batch, **kwargs)
        buffer = ReplayBuffer(capacity=100, storage_unit='fragments')
        _add_multi_agent_batch_to_buffer(buffer, num_policies=2, num_batches=2)
        assert len(buffer) == 2
        assert buffer._num_timesteps_added == 8
        assert buffer._num_timesteps_added_wrap == 8
        assert buffer._next_idx == 2
        assert buffer._eviction_started is False
        buffer.sample(3)
        assert buffer._num_timesteps_sampled == 12
        _add_multi_agent_batch_to_buffer(buffer, batch_size=100, num_policies=3, num_batches=3)
        assert len(buffer) == 5
        assert buffer._num_timesteps_added == 26
        assert buffer._num_timesteps_added_wrap == 26
        assert buffer._next_idx == 5

    def test_timesteps_unit(self):
        if False:
            print('Hello World!')
        'Tests adding, sampling, get-/set state, and eviction with\n        experiences stored by timesteps.\n        '
        self.batch_id = 0
        batch_size = 5
        buffer_size = 15
        buffer = ReplayBuffer(capacity=buffer_size)
        self._add_data_to_buffer(buffer, batch_size=batch_size, num_batches=1)
        self._add_data_to_buffer(buffer, batch_size=batch_size, num_batches=2)
        num_sampled_dict = {_id: 0 for _id in range(self.batch_id)}
        num_samples = 200
        for i in range(num_samples):
            _id = buffer.sample(1)['batch_id'][0]
            num_sampled_dict[_id] += 1
        assert np.allclose(np.array(list(num_sampled_dict.values())) / num_samples, len(num_sampled_dict) * [1 / 3], atol=0.1)
        state = buffer.get_state()
        other_buffer = ReplayBuffer(capacity=buffer_size)
        self._add_data_to_buffer(other_buffer, 1)
        other_buffer.set_state(state)
        assert other_buffer._storage == buffer._storage
        assert other_buffer._next_idx == buffer._next_idx
        assert other_buffer._num_timesteps_added == buffer._num_timesteps_added
        assert other_buffer._num_timesteps_added_wrap == buffer._num_timesteps_added_wrap
        assert other_buffer._num_timesteps_sampled == buffer._num_timesteps_sampled
        assert other_buffer._eviction_started == buffer._eviction_started
        assert other_buffer._est_size_bytes == buffer._est_size_bytes
        assert len(other_buffer) == len(other_buffer)

    def test_sequences_unit(self):
        if False:
            return 10
        'Tests adding, sampling and eviction of sequences.'
        buffer = ReplayBuffer(capacity=10, storage_unit='sequences')
        batches = [SampleBatch({SampleBatch.T: i * [np.random.random((4,))], SampleBatch.ACTIONS: i * [np.random.choice([0, 1])], SampleBatch.REWARDS: i * [np.random.rand()], SampleBatch.TERMINATEDS: i * [np.random.choice([False, True])], SampleBatch.TRUNCATEDS: i * [np.random.choice([False, True])], SampleBatch.SEQ_LENS: [i], 'batch_id': i * [i]}) for i in range(1, 4)]
        batches.append(SampleBatch({SampleBatch.T: 4 * [np.random.random((4,))], SampleBatch.ACTIONS: 4 * [np.random.choice([0, 1])], SampleBatch.REWARDS: 4 * [np.random.rand()], SampleBatch.TERMINATEDS: 4 * [np.random.choice([False, True])], SampleBatch.TRUNCATEDS: 4 * [np.random.choice([False, True])], SampleBatch.SEQ_LENS: [2, 2], 'batch_id': 4 * [4]}))
        for batch in batches:
            buffer.add(batch)
        num_sampled_dict = {_id: 0 for _id in range(1, 5)}
        num_samples = 200
        for i in range(num_samples):
            sample = buffer.sample(1)
            _id = sample['batch_id'][0]
            assert len(sample[SampleBatch.SEQ_LENS]) == 1
            num_sampled_dict[_id] += 1
        assert np.allclose(np.array(list(num_sampled_dict.values())) / num_samples, [1 / 5, 1 / 5, 1 / 5, 2 / 5], atol=0.1)
        buffer.add(SampleBatch({SampleBatch.T: 5 * [np.random.random((4,))], SampleBatch.ACTIONS: 5 * [np.random.choice([0, 1])], SampleBatch.REWARDS: 5 * [np.random.rand()], SampleBatch.TERMINATEDS: 5 * [np.random.choice([False, True])], SampleBatch.TRUNCATEDS: 5 * [np.random.choice([False, True])], SampleBatch.SEQ_LENS: [5], 'batch_id': 5 * [5]}))
        assert len(buffer) == 5
        assert buffer._num_timesteps_added == sum(range(1, 6))
        assert buffer._num_timesteps_added_wrap == 5
        assert buffer._next_idx == 1
        assert buffer._eviction_started is True
        num_sampled_dict = {_id: 0 for _id in range(2, 6)}
        num_samples = 200
        for i in range(num_samples):
            sample = buffer.sample(1)
            _id = sample['batch_id'][0]
            assert len(sample[SampleBatch.SEQ_LENS]) == 1
            num_sampled_dict[_id] += 1
        assert np.allclose(np.array(list(num_sampled_dict.values())) / num_samples, [1 / 5, 1 / 5, 2 / 5, 1 / 5], atol=0.1)

    def test_episodes_unit(self):
        if False:
            i = 10
            return i + 15
        'Tests adding, sampling, and eviction of episodes.'
        buffer = ReplayBuffer(capacity=18, storage_unit='episodes')
        batches = [SampleBatch({SampleBatch.T: [0, 1, 2, 3], SampleBatch.ACTIONS: 4 * [np.random.choice([0, 1])], SampleBatch.REWARDS: 4 * [np.random.rand()], SampleBatch.TERMINATEDS: [False, False, False, True], SampleBatch.TRUNCATEDS: [False, False, False, False], SampleBatch.SEQ_LENS: [4], SampleBatch.EPS_ID: 4 * [i]}) for i in range(3)]
        batches.append(SampleBatch({SampleBatch.T: [0, 1, 0, 1], SampleBatch.ACTIONS: 4 * [np.random.choice([0, 1])], SampleBatch.REWARDS: 4 * [np.random.rand()], SampleBatch.TERMINATEDS: [False, True, False, True], SampleBatch.TRUNCATEDS: [False, False, False, False], SampleBatch.SEQ_LENS: [2, 2], SampleBatch.EPS_ID: [3, 3, 4, 4]}))
        for batch in batches:
            buffer.add(batch)
        num_sampled_dict = {_id: 0 for _id in range(5)}
        num_samples = 200
        for i in range(num_samples):
            sample = buffer.sample(1)
            _id = sample[SampleBatch.EPS_ID][0]
            assert len(sample[SampleBatch.SEQ_LENS]) == 1
            num_sampled_dict[_id] += 1
        assert np.allclose(np.array(list(num_sampled_dict.values())) / num_samples, [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5], atol=0.1)
        buffer.add(SampleBatch({SampleBatch.T: [0, 1, 0, 1], SampleBatch.ACTIONS: 4 * [np.random.choice([0, 1])], SampleBatch.REWARDS: 4 * [np.random.rand()], SampleBatch.TERMINATEDS: [False, True, False, False], SampleBatch.TRUNCATEDS: [False, False, False, False], SampleBatch.SEQ_LENS: [2, 2], SampleBatch.EPS_ID: [5, 5, 6, 6]}))
        num_sampled_dict = {_id: 0 for _id in range(7)}
        num_samples = 200
        for i in range(num_samples):
            sample = buffer.sample(1)
            _id = sample[SampleBatch.EPS_ID][0]
            assert len(sample[SampleBatch.SEQ_LENS]) == 1
            num_sampled_dict[_id] += 1
        assert np.allclose(np.array(list(num_sampled_dict.values())) / num_samples, [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 0], atol=0.1)
        buffer.add(SampleBatch({SampleBatch.T: [0, 1, 2, 3], SampleBatch.ACTIONS: 4 * [np.random.choice([0, 1])], SampleBatch.REWARDS: 4 * [np.random.rand()], SampleBatch.TERMINATEDS: [False, False, False, True], SampleBatch.TRUNCATEDS: [False, False, False, False], SampleBatch.SEQ_LENS: [4], SampleBatch.EPS_ID: 4 * [7]}))
        assert len(buffer) == 6
        assert buffer._num_timesteps_added == 4 * 6 - 2
        assert buffer._num_timesteps_added_wrap == 4
        assert buffer._next_idx == 1
        assert buffer._eviction_started is True
        num_sampled_dict = {_id: 0 for _id in range(8)}
        num_samples = 200
        for i in range(num_samples):
            sample = buffer.sample(1)
            _id = sample[SampleBatch.EPS_ID][0]
            assert len(sample[SampleBatch.SEQ_LENS]) == 1
            num_sampled_dict[_id] += 1
        assert np.allclose(np.array(list(num_sampled_dict.values())) / num_samples, [0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 0, 1 / 6], atol=0.1)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))