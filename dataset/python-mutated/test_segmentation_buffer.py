import unittest
from typing import List, Union
import numpy as np
from rllib_dt.dt.segmentation_buffer import MultiAgentSegmentationBuffer, SegmentationBuffer
import ray
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, MultiAgentBatch, SampleBatch, concat_samples
from ray.rllib.utils import test_utils
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import PolicyID
(tf1, tf, tfv) = try_import_tf()
(torch, _) = try_import_torch()

def _generate_episode_batch(ep_len, eps_id, obs_dim=8, act_dim=3):
    if False:
        print('Hello World!')
    'Generate a batch containing one episode.'
    batch = SampleBatch({SampleBatch.OBS: np.full((ep_len, obs_dim), eps_id, dtype=np.float32), SampleBatch.ACTIONS: np.full((ep_len, act_dim), eps_id + 100, dtype=np.float32), SampleBatch.REWARDS: np.ones((ep_len,), dtype=np.float32), SampleBatch.RETURNS_TO_GO: np.arange(ep_len, -1, -1, dtype=np.float32).reshape((ep_len + 1, 1)), SampleBatch.EPS_ID: np.full((ep_len,), eps_id, dtype=np.int32), SampleBatch.T: np.arange(ep_len, dtype=np.int32), SampleBatch.ATTENTION_MASKS: np.ones(ep_len, dtype=np.float32), SampleBatch.TERMINATEDS: np.array([False] * (ep_len - 1) + [True]), SampleBatch.TRUNCATEDS: np.array([False] * ep_len)})
    return batch

def _assert_sample_batch_keys(batch: SampleBatch):
    if False:
        for i in range(10):
            print('nop')
    'Assert sampled batch has the requisite keys.'
    assert SampleBatch.OBS in batch
    assert SampleBatch.ACTIONS in batch
    assert SampleBatch.RETURNS_TO_GO in batch
    assert SampleBatch.T in batch
    assert SampleBatch.ATTENTION_MASKS in batch

def _assert_sample_batch_not_equal(b1: SampleBatch, b2: SampleBatch):
    if False:
        for i in range(10):
            print('nop')
    'Assert that the two batches are not equal.'
    for key in b1.keys() & b2.keys():
        if b1[key].shape == b2[key].shape:
            assert not np.allclose(b1[key], b2[key]), f'Key {key} contain the same value when they should not.'

def _assert_is_segment(segment: SampleBatch, episode: SampleBatch):
    if False:
        print('Hello World!')
    'Assert that the sampled segment is a segment of episode.'
    timesteps = segment[SampleBatch.T]
    masks = segment[SampleBatch.ATTENTION_MASKS] > 0.5
    seq_len = timesteps.shape[0]
    episode_segment = episode.slice(timesteps[0], timesteps[-1] + 1)
    assert np.allclose(segment[SampleBatch.OBS][masks], episode_segment[SampleBatch.OBS])
    assert np.allclose(segment[SampleBatch.ACTIONS][masks], episode_segment[SampleBatch.ACTIONS])
    assert np.allclose(segment[SampleBatch.RETURNS_TO_GO][:seq_len][masks], episode_segment[SampleBatch.RETURNS_TO_GO])

def _get_internal_buffer(buffer: Union[SegmentationBuffer, MultiAgentSegmentationBuffer], policy_id: PolicyID=DEFAULT_POLICY_ID) -> List[SampleBatch]:
    if False:
        return 10
    'Get the internal buffer list from the buffer. If MultiAgent then return the\n    internal buffer corresponding to the given policy_id.\n    '
    if type(buffer) == SegmentationBuffer:
        return buffer._buffer
    elif type(buffer) == MultiAgentSegmentationBuffer:
        return buffer.buffers[policy_id]._buffer
    else:
        raise NotImplementedError

def _as_sample_batch(batch: Union[SampleBatch, MultiAgentBatch], policy_id: PolicyID=DEFAULT_POLICY_ID) -> SampleBatch:
    if False:
        print('Hello World!')
    'Returns a SampleBatch. If MultiAgentBatch then return the SampleBatch\n    corresponding to the given policy_id.\n    '
    if type(batch) == SampleBatch:
        return batch
    elif type(batch) == MultiAgentBatch:
        return batch.policy_batches[policy_id]
    else:
        raise NotImplementedError

class TestSegmentationBuffer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        ray.init()

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        ray.shutdown()

    def test_add(self):
        if False:
            while True:
                i = 10
        'Test adding to segmentation buffer.'
        for buffer_cls in [SegmentationBuffer, MultiAgentSegmentationBuffer]:
            max_seq_len = 3
            max_ep_len = 10
            capacity = 1
            buffer = buffer_cls(capacity, max_seq_len, max_ep_len)
            episode_batches = []
            for i in range(4):
                episode_batches.append(_generate_episode_batch(max_ep_len, i))
            batch = concat_samples(episode_batches)
            buffer.add(batch)
            self.assertEqual(len(_get_internal_buffer(buffer)), 1, 'The internal buffer should only contain one SampleBatch since the capacity is 1.')
            test_utils.check(episode_batches[-1], _get_internal_buffer(buffer)[0])
            buffer.add(episode_batches[0])
            test_utils.check(episode_batches[0], _get_internal_buffer(buffer)[0])
            capacity = len(episode_batches)
            buffer = buffer_cls(capacity, max_seq_len, max_ep_len)
            buffer.add(batch)
            self.assertEqual(len(_get_internal_buffer(buffer)), len(episode_batches), "internal buffer doesn't have the right number of episodes.")
            for i in range(len(episode_batches)):
                test_utils.check(episode_batches[i], _get_internal_buffer(buffer)[i])
            new_batch = _generate_episode_batch(max_ep_len, 12345)
            buffer.add(new_batch)
            self.assertEqual(len(_get_internal_buffer(buffer)), len(episode_batches), "internal buffer doesn't have the right number of episodes.")
            found = False
            for episode_batch in _get_internal_buffer(buffer):
                if episode_batch[SampleBatch.EPS_ID][0] == 12345:
                    test_utils.check(episode_batch, new_batch)
                    found = True
                    break
            assert found, 'new_batch not added to buffer.'
            long_batch = _generate_episode_batch(max_ep_len + 1, 123)
            with self.assertRaises(ValueError):
                buffer.add(long_batch)

    def test_sample_basic(self):
        if False:
            for i in range(10):
                print('nop')
        'Test sampling from a segmentation buffer.'
        for buffer_cls in (SegmentationBuffer, MultiAgentSegmentationBuffer):
            max_seq_len = 5
            max_ep_len = 15
            capacity = 4
            obs_dim = 10
            act_dim = 2
            buffer = buffer_cls(capacity, max_seq_len, max_ep_len)
            episode_batches = []
            for i in range(8):
                episode_batches.append(_generate_episode_batch(max_ep_len, i, obs_dim, act_dim))
            batch = concat_samples(episode_batches)
            buffer.add(batch)
            for bs in range(10, 20):
                batch = _as_sample_batch(buffer.sample(bs))
                _assert_sample_batch_keys(batch)
                self.assertEquals(batch[SampleBatch.OBS].shape, (bs, max_seq_len, obs_dim))
                self.assertEquals(batch[SampleBatch.ACTIONS].shape, (bs, max_seq_len, act_dim))
                self.assertEquals(batch[SampleBatch.RETURNS_TO_GO].shape, (bs, max_seq_len + 1, 1))
                self.assertEquals(batch[SampleBatch.T].shape, (bs, max_seq_len))
                self.assertEquals(batch[SampleBatch.ATTENTION_MASKS].shape, (bs, max_seq_len))

    def test_sample_content(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the content of the sampling are valid.'
        for buffer_cls in (SegmentationBuffer, MultiAgentSegmentationBuffer):
            max_seq_len = 5
            max_ep_len = 200
            capacity = 1
            obs_dim = 11
            act_dim = 1
            buffer = buffer_cls(capacity, max_seq_len, max_ep_len)
            episode = _generate_episode_batch(max_ep_len, 123, obs_dim, act_dim)
            buffer.add(episode)
            sample1 = _as_sample_batch(buffer.sample(200))
            sample2 = _as_sample_batch(buffer.sample(200))
            _assert_sample_batch_keys(sample1)
            _assert_sample_batch_keys(sample2)
            _assert_sample_batch_not_equal(sample1, sample2)
            batch = _as_sample_batch(buffer.sample(1000))
            _assert_sample_batch_keys(batch)
            for elem in batch.rows():
                _assert_is_segment(SampleBatch(elem), episode)

    def test_sample_capacity(self):
        if False:
            while True:
                i = 10
        'Test that sampling from buffer of capacity > 1 works.'
        for buffer_cls in (SegmentationBuffer, MultiAgentSegmentationBuffer):
            max_seq_len = 3
            max_ep_len = 10
            capacity = 100
            obs_dim = 1
            act_dim = 1
            buffer = buffer_cls(capacity, max_seq_len, max_ep_len)
            episode_batches = []
            for i in range(capacity):
                episode_batches.append(_generate_episode_batch(max_ep_len, i, obs_dim, act_dim))
            buffer.add(concat_samples(episode_batches))
            batch = _as_sample_batch(buffer.sample(100))
            eps_ids = set()
            for i in range(100):
                eps_id = int(batch[SampleBatch.OBS][i, -1, 0])
                eps_ids.add(eps_id)
            self.assertGreater(len(eps_ids), 1, 'buffer.sample is always returning the same episode.')

    def test_padding(self):
        if False:
            print('Hello World!')
        'Test that sample will front pad segments.'
        for buffer_cls in (SegmentationBuffer, MultiAgentSegmentationBuffer):
            max_seq_len = 10
            max_ep_len = 100
            capacity = 1
            obs_dim = 3
            act_dim = 2
            buffer = buffer_cls(capacity, max_seq_len, max_ep_len)
            for ep_len in range(1, max_seq_len):
                batch = _generate_episode_batch(ep_len, 123, obs_dim, act_dim)
                buffer.add(batch)
                samples = _as_sample_batch(buffer.sample(50))
                for i in range(50):
                    num_pad = int(ep_len - samples[SampleBatch.ATTENTION_MASKS][i].sum())
                    for key in samples.keys():
                        assert np.allclose(samples[key][i, :num_pad], 0.0), 'samples were not padded correctly.'

    def test_multi_agent(self):
        if False:
            print('Hello World!')
        max_seq_len = 5
        max_ep_len = 20
        capacity = 10
        obs_dim = 3
        act_dim = 5
        ma_buffer = MultiAgentSegmentationBuffer(capacity, max_seq_len, max_ep_len)
        policy_id1 = '1'
        policy_id2 = '2'
        policy_id3 = '3'
        policy_ids = {policy_id1, policy_id2, policy_id3}
        policy1_batches = []
        for i in range(0, 10):
            policy1_batches.append(_generate_episode_batch(max_ep_len, i, obs_dim, act_dim))
        policy2_batches = []
        for i in range(10, 20):
            policy2_batches.append(_generate_episode_batch(max_ep_len, i, obs_dim, act_dim))
        policy3_batches = []
        for i in range(20, 30):
            policy3_batches.append(_generate_episode_batch(max_ep_len, i, obs_dim, act_dim))
        batches_mapping = {policy_id1: policy1_batches, policy_id2: policy2_batches, policy_id3: policy3_batches}
        ma_batch = MultiAgentBatch({policy_id1: concat_samples(policy1_batches), policy_id2: concat_samples(policy2_batches), policy_id3: concat_samples(policy3_batches)}, max_ep_len * 10)
        ma_buffer.add(ma_batch)
        for policy_id in policy_ids:
            assert policy_id in ma_buffer.buffers.keys()
        for (policy_id, buffer) in ma_buffer.buffers.items():
            assert policy_id in policy_ids
            for i in range(10):
                test_utils.check(batches_mapping[policy_id][i], _get_internal_buffer(buffer)[i])
        for _ in range(50):
            ma_sample = ma_buffer.sample(100)
            for policy_id in policy_ids:
                assert policy_id in ma_sample.policy_batches.keys()
            for (policy_id, batch) in ma_sample.policy_batches.items():
                eps_id_start = (int(policy_id) - 1) * 10
                eps_id_end = eps_id_start + 10
                _assert_sample_batch_keys(batch)
                for i in range(100):
                    eps_id = int(batch[SampleBatch.OBS][i, -1, 0])
                    assert eps_id_start <= eps_id < eps_id_end, "batch within multi agent batch has the wrong agent's episode."
        ma_sample1 = ma_buffer.sample(200)
        ma_sample2 = ma_buffer.sample(200)
        for policy_id in policy_ids:
            _assert_sample_batch_not_equal(ma_sample1.policy_batches[policy_id], ma_sample2.policy_batches[policy_id])
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', __file__]))