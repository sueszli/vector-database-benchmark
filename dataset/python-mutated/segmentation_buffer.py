import logging
import random
from collections import defaultdict
from typing import List
import numpy as np
from ray.rllib.evaluation.postprocessing import discount_cumsum
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch, concat_samples
from ray.rllib.utils.typing import SampleBatchType
logger = logging.getLogger(__name__)

def front_pad_with_zero(arr: np.ndarray, max_seq_len: int):
    if False:
        return 10
    'Pad arr on the front/left with 0 up to max_seq_len.'
    length = arr.shape[0]
    pad_length = max_seq_len - length
    if pad_length > 0:
        return np.concatenate([np.zeros((pad_length, *arr.shape[1:]), dtype=arr.dtype), arr], axis=0)
    else:
        return arr

class SegmentationBuffer:
    """A minimal replay buffer used by Decision Transformer (DT)
    to process episodes into max_seq_len length segments and do shuffling.
    """

    def __init__(self, capacity: int=20, max_seq_len: int=20, max_ep_len: int=1000):
        if False:
            return 10
        '\n        Args:\n            capacity: Maximum number of episodes the buffer can store.\n            max_seq_len: Length of segments that are sampled.\n            max_ep_len: Maximum length of episodes added.\n        '
        self.capacity = capacity
        self.max_seq_len = max_seq_len
        self.max_ep_len = max_ep_len
        self._buffer: List[SampleBatch] = []

    def add(self, batch: SampleBatch):
        if False:
            return 10
        'Add a SampleBatch of episodes. Replace if full.\n\n        Args:\n            batch: SampleBatch of full episodes.\n        '
        episodes = batch.split_by_episode(key=SampleBatch.DONES)
        for episode in episodes:
            self._add_single_episode(episode)

    def _add_single_episode(self, episode: SampleBatch):
        if False:
            for i in range(10):
                print('nop')
        ep_len = episode.env_steps()
        if ep_len > self.max_ep_len:
            raise ValueError(f'The maximum rollout length is {self.max_ep_len} but we tried to add arollout of {episode.env_steps()} steps to the SegmentationBuffer.')
        rewards = episode[SampleBatch.REWARDS].reshape(-1)
        rtg = discount_cumsum(rewards, 1.0)
        rtg = np.concatenate([rtg, np.zeros((1,), dtype=np.float32)], axis=0)
        episode[SampleBatch.RETURNS_TO_GO] = rtg[:, None]
        episode[SampleBatch.T] = np.arange(ep_len, dtype=np.int32)
        episode[SampleBatch.ATTENTION_MASKS] = np.ones(ep_len, dtype=np.float32)
        if len(self._buffer) < self.capacity:
            self._buffer.append(episode)
        else:
            replace_ind = random.randint(0, self.capacity - 1)
            self._buffer[replace_ind] = episode

    def sample(self, batch_size: int) -> SampleBatch:
        if False:
            while True:
                i = 10
        'Sample segments from the buffer.\n\n        Args:\n            batch_size: number of segments to sample.\n\n        Returns:\n            SampleBatch of segments with keys and shape {\n                OBS: [batch_size, max_seq_len, obs_dim],\n                ACTIONS: [batch_size, max_seq_len, act_dim],\n                RETURNS_TO_GO: [batch_size, max_seq_len + 1, 1],\n                T: [batch_size, max_seq_len],\n                ATTENTION_MASKS: [batch_size, max_seq_len],\n            }\n        '
        samples = [self._sample_single() for _ in range(batch_size)]
        return concat_samples(samples)

    def _sample_single(self) -> SampleBatch:
        if False:
            return 10
        buffer_ind = random.randint(0, len(self._buffer) - 1)
        episode = self._buffer[buffer_ind]
        ep_len = episode[SampleBatch.OBS].shape[0]
        ei = random.randint(1, ep_len)
        si = max(ei - self.max_seq_len, 0)
        obs = episode[SampleBatch.OBS][si:ei]
        actions = episode[SampleBatch.ACTIONS][si:ei]
        timesteps = episode[SampleBatch.T][si:ei]
        masks = episode[SampleBatch.ATTENTION_MASKS][si:ei]
        returns_to_go = episode[SampleBatch.RETURNS_TO_GO][si:ei + 1]
        obs = front_pad_with_zero(obs, self.max_seq_len)
        actions = front_pad_with_zero(actions, self.max_seq_len)
        returns_to_go = front_pad_with_zero(returns_to_go, self.max_seq_len + 1)
        timesteps = front_pad_with_zero(timesteps, self.max_seq_len)
        masks = front_pad_with_zero(masks, self.max_seq_len)
        assert obs.shape[0] == self.max_seq_len
        assert actions.shape[0] == self.max_seq_len
        assert timesteps.shape[0] == self.max_seq_len
        assert masks.shape[0] == self.max_seq_len
        assert returns_to_go.shape[0] == self.max_seq_len + 1
        return SampleBatch({SampleBatch.OBS: obs[None], SampleBatch.ACTIONS: actions[None], SampleBatch.RETURNS_TO_GO: returns_to_go[None], SampleBatch.T: timesteps[None], SampleBatch.ATTENTION_MASKS: masks[None]})

class MultiAgentSegmentationBuffer:
    """A minimal replay buffer used by Decision Transformer (DT)
    to process episodes into max_seq_len length segments and do shuffling.
    Stores MultiAgentSample.
    """

    def __init__(self, capacity: int=20, max_seq_len: int=20, max_ep_len: int=1000):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            capacity: Maximum number of episodes the buffer can store.\n            max_seq_len: Length of segments that are sampled.\n            max_ep_len: Maximum length of episodes added.\n        '

        def new_buffer():
            if False:
                print('Hello World!')
            return SegmentationBuffer(capacity, max_seq_len, max_ep_len)
        self.buffers = defaultdict(new_buffer)

    def add(self, batch: SampleBatchType):
        if False:
            i = 10
            return i + 15
        'Add a MultiAgentBatch of episodes. Replace if full.\n\n        Args:\n            batch: MultiAgentBatch of full episodes.\n        '
        batch = batch.copy()
        batch = batch.as_multi_agent()
        for (policy_id, sample_batch) in batch.policy_batches.items():
            self.buffers[policy_id].add(sample_batch)

    def sample(self, batch_size: int) -> MultiAgentBatch:
        if False:
            print('Hello World!')
        'Sample segments from the buffer.\n\n        Args:\n            batch_size: number of segments to sample.\n\n        Returns:\n            MultiAgentBatch of segments with keys and shape {\n                OBS: [batch_size, max_seq_len, obs_dim],\n                ACTIONS: [batch_size, max_seq_len, act_dim],\n                RETURNS_TO_GO: [batch_size, max_seq_len + 1, 1],\n                T: [batch_size, max_seq_len],\n                ATTENTION_MASKS: [batch_size, max_seq_len],\n            }\n        '
        samples = {}
        for (policy_id, buffer) in self.buffers.items():
            samples[policy_id] = buffer.sample(batch_size)
        return MultiAgentBatch(samples, batch_size)