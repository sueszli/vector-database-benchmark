import random
from typing import Any, Dict, List, Optional
import numpy as np
import ray
import psutil
from ray.rllib.execution.segment_tree import SumSegmentTree, MinSegmentTree
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics.window_stat import WindowStat
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer
from ray.rllib.utils.typing import SampleBatchType
from ray.util.annotations import DeveloperAPI

@DeveloperAPI
class PrioritizedReplayBuffer(ReplayBuffer):
    """This buffer implements Prioritized Experience Replay.

    The algorithm has been described by Tom Schaul et. al. in "Prioritized
    Experience Replay". See https://arxiv.org/pdf/1511.05952.pdf for
    the full paper.
    """

    def __init__(self, capacity: int=10000, storage_unit: str='timesteps', alpha: float=1.0, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Initializes a PrioritizedReplayBuffer instance.\n\n        Args:\n            capacity: Max number of timesteps to store in the FIFO\n                buffer. After reaching this number, older samples will be\n                dropped to make space for new ones.\n            storage_unit: Either 'timesteps', 'sequences' or\n                'episodes'. Specifies how experiences are stored.\n            alpha: How much prioritization is used\n                (0.0=no prioritization, 1.0=full prioritization).\n            ``**kwargs``: Forward compatibility kwargs.\n        "
        ReplayBuffer.__init__(self, capacity, storage_unit, **kwargs)
        assert alpha > 0
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < self.capacity:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self._prio_change_stats = WindowStat('reprio', 1000)

    @DeveloperAPI
    @override(ReplayBuffer)
    def _add_single_batch(self, item: SampleBatchType, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add a batch of experiences to self._storage with weight.\n\n        An item consists of either one or more timesteps, a sequence or an\n        episode. Differs from add() in that it does not consider the storage\n        unit or type of batch and simply stores it.\n\n        Args:\n            item: The item to be added.\n            ``**kwargs``: Forward compatibility kwargs.\n        '
        weight = kwargs.get('weight', None)
        if weight is None:
            weight = self._max_priority
        self._it_sum[self._next_idx] = weight ** self._alpha
        self._it_min[self._next_idx] = weight ** self._alpha
        ReplayBuffer._add_single_batch(self, item)

    def _sample_proportional(self, num_items: int) -> List[int]:
        if False:
            i = 10
            return i + 15
        res = []
        for _ in range(num_items):
            mass = random.random() * self._it_sum.sum(0, len(self._storage))
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    @DeveloperAPI
    @override(ReplayBuffer)
    def sample(self, num_items: int, beta: float, **kwargs) -> Optional[SampleBatchType]:
        if False:
            for i in range(10):
                print('nop')
        'Sample `num_items` items from this buffer, including prio. weights.\n\n        Samples in the results may be repeated.\n\n        Examples for storage of SamplesBatches:\n        - If storage unit `timesteps` has been chosen and batches of\n        size 5 have been added, sample(5) will yield a concatenated batch of\n        15 timesteps.\n        - If storage unit \'sequences\' has been chosen and sequences of\n        different lengths have been added, sample(5) will yield a concatenated\n        batch with a number of timesteps equal to the sum of timesteps in\n        the 5 sampled sequences.\n        - If storage unit \'episodes\' has been chosen and episodes of\n        different lengths have been added, sample(5) will yield a concatenated\n        batch with a number of timesteps equal to the sum of timesteps in\n        the 5 sampled episodes.\n\n        Args:\n            num_items: Number of items to sample from this buffer.\n            beta: To what degree to use importance weights (0 - no corrections,\n            1 - full correction).\n            ``**kwargs``: Forward compatibility kwargs.\n\n        Returns:\n            Concatenated SampleBatch of items including "weights" and\n            "batch_indexes" fields denoting IS of each sampled\n            transition and original idxes in buffer of sampled experiences.\n        '
        assert beta >= 0.0
        if len(self) == 0:
            raise ValueError('Trying to sample from an empty buffer.')
        idxes = self._sample_proportional(num_items)
        weights = []
        batch_indexes = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-beta)
            count = self._storage[idx].count
            if isinstance(self._storage[idx], SampleBatch) and self._storage[idx].zero_padded:
                actual_size = self._storage[idx].max_seq_len
            else:
                actual_size = count
            weights.extend([weight / max_weight] * actual_size)
            batch_indexes.extend([idx] * actual_size)
            self._num_timesteps_sampled += count
        batch = self._encode_sample(idxes)
        if isinstance(batch, SampleBatch):
            batch['weights'] = np.array(weights)
            batch['batch_indexes'] = np.array(batch_indexes)
        return batch

    @DeveloperAPI
    def update_priorities(self, idxes: List[int], priorities: List[float]) -> None:
        if False:
            return 10
        'Update priorities of items at given indices.\n\n        Sets priority of item at index idxes[i] in buffer\n        to priorities[i].\n\n        Args:\n            idxes: List of indices of items\n            priorities: List of updated priorities corresponding to items at the\n            idxes denoted by variable `idxes`.\n        '
        assert isinstance(idxes, (list, np.ndarray)), 'ERROR: `idxes` is not a list or np.ndarray, but {}!'.format(type(idxes).__name__)
        assert len(idxes) == len(priorities)
        for (idx, priority) in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            delta = priority ** self._alpha - self._it_sum[idx]
            self._prio_change_stats.push(delta)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)

    @DeveloperAPI
    @override(ReplayBuffer)
    def stats(self, debug: bool=False) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        'Returns the stats of this buffer.\n\n        Args:\n            debug: If true, adds sample eviction statistics to the returned stats dict.\n\n        Returns:\n            A dictionary of stats about this buffer.\n        '
        parent = ReplayBuffer.stats(self, debug)
        if debug:
            parent.update(self._prio_change_stats.stats())
        return parent

    @DeveloperAPI
    @override(ReplayBuffer)
    def get_state(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Returns all local state.\n\n        Returns:\n            The serializable local state.\n        '
        state = super().get_state()
        state.update({'sum_segment_tree': self._it_sum.get_state(), 'min_segment_tree': self._it_min.get_state(), 'max_priority': self._max_priority})
        return state

    @DeveloperAPI
    @override(ReplayBuffer)
    def set_state(self, state: Dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        'Restores all local state to the provided `state`.\n\n        Args:\n            state: The new state to set this buffer. Can be obtained by calling\n            `self.get_state()`.\n        '
        super().set_state(state)
        self._it_sum.set_state(state['sum_segment_tree'])
        self._it_min.set_state(state['min_segment_tree'])
        self._max_priority = state['max_priority']