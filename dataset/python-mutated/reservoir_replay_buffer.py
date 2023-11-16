from typing import Any, Dict
import random
import ray
import psutil
from ray.rllib.utils.annotations import ExperimentalAPI, override
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer, warn_replay_capacity
from ray.rllib.utils.typing import SampleBatchType

@ExperimentalAPI
class ReservoirReplayBuffer(ReplayBuffer):
    """This buffer implements reservoir sampling.

    The algorithm has been described by Jeffrey S. Vitter in "Random sampling
    with a reservoir".
    """

    def __init__(self, capacity: int=10000, storage_unit: str='timesteps', **kwargs):
        if False:
            while True:
                i = 10
        "Initializes a ReservoirBuffer instance.\n\n        Args:\n            capacity: Max number of timesteps to store in the FIFO\n                    buffer. After reaching this number, older samples will be\n                    dropped to make space for new ones.\n            storage_unit: Either 'timesteps', 'sequences' or\n                    'episodes'. Specifies how experiences are stored.\n        "
        ReplayBuffer.__init__(self, capacity, storage_unit)
        self._num_add_calls = 0
        self._num_evicted = 0

    @ExperimentalAPI
    @override(ReplayBuffer)
    def _add_single_batch(self, item: SampleBatchType, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        'Add a SampleBatch of experiences to self._storage.\n\n        An item consists of either one or more timesteps, a sequence or an\n        episode. Differs from add() in that it does not consider the storage\n        unit or type of batch and simply stores it.\n\n        Args:\n            item: The batch to be added.\n            ``**kwargs``: Forward compatibility kwargs.\n        '
        self._num_timesteps_added += item.count
        self._num_timesteps_added_wrap += item.count
        self._num_add_calls += 1
        if self._num_timesteps_added < self.capacity:
            self._storage.append(item)
            self._est_size_bytes += item.size_bytes()
        else:
            self._eviction_started = True
            idx = random.randint(0, self._num_add_calls - 1)
            if idx < len(self._storage):
                self._num_evicted += 1
                self._evicted_hit_stats.push(self._hit_count[idx])
                self._hit_count[idx] = 0
                self._next_idx = idx
                self._evicted_hit_stats.push(self._hit_count[idx])
                self._hit_count[idx] = 0
                item_to_be_removed = self._storage[idx]
                self._est_size_bytes -= item_to_be_removed.size_bytes()
                self._storage[idx] = item
                self._est_size_bytes += item.size_bytes()
                assert item.count > 0, item
                warn_replay_capacity(item=item, num_items=self.capacity / item.count)

    @ExperimentalAPI
    @override(ReplayBuffer)
    def stats(self, debug: bool=False) -> dict:
        if False:
            for i in range(10):
                print('nop')
        'Returns the stats of this buffer.\n\n        Args:\n            debug: If True, adds sample eviction statistics to the returned\n                    stats dict.\n\n        Returns:\n            A dictionary of stats about this buffer.\n        '
        data = {'num_evicted': self._num_evicted, 'num_add_calls': self._num_add_calls}
        parent = ReplayBuffer.stats(self, debug)
        parent.update(data)
        return parent

    @ExperimentalAPI
    @override(ReplayBuffer)
    def get_state(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Returns all local state.\n\n        Returns:\n            The serializable local state.\n        '
        parent = ReplayBuffer.get_state(self)
        parent.update(self.stats())
        return parent

    @ExperimentalAPI
    @override(ReplayBuffer)
    def set_state(self, state: Dict[str, Any]) -> None:
        if False:
            return 10
        'Restores all local state to the provided `state`.\n\n        Args:\n            state: The new state to set this buffer. Can be\n                    obtained by calling `self.get_state()`.\n        '
        self._num_evicted = state['num_evicted']
        self._num_add_calls = state['num_add_calls']
        ReplayBuffer.set_state(self, state)