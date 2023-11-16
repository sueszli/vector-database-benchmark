import numpy as np
from typing import Any, Dict, Optional
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer, StorageUnit
from ray.rllib.utils.typing import SampleBatchType
from ray.util.annotations import DeveloperAPI

@DeveloperAPI
class FifoReplayBuffer(ReplayBuffer):
    """This replay buffer implements a FIFO queue.

    Sometimes, e.g. for offline use cases, it may be desirable to use
    off-policy algorithms without a Replay Buffer.
    This FifoReplayBuffer can be used in-place to achieve the same effect
    without having to introduce separate algorithm execution branches.

    For simplicity and efficiency reasons, this replay buffer stores incoming
    sample batches as-is, and returns them one at time.
    This is to avoid any additional load when this replay buffer is used.
    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a FifoReplayBuffer.\n\n        Args:\n            ``*args``   : Forward compatibility args.\n            ``**kwargs``: Forward compatibility kwargs.\n        '
        ReplayBuffer.__init__(self, 1, StorageUnit.FRAGMENTS, **kwargs)
        self._queue = []

    @DeveloperAPI
    @override(ReplayBuffer)
    def add(self, batch: SampleBatchType, **kwargs) -> None:
        if False:
            return 10
        return self._queue.append(batch)

    @DeveloperAPI
    @override(ReplayBuffer)
    def sample(self, *args, **kwargs) -> Optional[SampleBatchType]:
        if False:
            print('Hello World!')
        'Sample a saved training batch from this buffer.\n\n        Args:\n            ``*args``   : Forward compatibility args.\n            ``**kwargs``: Forward compatibility kwargs.\n\n        Returns:\n            A single training batch from the queue.\n        '
        if len(self._queue) <= 0:
            return MultiAgentBatch({}, 0)
        batch = self._queue.pop(0)
        batch['weights'] = np.ones(len(batch))
        return batch

    @DeveloperAPI
    def update_priorities(self, *args, **kwargs) -> None:
        if False:
            return 10
        'Update priorities of items at given indices.\n\n        No-op for this replay buffer.\n\n        Args:\n            ``*args``   : Forward compatibility args.\n            ``**kwargs``: Forward compatibility kwargs.\n        '
        pass

    @DeveloperAPI
    @override(ReplayBuffer)
    def stats(self, debug: bool=False) -> Dict:
        if False:
            i = 10
            return i + 15
        'Returns the stats of this buffer.\n\n        Args:\n            debug: If true, adds sample eviction statistics to the returned stats dict.\n\n        Returns:\n            A dictionary of stats about this buffer.\n        '
        return {}

    @DeveloperAPI
    @override(ReplayBuffer)
    def get_state(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Returns all local state.\n\n        Returns:\n            The serializable local state.\n        '
        return {}

    @DeveloperAPI
    @override(ReplayBuffer)
    def set_state(self, state: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        'Restores all local state to the provided `state`.\n\n        Args:\n            state: The new state to set this buffer. Can be obtained by calling\n            `self.get_state()`.\n        '
        pass