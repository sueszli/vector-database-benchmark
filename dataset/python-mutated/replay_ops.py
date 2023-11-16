from typing import Optional
import random
from ray.rllib.utils.replay_buffers.replay_buffer import warn_replay_capacity
from ray.rllib.utils.typing import SampleBatchType

class SimpleReplayBuffer:
    """Simple replay buffer that operates over batches."""

    def __init__(self, num_slots: int, replay_proportion: Optional[float]=None):
        if False:
            i = 10
            return i + 15
        'Initialize SimpleReplayBuffer.\n\n        Args:\n            num_slots: Number of batches to store in total.\n        '
        self.num_slots = num_slots
        self.replay_batches = []
        self.replay_index = 0

    def add_batch(self, sample_batch: SampleBatchType) -> None:
        if False:
            i = 10
            return i + 15
        warn_replay_capacity(item=sample_batch, num_items=self.num_slots)
        if self.num_slots > 0:
            if len(self.replay_batches) < self.num_slots:
                self.replay_batches.append(sample_batch)
            else:
                self.replay_batches[self.replay_index] = sample_batch
                self.replay_index += 1
                self.replay_index %= self.num_slots

    def replay(self) -> SampleBatchType:
        if False:
            while True:
                i = 10
        return random.choice(self.replay_batches)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.replay_batches)