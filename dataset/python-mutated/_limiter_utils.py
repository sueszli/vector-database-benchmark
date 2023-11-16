import collections
from typing import Deque, Optional
import torch

class _FreeEventQueue:
    """
    This tracks all pending frees corresponding to inflight all-gathers. The
    queueing pattern is iterative enqueues with a single dequeue per iteration
    once the limit ``_max_num_inflight_all_gathers`` is reached.
    """

    def __init__(self) -> None:
        if False:
            return 10
        self._queue: Deque[torch.cuda.Event] = collections.deque()
        self._max_num_inflight_all_gathers = 2

    def enqueue(self, free_event: torch.cuda.Event) -> None:
        if False:
            i = 10
            return i + 15
        'Enqueues a free event.'
        self._queue.append(free_event)

    def dequeue_if_needed(self) -> Optional[torch.cuda.Event]:
        if False:
            while True:
                i = 10
        'Dequeues a single event if the limit is reached.'
        if len(self._queue) >= self._max_num_inflight_all_gathers:
            return self._dequeue()
        return None

    def _dequeue(self) -> Optional[torch.cuda.Event]:
        if False:
            print('Hello World!')
        'Dequeues a free event if possible.'
        if self._queue:
            event = self._queue.popleft()
            return event
        return None