from typing import Any, Tuple
import queue

class MinibatchBuffer:
    """Ring buffer of recent data batches for minibatch SGD.

    This is for use with AsyncSamplesOptimizer.
    """

    def __init__(self, inqueue: queue.Queue, size: int, timeout: float, num_passes: int, init_num_passes: int=1):
        if False:
            print('Hello World!')
        'Initialize a minibatch buffer.\n\n        Args:\n           inqueue (queue.Queue): Queue to populate the internal ring buffer\n           from.\n           size: Max number of data items to buffer.\n           timeout: Queue timeout\n           num_passes: Max num times each data item should be emitted.\n           init_num_passes: Initial passes for each data item.\n           Maxiumum number of passes per item are increased to num_passes over\n           time.\n        '
        self.inqueue = inqueue
        self.size = size
        self.timeout = timeout
        self.max_initial_ttl = num_passes
        self.cur_initial_ttl = init_num_passes
        self.buffers = [None] * size
        self.ttl = [0] * size
        self.idx = 0

    def get(self) -> Tuple[Any, bool]:
        if False:
            while True:
                i = 10
        'Get a new batch from the internal ring buffer.\n\n        Returns:\n           buf: Data item saved from inqueue.\n           released: True if the item is now removed from the ring buffer.\n        '
        if self.ttl[self.idx] <= 0:
            self.buffers[self.idx] = self.inqueue.get(timeout=self.timeout)
            self.ttl[self.idx] = self.cur_initial_ttl
            if self.cur_initial_ttl < self.max_initial_ttl:
                self.cur_initial_ttl += 1
        buf = self.buffers[self.idx]
        self.ttl[self.idx] -= 1
        released = self.ttl[self.idx] <= 0
        if released:
            self.buffers[self.idx] = None
        self.idx = (self.idx + 1) % len(self.buffers)
        return (buf, released)