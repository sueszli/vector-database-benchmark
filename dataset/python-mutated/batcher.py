from typing import Any, Callable, List

class Batcher:

    def __init__(self, batch_size: int, flush_handler: Callable[[List[Any]], None]):
        if False:
            i = 10
            return i + 15
        self.batch_size = batch_size
        self.buffer = []
        self.flush_handler = flush_handler

    def add(self, item: Any):
        if False:
            print('Hello World!')
        self.buffer.append(item)
        self._flush_if_necessary()

    def flush(self):
        if False:
            while True:
                i = 10
        if len(self.buffer) == 0:
            return
        self.flush_handler(list(self.buffer))
        self.buffer.clear()

    def _flush_if_necessary(self):
        if False:
            while True:
                i = 10
        if len(self.buffer) >= self.batch_size:
            self.flush()