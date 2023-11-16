import functools
from .messenger import Messenger

class ReentrantMessenger(Messenger):

    def __init__(self):
        if False:
            while True:
                i = 10
        self._ref_count = 0
        super().__init__()

    def __call__(self, fn):
        if False:
            for i in range(10):
                print('nop')
        return functools.wraps(fn)(super().__call__(fn))

    def __enter__(self):
        if False:
            while True:
                i = 10
        self._ref_count += 1
        if self._ref_count == 1:
            super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            i = 10
            return i + 15
        self._ref_count -= 1
        if self._ref_count == 0:
            super().__exit__(exc_type, exc_value, traceback)