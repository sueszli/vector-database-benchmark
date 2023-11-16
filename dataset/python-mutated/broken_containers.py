try:
    from collections.abc import Sequence, Mapping
except ImportError:
    from collections import Sequence, Mapping
__all__ = ['BROKEN_ITERABLE', 'BROKEN_SEQUENCE', 'BROKEN_MAPPING']

class BrokenIterable:

    def __iter__(self):
        if False:
            print('Hello World!')
        yield 'x'
        raise ValueError(type(self).__name__)

    def __getitem__(self, item):
        if False:
            while True:
                i = 10
        return item

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return 2

class BrokenSequence(BrokenIterable, Sequence):
    pass

class BrokenMapping(BrokenIterable, Mapping):
    pass
BROKEN_ITERABLE = BrokenIterable()
BROKEN_SEQUENCE = BrokenSequence()
BROKEN_MAPPING = BrokenMapping()