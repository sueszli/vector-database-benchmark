"""
Implementation of the iterator pattern using the iterator protocol from Python

*TL;DR
Traverses a container and accesses the container's elements.
"""
from __future__ import annotations

class NumberWords:
    """Counts by word numbers, up to a maximum of five"""
    _WORD_MAP = ('one', 'two', 'three', 'four', 'five')

    def __init__(self, start: int, stop: int) -> None:
        if False:
            while True:
                i = 10
        self.start = start
        self.stop = stop

    def __iter__(self) -> NumberWords:
        if False:
            for i in range(10):
                print('nop')
        return self

    def __next__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        if self.start > self.stop or self.start > len(self._WORD_MAP):
            raise StopIteration
        current = self.start
        self.start += 1
        return self._WORD_MAP[current - 1]

def main():
    if False:
        for i in range(10):
            print('nop')
    '\n    # Counting to two...\n    >>> for number in NumberWords(start=1, stop=2):\n    ...     print(number)\n    one\n    two\n\n    # Counting to five...\n    >>> for number in NumberWords(start=1, stop=5):\n    ...     print(number)\n    one\n    two\n    three\n    four\n    five\n    '
if __name__ == '__main__':
    import doctest
    doctest.testmod()