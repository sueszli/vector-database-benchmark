"""
Provides all sorts of iterator-related stuff.
"""
from typing import Iterable

def denote_last(iterable: Iterable):
    if False:
        while True:
            i = 10
    '\n    Similar to enumerate, this iterates over an iterable, and yields\n    tuples of item, is_last.\n    '
    iterator = iter(iterable)
    current = next(iterator)
    for future in iterator:
        yield (current, False)
        current = future
    yield (current, True)