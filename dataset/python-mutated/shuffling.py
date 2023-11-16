from __future__ import annotations
import itertools
import random
import types
import typing

def shuffle(stream: typing.Iterator, buffer_size: int, seed: int | None=None):
    if False:
        return 10
    'Shuffles a stream of data.\n\n    This works by maintaining a buffer of elements. The first `buffer_size` elements are stored in\n    memory. Once the buffer is full, a random element inside the buffer is yielded. Every time an\n    element is yielded, the next element in the stream replaces it and the buffer is sampled again.\n    Increasing `buffer_size` will improve the quality of the shuffling.\n\n    If you really want to stream over your dataset in a "good" random order, the best way is to\n    split your dataset into smaller datasets and loop over them in a round-robin fashion. You may\n    do this by using the `roundrobin` recipe from the `itertools` module.\n\n    Parameters\n    ----------\n    stream\n        The stream to shuffle.\n    buffer_size\n        The size of the buffer which contains the elements help in memory. Increasing this will\n        increase randomness but will incur more memory usage.\n    seed\n        Random seed used for sampling.\n\n    Examples\n    --------\n\n    >>> from river import stream\n\n    >>> for i in stream.shuffle(range(15), buffer_size=5, seed=42):\n    ...     print(i)\n    0\n    5\n    2\n    1\n    8\n    9\n    6\n    4\n    11\n    12\n    10\n    7\n    14\n    13\n    3\n\n    References\n    ----------\n    [^1]: [Visualizing TensorFlow\'s streaming shufflers](http://www.moderndescartes.com/essays/shuffle_viz/)\n\n    '
    rng = random.Random(seed)
    if not isinstance(stream, types.GeneratorType):
        stream = iter(stream)
    buffer = list(itertools.islice(stream, buffer_size))
    for element in stream:
        i = rng.randint(0, len(buffer) - 1)
        yield buffer[i]
        buffer[i] = element
    rng.shuffle(buffer)
    yield from buffer