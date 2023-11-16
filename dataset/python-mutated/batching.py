from itertools import islice

def get_batches_from_generator(iterable, n):
    if False:
        return 10
    '\n    Batch elements of an iterable into fixed-length chunks or blocks.\n    '
    it = iter(iterable)
    x = tuple(islice(it, n))
    while x:
        yield x
        x = tuple(islice(it, n))