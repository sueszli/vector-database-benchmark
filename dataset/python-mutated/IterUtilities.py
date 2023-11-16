from itertools import tee

def partition(iterable, predicate):
    if False:
        return 10
    '\n    Partitions the iterable into two iterables based on the given predicate.\n\n    :param predicate:   A function that takes an item of the iterable and\n                        returns a boolean\n    :return:            Two iterators pointing to the original iterable\n    '
    (a, b) = tee(((predicate(item), item) for item in iterable))
    return ((item for (pred, item) in a if pred), (item for (pred, item) in b if not pred))