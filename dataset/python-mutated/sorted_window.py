from __future__ import annotations
import bisect
import collections

class SortedWindow(collections.UserList):
    """Sorted running window data structure.

    Parameters
    ----------
    size
        Size of the window to compute the rolling quantile.

    Examples
    --------

    >>> from river import utils

    >>> window = utils.SortedWindow(size=3)

    >>> for i in reversed(range(9)):
    ...     print(window.append(i))
    [8]
    [7, 8]
    [6, 7, 8]
    [5, 6, 7]
    [4, 5, 6]
    [3, 4, 5]
    [2, 3, 4]
    [1, 2, 3]
    [0, 1, 2]

    References
    ----------
    [^1]: [Left sorted inserts in Python](https://stackoverflow.com/questions/8024571/insert-an-item-into-sorted-list-in-python)

    """

    def __init__(self, size: int):
        if False:
            while True:
                i = 10
        super().__init__()
        self.unsorted_window: collections.deque = collections.deque(maxlen=size)

    @property
    def size(self):
        if False:
            for i in range(10):
                print('nop')
        return self.unsorted_window.maxlen

    def append(self, x):
        if False:
            i = 10
            return i + 15
        if len(self) >= self.size:
            start_deque = bisect.bisect_left(self, self.unsorted_window[0])
            del self[start_deque]
        bisect.insort_left(self, x)
        self.unsorted_window.append(x)
        return self