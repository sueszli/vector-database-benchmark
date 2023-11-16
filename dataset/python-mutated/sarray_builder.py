"""
An interface for creating an SArray over time.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .._cython.cy_sarray_builder import UnitySArrayBuilderProxy
from .sarray import SArray

class SArrayBuilder(object):
    """
    An interface to incrementally build an SArray element by element.

    Once closed, the SArray cannot be "reopened" using this interface.

    Parameters
    ----------
    dtype : type
        The type of the elements in the SArray.

    num_segments : int, optional
        Number of segments that can be written in parallel.

    history_size : int, optional
        The number of elements to be cached as history. Caches the last
        `history_size` elements added with `append` or `append_multiple`.

    Returns
    -------
    out : SArrayBuilder

    Examples
    --------
    >>> from turicreate import SArrayBuilder

    >>> sb = SArrayBuilder(int)

    >>> sb.append(1)

    >>> sb.append_multiple([2,3])

    >>> sb.close()
    dtype: int
    Rows: 3
    [1, 2, 3]
    """

    def __init__(self, dtype, num_segments=1, history_size=10):
        if False:
            while True:
                i = 10
        self._builder = UnitySArrayBuilderProxy()
        self._builder.init(num_segments, history_size, dtype)
        self._block_size = 1024

    def append(self, data, segment=0):
        if False:
            i = 10
            return i + 15
        '\n        Append a single element to an SArray.\n\n        Throws a RuntimeError if the type of `data` is incompatible with\n        the type of the SArray.\n\n        Parameters\n        ----------\n        data  : any SArray-supported type\n            A data element to add to the SArray.\n\n        segment : int\n            The segment to write this element. Each segment is numbered\n            sequentially, starting with 0. Any value in segment 1 will be after\n            any value in segment 0, and the order of elements in each segment is\n            preserved as they are added.\n        '
        self._builder.append(data, segment)

    def append_multiple(self, data, segment=0):
        if False:
            print('Hello World!')
        '\n        Append multiple elements to an SArray.\n\n        Throws a RuntimeError if the type of `data` is incompatible with\n        the type of the SArray.\n\n        Parameters\n        ----------\n        data  : any SArray-supported type\n            A data element to add to the SArray.\n\n        segment : int\n            The segment to write this element. Each segment is numbered\n            sequentially, starting with 0. Any value in segment 1 will be after\n            any value in segment 0, and the order of elements in each segment is\n            preserved as they are added.\n        '
        if not hasattr(data, '__iter__'):
            raise TypeError('append_multiple must be passed an iterable object')
        tmp_list = []
        for i in data:
            tmp_list.append(i)
            if len(tmp_list) >= self._block_size:
                self._builder.append_multiple(tmp_list, segment)
                tmp_list = []
        if len(tmp_list) > 0:
            self._builder.append_multiple(tmp_list, segment)

    def get_type(self):
        if False:
            print('Hello World!')
        '\n        The type the result SArray will be if `close` is called.\n        '
        return self._builder.get_type()

    def read_history(self, num=10, segment=0):
        if False:
            print('Hello World!')
        '\n        Outputs the last `num` elements that were appended either by `append` or\n        `append_multiple`.\n\n        Returns\n        -------\n        out : list\n\n        '
        if num < 0:
            num = 0
        if segment < 0:
            raise TypeError('segment must be >= 0')
        return self._builder.read_history(num, segment)

    def close(self):
        if False:
            return 10
        '\n        Creates an SArray from all values that were appended to the\n        SArrayBuilder. No function that appends data may be called after this\n        is called.\n\n        Returns\n        -------\n        out : SArray\n\n        '
        return SArray(_proxy=self._builder.close())