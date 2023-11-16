class Sliceable(object):
    """
    >>> sl = Sliceable()

    >>> sl[1:2]
    (1, 2, None)
    >>> py_slice2(sl, 1, 2)
    (1, 2, None)

    >>> sl[1:None]
    (1, None, None)
    >>> py_slice2(sl, 1, None)
    (1, None, None)

    >>> sl[None:2]
    (None, 2, None)
    >>> py_slice2(sl, None, 2)
    (None, 2, None)

    >>> sl[None:None]
    (None, None, None)
    >>> py_slice2(sl, None, None)
    (None, None, None)
    """

    def __getitem__(self, sl):
        if False:
            i = 10
            return i + 15
        return (sl.start, sl.stop, sl.step)

def py_slice2(obj, a, b):
    if False:
        while True:
            i = 10
    '\n    >>> [1,2,3][1:2]\n    [2]\n    >>> py_slice2([1,2,3], 1, 2)\n    [2]\n\n    >>> [1,2,3][None:2]\n    [1, 2]\n    >>> py_slice2([1,2,3], None, 2)\n    [1, 2]\n\n    >>> [1,2,3][None:None]\n    [1, 2, 3]\n    >>> py_slice2([1,2,3], None, None)\n    [1, 2, 3]\n    '
    return obj[a:b]