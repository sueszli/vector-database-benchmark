def del_constant_start_stop(x):
    if False:
        i = 10
        return i + 15
    '\n    >>> l = [1,2,3,4]\n    >>> del_constant_start_stop(l)\n    [1, 2]\n\n    >>> l = [1,2,3,4,5,6,7]\n    >>> del_constant_start_stop(l)\n    [1, 2, 7]\n    '
    del x[2:6]
    return x

def del_start(x, start):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> l = [1,2,3,4]\n    >>> del_start(l, 2)\n    [1, 2]\n\n    >>> l = [1,2,3,4,5,6,7]\n    >>> del_start(l, 20)\n    [1, 2, 3, 4, 5, 6, 7]\n    >>> del_start(l, 8)\n    [1, 2, 3, 4, 5, 6, 7]\n    >>> del_start(l, 4)\n    [1, 2, 3, 4]\n\n    >>> del_start(l, -2)\n    [1, 2]\n    >>> l\n    [1, 2]\n    >>> del_start(l, -2)\n    []\n    >>> del_start(l, 2)\n    []\n    >>> del_start(l, -2)\n    []\n    >>> del_start(l, 20)\n    []\n\n    >>> del_start([1,2,3,4], -20)\n    []\n    >>> del_start([1,2,3,4], 0)\n    []\n    '
    del x[start:]
    return x

def del_stop(x, stop):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> l = [1,2,3,4]\n    >>> del_stop(l, 2)\n    [3, 4]\n\n    >>> l = [1,2,3,4,5,6,7]\n    >>> del_stop(l, -20)\n    [1, 2, 3, 4, 5, 6, 7]\n    >>> del_stop(l, -8)\n    [1, 2, 3, 4, 5, 6, 7]\n    >>> del_stop(l, -4)\n    [4, 5, 6, 7]\n\n    >>> del_stop(l, -2)\n    [6, 7]\n    >>> l\n    [6, 7]\n    >>> del_stop(l, -2)\n    [6, 7]\n    >>> del_stop(l, 2)\n    []\n    >>> del_stop(l, -2)\n    []\n    >>> del_stop(l, 20)\n    []\n\n    >>> del_stop([1,2,3,4], -20)\n    [1, 2, 3, 4]\n    >>> del_stop([1,2,3,4], 0)\n    [1, 2, 3, 4]\n    '
    del x[:stop]
    return x

def del_start_stop(x, start, stop):
    if False:
        while True:
            i = 10
    '\n    >>> l = [1,2,3,4]\n    >>> del_start_stop(l, 0, 2)\n    [3, 4]\n    >>> l\n    [3, 4]\n\n    >>> l = [1,2,3,4,5,6,7]\n    >>> del_start_stop(l, -1, -20)\n    [1, 2, 3, 4, 5, 6, 7]\n    >>> del_start_stop(l, -20, -8)\n    [1, 2, 3, 4, 5, 6, 7]\n    >>> del_start_stop(l, -6, -4)\n    [1, 4, 5, 6, 7]\n\n    >>> del_start_stop(l, -20, -2)\n    [6, 7]\n    >>> l\n    [6, 7]\n    >>> del_start_stop(l, -2, 1)\n    [7]\n    >>> del_start_stop(l, -2, 3)\n    []\n    >>> del_start_stop(l, 2, 4)\n    []\n\n    >>> del_start_stop([1,2,3,4], 20, -20)\n    [1, 2, 3, 4]\n    >>> del_start_stop([1,2,3,4], 0, 0)\n    [1, 2, 3, 4]\n    '
    del x[start:stop]
    return x