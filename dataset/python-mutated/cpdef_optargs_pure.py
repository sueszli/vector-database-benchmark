import cython

class PyClass(object):
    a = 2

class PyClass99(object):
    a = 99

    def pymethod(self, x, y=1, z=PyClass):
        if False:
            i = 10
            return i + 15
        '\n        >>> obj = PyClass99()\n        >>> obj.pymethod(0)\n        (0, 1, 2)\n        '
        return (x, y, z.a)

def func(x, y=1, z=PyClass):
    if False:
        print('Hello World!')
    "\n    >>> func(0)\n    (0, 1, 2)\n    >>> func(0, 3)\n    (0, 3, 2)\n    >>> func(0, 3, PyClass)\n    (0, 3, 2)\n    >>> func(0, 3, 5)\n    Traceback (most recent call last):\n    AttributeError: 'int' object has no attribute 'a'\n    "
    return (x, y, z.a)

@cython.ccall
def pyfunc(x, y=1, z=PyClass):
    if False:
        while True:
            i = 10
    "\n    >>> pyfunc(0)\n    (0, 1, 2)\n    >>> pyfunc(0, 3)\n    (0, 3, 2)\n    >>> pyfunc(0, 3, PyClass)\n    (0, 3, 2)\n    >>> pyfunc(0, 3, 5)\n    Traceback (most recent call last):\n    AttributeError: 'int' object has no attribute 'a'\n    "
    return (x, y, z.a)