try:
    from heapq import *
except ImportError:
    pass

def f(a):
    if False:
        i = 10
        return i + 15
    "\n    Py<=3.3 gives 'global name ...', Py3.4+ only 'name ...'\n\n    >>> f(1)   # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    NameError: ...name 'definitely_unknown_name' is not defined\n    "
    a = f
    a = definitely_unknown_name