"""
This module is designed to be used as follows::

    from past.builtins.noniterators import filter, map, range, reduce, zip

And then, for example::

    assert isinstance(range(5), list)

The list-producing functions this brings in are::

- ``filter``
- ``map``
- ``range``
- ``reduce``
- ``zip``

"""
from __future__ import division, absolute_import, print_function
from itertools import chain, starmap
import itertools
from past.types import basestring
from past.utils import PY3

def flatmap(f, items):
    if False:
        return 10
    return chain.from_iterable(map(f, items))
if PY3:
    import builtins

    def oldfilter(*args):
        if False:
            i = 10
            return i + 15
        '\n        filter(function or None, sequence) -> list, tuple, or string\n\n        Return those items of sequence for which function(item) is true.\n        If function is None, return the items that are true.  If sequence\n        is a tuple or string, return the same type, else return a list.\n        '
        mytype = type(args[1])
        if isinstance(args[1], basestring):
            return mytype().join(builtins.filter(*args))
        elif isinstance(args[1], (tuple, list)):
            return mytype(builtins.filter(*args))
        else:
            return list(builtins.filter(*args))

    def oldmap(func, *iterables):
        if False:
            i = 10
            return i + 15
        "\n        map(function, sequence[, sequence, ...]) -> list\n\n        Return a list of the results of applying the function to the\n        items of the argument sequence(s).  If more than one sequence is\n        given, the function is called with an argument list consisting of\n        the corresponding item of each sequence, substituting None for\n        missing values when not all sequences have the same length.  If\n        the function is None, return a list of the items of the sequence\n        (or a list of tuples if more than one sequence).\n\n        Test cases:\n        >>> oldmap(None, 'hello world')\n        ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']\n\n        >>> oldmap(None, range(4))\n        [0, 1, 2, 3]\n\n        More test cases are in test_past.test_builtins.\n        "
        zipped = itertools.zip_longest(*iterables)
        l = list(zipped)
        if len(l) == 0:
            return []
        if func is None:
            result = l
        else:
            result = list(starmap(func, l))
        try:
            if max([len(item) for item in result]) == 1:
                return list(chain.from_iterable(result))
        except TypeError as e:
            pass
        return result

    def oldrange(*args, **kwargs):
        if False:
            while True:
                i = 10
        return list(builtins.range(*args, **kwargs))

    def oldzip(*args, **kwargs):
        if False:
            while True:
                i = 10
        return list(builtins.zip(*args, **kwargs))
    filter = oldfilter
    map = oldmap
    range = oldrange
    from functools import reduce
    zip = oldzip
    __all__ = ['filter', 'map', 'range', 'reduce', 'zip']
else:
    import __builtin__
    filter = __builtin__.filter
    map = __builtin__.map
    range = __builtin__.range
    reduce = __builtin__.reduce
    zip = __builtin__.zip
    __all__ = []