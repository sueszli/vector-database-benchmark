from __future__ import annotations

def f(a: 1 + 2 == 3, b: list, c: this_cant_evaluate, d: 'Hello from inside a string') -> 'Return me!':
    if False:
        print('Hello World!')
    '\n    The absolute exact strings aren\'t reproducible according to the PEP,\n    so be careful to avoid being too specific\n    >>> stypes = (type(""), type(u"")) # Python 2 is a bit awkward here\n    >>> eval(f.__annotations__[\'a\'])\n    True\n    >>> isinstance(f.__annotations__[\'a\'], stypes)\n    True\n    >>> print(f.__annotations__[\'b\'])\n    list\n    >>> print(f.__annotations__[\'c\'])\n    this_cant_evaluate\n    >>> isinstance(eval(f.__annotations__[\'d\']), stypes)\n    True\n    >>> print(f.__annotations__[\'return\'][1:-1]) # First and last could be either " or \'\n    Return me!\n    >>> f.__annotations__[\'return\'][0] == f.__annotations__[\'return\'][-1]\n    True\n    '
    pass

def empty_decorator(cls):
    if False:
        for i in range(10):
            print('nop')
    return cls

@empty_decorator
class DecoratedStarship(object):
    """
    >>> sorted(DecoratedStarship.__annotations__.items())
    [('captain', 'str'), ('damage', 'cython.int')]
    """
    captain: str = 'Picard'
    damage: cython.int