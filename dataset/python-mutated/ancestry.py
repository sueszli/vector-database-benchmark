"""
Routines for obtaining the class names
of an object and its parent classes.
"""
from more_itertools import unique_everseen

def all_bases(c):
    if False:
        i = 10
        return i + 15
    '\n    return a tuple of all base classes the class c has as a parent.\n    >>> object in all_bases(list)\n    True\n    '
    return c.mro()[1:]

def all_classes(c):
    if False:
        print('Hello World!')
    '\n    return a tuple of all classes to which c belongs\n    >>> list in all_classes(list)\n    True\n    '
    return c.mro()

def iter_subclasses(cls):
    if False:
        while True:
            i = 10
    "\n    Generator over all subclasses of a given class, in depth-first order.\n\n    >>> bool in list(iter_subclasses(int))\n    True\n    >>> class A(object): pass\n    >>> class B(A): pass\n    >>> class C(A): pass\n    >>> class D(B,C): pass\n    >>> class E(D): pass\n    >>>\n    >>> for cls in iter_subclasses(A):\n    ...     print(cls.__name__)\n    B\n    D\n    E\n    C\n    >>> # get ALL classes currently defined\n    >>> res = [cls.__name__ for cls in iter_subclasses(object)]\n    >>> 'type' in res\n    True\n    >>> 'tuple' in res\n    True\n    >>> len(res) > 100\n    True\n    "
    return unique_everseen(_iter_all_subclasses(cls))

def _iter_all_subclasses(cls):
    if False:
        while True:
            i = 10
    try:
        subs = cls.__subclasses__()
    except TypeError:
        subs = cls.__subclasses__(cls)
    for sub in subs:
        yield sub
        yield from iter_subclasses(sub)