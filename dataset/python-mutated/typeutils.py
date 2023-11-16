"""Python's built-in :mod:`functools` module builds several useful
utilities on top of Python's first-class function support.
``typeutils`` attempts to do the same for metaprogramming with types
and instances.
"""
import sys
from collections import deque
_issubclass = issubclass

def make_sentinel(name='_MISSING', var_name=None):
    if False:
        print('Hello World!')
    'Creates and returns a new **instance** of a new class, suitable for\n    usage as a "sentinel", a kind of singleton often used to indicate\n    a value is missing when ``None`` is a valid input.\n\n    Args:\n        name (str): Name of the Sentinel\n        var_name (str): Set this name to the name of the variable in\n            its respective module enable pickleability. Note:\n            pickleable sentinels should be global constants at the top\n            level of their module.\n\n    >>> make_sentinel(var_name=\'_MISSING\')\n    _MISSING\n\n    The most common use cases here in boltons are as default values\n    for optional function arguments, partly because of its\n    less-confusing appearance in automatically generated\n    documentation. Sentinels also function well as placeholders in queues\n    and linked lists.\n\n    .. note::\n\n      By design, additional calls to ``make_sentinel`` with the same\n      values will not produce equivalent objects.\n\n      >>> make_sentinel(\'TEST\') == make_sentinel(\'TEST\')\n      False\n      >>> type(make_sentinel(\'TEST\')) == type(make_sentinel(\'TEST\'))\n      False\n\n    '

    class Sentinel(object):

        def __init__(self):
            if False:
                return 10
            self.name = name
            self.var_name = var_name

        def __repr__(self):
            if False:
                return 10
            if self.var_name:
                return self.var_name
            return '%s(%r)' % (self.__class__.__name__, self.name)
        if var_name:

            def __reduce__(self):
                if False:
                    print('Hello World!')
                return self.var_name

        def __nonzero__(self):
            if False:
                print('Hello World!')
            return False
        __bool__ = __nonzero__
    if var_name:
        frame = sys._getframe(1)
        module = frame.f_globals.get('__name__')
        if not module or module not in sys.modules:
            raise ValueError('Pickleable sentinel objects (with var_name) can only be created from top-level module scopes')
        Sentinel.__module__ = module
    return Sentinel()

def issubclass(subclass, baseclass):
    if False:
        i = 10
        return i + 15
    "Just like the built-in :func:`issubclass`, this function checks\n    whether *subclass* is inherited from *baseclass*. Unlike the\n    built-in function, this ``issubclass`` will simply return\n    ``False`` if either argument is not suitable (e.g., if *subclass*\n    is not an instance of :class:`type`), instead of raising\n    :exc:`TypeError`.\n\n    Args:\n        subclass (type): The target class to check.\n        baseclass (type): The base class *subclass* will be checked against.\n\n    >>> class MyObject(object): pass\n    ...\n    >>> issubclass(MyObject, object)  # always a fun fact\n    True\n    >>> issubclass('hi', 'friend')\n    False\n    "
    try:
        return _issubclass(subclass, baseclass)
    except TypeError:
        return False

def get_all_subclasses(cls):
    if False:
        while True:
            i = 10
    "Recursively finds and returns a :class:`list` of all types\n    inherited from *cls*.\n\n    >>> class A(object):\n    ...     pass\n    ...\n    >>> class B(A):\n    ...     pass\n    ...\n    >>> class C(B):\n    ...     pass\n    ...\n    >>> class D(A):\n    ...     pass\n    ...\n    >>> [t.__name__ for t in get_all_subclasses(A)]\n    ['B', 'D', 'C']\n    >>> [t.__name__ for t in get_all_subclasses(B)]\n    ['C']\n\n    "
    try:
        to_check = deque(cls.__subclasses__())
    except (AttributeError, TypeError):
        raise TypeError('expected type object, not %r' % cls)
    (seen, ret) = (set(), [])
    while to_check:
        cur = to_check.popleft()
        if cur in seen:
            continue
        ret.append(cur)
        seen.add(cur)
        to_check.extend(cur.__subclasses__())
    return ret

class classproperty(object):
    """Much like a :class:`property`, but the wrapped get function is a
    class method.  For simplicity, only read-only properties are
    implemented.
    """

    def __init__(self, fn):
        if False:
            print('Hello World!')
        self.fn = fn

    def __get__(self, instance, cls):
        if False:
            return 10
        return self.fn(cls)