"""
Construction of sentinel objects.

Sentinel objects are used when you only care to check for object identity.
"""
import sys
from textwrap import dedent

class _Sentinel(object):
    """Base class for Sentinel objects.
    """
    __slots__ = ('__weakref__',)

def is_sentinel(obj):
    if False:
        for i in range(10):
            print('nop')
    return isinstance(obj, _Sentinel)

def sentinel(name, doc=None):
    if False:
        print('Hello World!')
    try:
        value = sentinel._cache[name]
    except KeyError:
        pass
    else:
        if doc == value.__doc__:
            return value
        raise ValueError(dedent('            New sentinel value %r conflicts with an existing sentinel of the\n            same name.\n            Old sentinel docstring: %r\n            New sentinel docstring: %r\n\n            The old sentinel was created at: %s\n\n            Resolve this conflict by changing the name of one of the sentinels.\n            ') % (name, value.__doc__, doc, value._created_at))
    try:
        frame = sys._getframe(1)
    except ValueError:
        frame = None
    if frame is None:
        created_at = '<unknown>'
    else:
        created_at = '%s:%s' % (frame.f_code.co_filename, frame.f_lineno)

    @object.__new__
    class Sentinel(_Sentinel):
        __doc__ = doc
        __name__ = name
        _created_at = created_at

        def __new__(cls):
            if False:
                i = 10
                return i + 15
            raise TypeError('cannot create %r instances' % name)

        def __repr__(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'sentinel(%r)' % name

        def __reduce__(self):
            if False:
                for i in range(10):
                    print('nop')
            return (sentinel, (name, doc))

        def __deepcopy__(self, _memo):
            if False:
                for i in range(10):
                    print('nop')
            return self

        def __copy__(self):
            if False:
                i = 10
                return i + 15
            return self
    cls = type(Sentinel)
    try:
        cls.__module__ = frame.f_globals['__name__']
    except (AttributeError, KeyError):
        cls.__module__ = None
    sentinel._cache[name] = Sentinel
    return Sentinel
sentinel._cache = {}