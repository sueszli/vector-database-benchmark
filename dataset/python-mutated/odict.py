"""
    :codeauthor: Pedro Algarvio (pedro@algarvio.me)


    salt.utils.odict
    ~~~~~~~~~~~~~~~~

    This is a compatibility/"importability" layer for an ordered dictionary.
    Tries to import from the standard library if python >= 2.7, then from the
    ``ordereddict`` package available from PyPi, and, as a last resort,
    provides an ``OrderedDict`` implementation based on::

        http://code.activestate.com/recipes/576669/

    It also implements a DefaultOrderedDict Class that serves  as a
    combination of ``OrderedDict`` and ``defaultdict``
    It's source was submitted here::

        http://stackoverflow.com/questions/6190331/
"""
from collections.abc import Callable
try:
    import collections

    class OrderedDict(collections.OrderedDict):
        __hash__ = None
except (ImportError, AttributeError):
    try:
        import ordereddict

        class OrderedDict(ordereddict.OrderedDict):
            __hash_ = None
    except ImportError:
        try:
            from _thread import get_ident as _get_ident
        except ImportError:
            from _dummy_thread import get_ident as _get_ident

        class OrderedDict(dict):
            """Dictionary that remembers insertion order"""
            __hash_ = None

            def __init__(self, *args, **kwds):
                if False:
                    while True:
                        i = 10
                'Initialize an ordered dictionary.  Signature is the same as for\n                regular dictionaries, but keyword arguments are not recommended\n                because their insertion order is arbitrary.\n\n                '
                super().__init__()
                if len(args) > 1:
                    raise TypeError(f'expected at most 1 arguments, got {len(args)}')
                try:
                    self.__root
                except AttributeError:
                    self.__root = root = []
                    root[:] = [root, root, None]
                    self.__map = {}
                self.__update(*args, **kwds)

            def __setitem__(self, key, value, dict_setitem=dict.__setitem__):
                if False:
                    while True:
                        i = 10
                'od.__setitem__(i, y) <==> od[i]=y'
                if key not in self:
                    root = self.__root
                    last = root[0]
                    last[1] = root[0] = self.__map[key] = [last, root, key]
                dict_setitem(self, key, value)

            def __delitem__(self, key, dict_delitem=dict.__delitem__):
                if False:
                    for i in range(10):
                        print('nop')
                'od.__delitem__(y) <==> del od[y]'
                dict_delitem(self, key)
                (link_prev, link_next, key) = self.__map.pop(key)
                link_prev[1] = link_next
                link_next[0] = link_prev

            def __iter__(self):
                if False:
                    for i in range(10):
                        print('nop')
                'od.__iter__() <==> iter(od)'
                root = self.__root
                curr = root[1]
                while curr is not root:
                    yield curr[2]
                    curr = curr[1]

            def __reversed__(self):
                if False:
                    while True:
                        i = 10
                'od.__reversed__() <==> reversed(od)'
                root = self.__root
                curr = root[0]
                while curr is not root:
                    yield curr[2]
                    curr = curr[0]

            def clear(self):
                if False:
                    return 10
                'od.clear() -> None.  Remove all items from od.'
                try:
                    for node in self.__map.values():
                        del node[:]
                    root = self.__root
                    root[:] = [root, root, None]
                    self.__map.clear()
                except AttributeError:
                    pass
                dict.clear(self)

            def popitem(self, last=True):
                if False:
                    for i in range(10):
                        print('nop')
                'od.popitem() -> (k, v), return and remove a (key, value) pair.\n                Pairs are returned in LIFO order if last is true or FIFO order if false.\n\n                '
                if not self:
                    raise KeyError('dictionary is empty')
                root = self.__root
                if last:
                    link = root[0]
                    link_prev = link[0]
                    link_prev[1] = root
                    root[0] = link_prev
                else:
                    link = root[1]
                    link_next = link[1]
                    root[1] = link_next
                    link_next[0] = root
                key = link[2]
                del self.__map[key]
                value = dict.pop(self, key)
                return (key, value)

            def keys(self):
                if False:
                    while True:
                        i = 10
                'od.keys() -> list of keys in od'
                return list(self)

            def values(self):
                if False:
                    print('Hello World!')
                'od.values() -> list of values in od'
                return [self[key] for key in self]

            def items(self):
                if False:
                    for i in range(10):
                        print('nop')
                'od.items() -> list of (key, value) pairs in od'
                return [(key, self[key]) for key in self]

            def iterkeys(self):
                if False:
                    for i in range(10):
                        print('nop')
                'od.iterkeys() -> an iterator over the keys in od'
                return iter(self)

            def itervalues(self):
                if False:
                    i = 10
                    return i + 15
                'od.itervalues -> an iterator over the values in od'
                for k in self:
                    yield self[k]

            def iteritems(self):
                if False:
                    return 10
                'od.iteritems -> an iterator over the (key, value) items in od'
                for k in self:
                    yield (k, self[k])

            def update(*args, **kwds):
                if False:
                    return 10
                'od.update(E, **F) -> None.  Update od from dict/iterable E and F.\n\n                If E is a dict instance, does:           for k in E: od[k] = E[k]\n                If E has a .keys() method, does:         for k in E.keys(): od[k] = E[k]\n                Or if E is an iterable of items, does:   for k, v in E: od[k] = v\n                In either case, this is followed by:     for k, v in F.items(): od[k] = v\n\n                '
                if len(args) > 2:
                    raise TypeError('update() takes at most 2 positional arguments ({} given)'.format(len(args)))
                elif not args:
                    raise TypeError('update() takes at least 1 argument (0 given)')
                self = args[0]
                other = ()
                if len(args) == 2:
                    other = args[1]
                if isinstance(other, dict):
                    for key in other:
                        self[key] = other[key]
                elif hasattr(other, 'keys'):
                    for key in other:
                        self[key] = other[key]
                else:
                    for (key, value) in other:
                        self[key] = value
                for (key, value) in kwds.items():
                    self[key] = value
            __update = update
            __marker = object()

            def pop(self, key, default=__marker):
                if False:
                    print('Hello World!')
                'od.pop(k[,d]) -> v, remove specified key and return the corresponding value.\n                If key is not found, d is returned if given, otherwise KeyError is raised.\n\n                '
                if key in self:
                    result = self[key]
                    del self[key]
                    return result
                if default is self.__marker:
                    raise KeyError(key)
                return default

            def setdefault(self, key, default=None):
                if False:
                    return 10
                'od.setdefault(k[,d]) -> od.get(k,d), also set od[k]=d if k not in od'
                if key in self:
                    return self[key]
                self[key] = default
                return default

            def __repr__(self, _repr_running={}):
                if False:
                    while True:
                        i = 10
                'od.__repr__() <==> repr(od)'
                call_key = (id(self), _get_ident())
                if call_key in _repr_running:
                    return '...'
                _repr_running[call_key] = 1
                try:
                    if not self:
                        return f'{self.__class__.__name__}()'
                    return "{}('{}')".format(self.__class__.__name__, list(self.items()))
                finally:
                    del _repr_running[call_key]

            def __reduce__(self):
                if False:
                    while True:
                        i = 10
                'Return state information for pickling'
                items = [[k, self[k]] for k in self]
                inst_dict = vars(self).copy()
                for k in vars(OrderedDict()):
                    inst_dict.pop(k, None)
                if inst_dict:
                    return (self.__class__, (items,), inst_dict)
                return (self.__class__, (items,))

            def copy(self):
                if False:
                    i = 10
                    return i + 15
                'od.copy() -> a shallow copy of od'
                return self.__class__(self)

            @classmethod
            def fromkeys(cls, iterable, value=None):
                if False:
                    while True:
                        i = 10
                'OD.fromkeys(S[, v]) -> New ordered dictionary with keys from S\n                and values equal to v (which defaults to None).\n\n                '
                d = cls()
                for key in iterable:
                    d[key] = value
                return d

            def __eq__(self, other):
                if False:
                    while True:
                        i = 10
                'od.__eq__(y) <==> od==y.  Comparison to another OD is order-sensitive\n                while comparison to a regular mapping is order-insensitive.\n\n                '
                if isinstance(other, OrderedDict):
                    return len(self) == len(other) and self.items() == other.items()
                return dict.__eq__(self, other)

            def __ne__(self, other):
                if False:
                    print('Hello World!')
                return not self == other

class DefaultOrderedDict(OrderedDict):
    """
    Dictionary that remembers insertion order
    """

    def __init__(self, default_factory=None, *a, **kw):
        if False:
            print('Hello World!')
        if default_factory is not None and (not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        super().__init__(*a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if False:
            print('Hello World!')
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
        if self.default_factory is None:
            args = tuple()
        else:
            args = (self.default_factory,)
        return (type(self), args, None, None, self.items())

    def copy(self):
        if False:
            i = 10
            return i + 15
        return self.__copy__()

    def __copy__(self):
        if False:
            print('Hello World!')
        return type(self)(self.default_factory, self)

    def __deepcopy__(self):
        if False:
            while True:
                i = 10
        import copy
        return type(self)(self.default_factory, copy.deepcopy(self.items()))

    def __repr__(self, _repr_running={}):
        if False:
            for i in range(10):
                print('nop')
        return 'DefaultOrderedDict({}, {})'.format(self.default_factory, super().__repr__())