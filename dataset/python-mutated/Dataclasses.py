from collections import namedtuple
try:
    from types import MappingProxyType
except ImportError:
    MappingProxyType = lambda x: x

class _MISSING_TYPE(object):
    pass
MISSING = _MISSING_TYPE()
_DataclassParams = namedtuple('_DataclassParams', ['init', 'repr', 'eq', 'order', 'unsafe_hash', 'frozen', 'match_args', 'kw_only', 'slots', 'weakref_slot'])

class Field(object):
    __slots__ = ('name', 'type', 'default', 'default_factory', 'repr', 'hash', 'init', 'compare', 'metadata', 'kw_only', '_field_type')

    def __init__(self, default, default_factory, init, repr, hash, compare, metadata, kw_only):
        if False:
            i = 10
            return i + 15
        self.name = None
        self.type = None
        self.default = default
        self.default_factory = default_factory
        self.init = init
        self.repr = repr
        self.hash = hash
        self.compare = compare
        self.metadata = MappingProxyType({}) if metadata is None else MappingProxyType(metadata)
        self.kw_only = kw_only
        self._field_type = None

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'Field(name={0!r},type={1!r},default={2!r},default_factory={3!r},init={4!r},repr={5!r},hash={6!r},compare={7!r},metadata={8!r},kwonly={9!r},)'.format(self.name, self.type, self.default, self.default_factory, self.init, self.repr, self.hash, self.compare, self.metadata, self.kw_only)

class _HAS_DEFAULT_FACTORY_CLASS:

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<factory>'
_HAS_DEFAULT_FACTORY = _HAS_DEFAULT_FACTORY_CLASS()

def dataclass(*args, **kwds):
    if False:
        print('Hello World!')
    raise NotImplementedError("Standard library 'dataclasses' moduleis unavailable, likely due to the version of Python you're using.")

class _FIELD_BASE:

    def __init__(self, name):
        if False:
            while True:
                i = 10
        self.name = name

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.name
_FIELD = _FIELD_BASE('_FIELD')
_FIELD_CLASSVAR = _FIELD_BASE('_FIELD_CLASSVAR')
_FIELD_INITVAR = _FIELD_BASE('_FIELD_INITVAR')

def field(*ignore, **kwds):
    if False:
        i = 10
        return i + 15
    default = kwds.pop('default', MISSING)
    default_factory = kwds.pop('default_factory', MISSING)
    init = kwds.pop('init', True)
    repr = kwds.pop('repr', True)
    hash = kwds.pop('hash', None)
    compare = kwds.pop('compare', True)
    metadata = kwds.pop('metadata', None)
    kw_only = kwds.pop('kw_only', None)
    if kwds:
        raise ValueError('field received unexpected keyword arguments: %s' % list(kwds.keys()))
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError('cannot specify both default and default_factory')
    if ignore:
        raise ValueError("'field' does not take any positional arguments")
    return Field(default, default_factory, init, repr, hash, compare, metadata, kw_only)