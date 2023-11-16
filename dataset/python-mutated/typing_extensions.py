import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
__all__ = ['Any', 'ClassVar', 'Concatenate', 'Final', 'LiteralString', 'ParamSpec', 'ParamSpecArgs', 'ParamSpecKwargs', 'Self', 'Type', 'TypeVar', 'TypeVarTuple', 'Unpack', 'Awaitable', 'AsyncIterator', 'AsyncIterable', 'Coroutine', 'AsyncGenerator', 'AsyncContextManager', 'Buffer', 'ChainMap', 'ContextManager', 'Counter', 'Deque', 'DefaultDict', 'NamedTuple', 'OrderedDict', 'TypedDict', 'SupportsAbs', 'SupportsBytes', 'SupportsComplex', 'SupportsFloat', 'SupportsIndex', 'SupportsInt', 'SupportsRound', 'Annotated', 'assert_never', 'assert_type', 'clear_overloads', 'dataclass_transform', 'deprecated', 'get_overloads', 'final', 'get_args', 'get_origin', 'get_original_bases', 'get_protocol_members', 'get_type_hints', 'IntVar', 'is_protocol', 'is_typeddict', 'Literal', 'NewType', 'overload', 'override', 'Protocol', 'reveal_type', 'runtime', 'runtime_checkable', 'Text', 'TypeAlias', 'TypeAliasType', 'TypeGuard', 'TYPE_CHECKING', 'Never', 'NoReturn', 'Required', 'NotRequired', 'AbstractSet', 'AnyStr', 'BinaryIO', 'Callable', 'Collection', 'Container', 'Dict', 'ForwardRef', 'FrozenSet', 'Generator', 'Generic', 'Hashable', 'IO', 'ItemsView', 'Iterable', 'Iterator', 'KeysView', 'List', 'Mapping', 'MappingView', 'Match', 'MutableMapping', 'MutableSequence', 'MutableSet', 'Optional', 'Pattern', 'Reversible', 'Sequence', 'Set', 'Sized', 'TextIO', 'Tuple', 'Union', 'ValuesView', 'cast', 'no_type_check', 'no_type_check_decorator']
PEP_560 = True
GenericMeta = type

class _Sentinel:

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<sentinel>'
_marker = _Sentinel()

def _check_generic(cls, parameters, elen=_marker):
    if False:
        return 10
    'Check correct count for parameters of a generic cls (internal helper).\n    This gives a nice error message in case of count mismatch.\n    '
    if not elen:
        raise TypeError(f'{cls} is not a generic class')
    if elen is _marker:
        if not hasattr(cls, '__parameters__') or not cls.__parameters__:
            raise TypeError(f'{cls} is not a generic class')
        elen = len(cls.__parameters__)
    alen = len(parameters)
    if alen != elen:
        if hasattr(cls, '__parameters__'):
            parameters = [p for p in cls.__parameters__ if not _is_unpack(p)]
            num_tv_tuples = sum((isinstance(p, TypeVarTuple) for p in parameters))
            if num_tv_tuples > 0 and alen >= elen - num_tv_tuples:
                return
        raise TypeError(f"Too {('many' if alen > elen else 'few')} parameters for {cls}; actual {alen}, expected {elen}")
if sys.version_info >= (3, 10):

    def _should_collect_from_parameters(t):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(t, (typing._GenericAlias, _types.GenericAlias, _types.UnionType))
elif sys.version_info >= (3, 9):

    def _should_collect_from_parameters(t):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(t, (typing._GenericAlias, _types.GenericAlias))
else:

    def _should_collect_from_parameters(t):
        if False:
            while True:
                i = 10
        return isinstance(t, typing._GenericAlias) and (not t._special)

def _collect_type_vars(types, typevar_types=None):
    if False:
        for i in range(10):
            print('nop')
    'Collect all type variable contained in types in order of\n    first appearance (lexicographic order). For example::\n\n        _collect_type_vars((T, List[S, T])) == (T, S)\n    '
    if typevar_types is None:
        typevar_types = typing.TypeVar
    tvars = []
    for t in types:
        if isinstance(t, typevar_types) and t not in tvars and (not _is_unpack(t)):
            tvars.append(t)
        if _should_collect_from_parameters(t):
            tvars.extend([t for t in t.__parameters__ if t not in tvars])
    return tuple(tvars)
NoReturn = typing.NoReturn
T = typing.TypeVar('T')
KT = typing.TypeVar('KT')
VT = typing.TypeVar('VT')
T_co = typing.TypeVar('T_co', covariant=True)
T_contra = typing.TypeVar('T_contra', contravariant=True)
if sys.version_info >= (3, 11):
    from typing import Any
else:

    class _AnyMeta(type):

        def __instancecheck__(self, obj):
            if False:
                for i in range(10):
                    print('nop')
            if self is Any:
                raise TypeError('typing_extensions.Any cannot be used with isinstance()')
            return super().__instancecheck__(obj)

        def __repr__(self):
            if False:
                i = 10
                return i + 15
            if self is Any:
                return 'typing_extensions.Any'
            return super().__repr__()

    class Any(metaclass=_AnyMeta):
        """Special type indicating an unconstrained type.
        - Any is compatible with every type.
        - Any assumed to have all methods.
        - All values assumed to be instances of Any.
        Note that all the above statements are true from the point of view of
        static type checkers. At runtime, Any should not be used with instance
        checks.
        """

        def __new__(cls, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            if cls is Any:
                raise TypeError('Any cannot be instantiated')
            return super().__new__(cls, *args, **kwargs)
ClassVar = typing.ClassVar

class _ExtensionsSpecialForm(typing._SpecialForm, _root=True):

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'typing_extensions.' + self._name
if hasattr(typing, 'Final') and sys.version_info[:2] >= (3, 7):
    Final = typing.Final
else:

    class _FinalForm(_ExtensionsSpecialForm, _root=True):

        def __getitem__(self, parameters):
            if False:
                i = 10
                return i + 15
            item = typing._type_check(parameters, f'{self._name} accepts only a single type.')
            return typing._GenericAlias(self, (item,))
    Final = _FinalForm('Final', doc='A special typing construct to indicate that a name\n                       cannot be re-assigned or overridden in a subclass.\n                       For example:\n\n                           MAX_SIZE: Final = 9000\n                           MAX_SIZE += 1  # Error reported by type checker\n\n                           class Connection:\n                               TIMEOUT: Final[int] = 10\n                           class FastConnector(Connection):\n                               TIMEOUT = 1  # Error reported by type checker\n\n                       There is no runtime checking of these properties.')
if sys.version_info >= (3, 11):
    final = typing.final
else:

    def final(f):
        if False:
            i = 10
            return i + 15
        'This decorator can be used to indicate to type checkers that\n        the decorated method cannot be overridden, and decorated class\n        cannot be subclassed. For example:\n\n            class Base:\n                @final\n                def done(self) -> None:\n                    ...\n            class Sub(Base):\n                def done(self) -> None:  # Error reported by type checker\n                    ...\n            @final\n            class Leaf:\n                ...\n            class Other(Leaf):  # Error reported by type checker\n                ...\n\n        There is no runtime checking of these properties. The decorator\n        sets the ``__final__`` attribute to ``True`` on the decorated object\n        to allow runtime introspection.\n        '
        try:
            f.__final__ = True
        except (AttributeError, TypeError):
            pass
        return f

def IntVar(name):
    if False:
        while True:
            i = 10
    return typing.TypeVar(name)
if sys.version_info >= (3, 10, 1):
    Literal = typing.Literal
else:

    def _flatten_literal_params(parameters):
        if False:
            i = 10
            return i + 15
        'An internal helper for Literal creation: flatten Literals among parameters'
        params = []
        for p in parameters:
            if isinstance(p, _LiteralGenericAlias):
                params.extend(p.__args__)
            else:
                params.append(p)
        return tuple(params)

    def _value_and_type_iter(params):
        if False:
            i = 10
            return i + 15
        for p in params:
            yield (p, type(p))

    class _LiteralGenericAlias(typing._GenericAlias, _root=True):

        def __eq__(self, other):
            if False:
                return 10
            if not isinstance(other, _LiteralGenericAlias):
                return NotImplemented
            these_args_deduped = set(_value_and_type_iter(self.__args__))
            other_args_deduped = set(_value_and_type_iter(other.__args__))
            return these_args_deduped == other_args_deduped

        def __hash__(self):
            if False:
                while True:
                    i = 10
            return hash(frozenset(_value_and_type_iter(self.__args__)))

    class _LiteralForm(_ExtensionsSpecialForm, _root=True):

        def __init__(self, doc: str):
            if False:
                return 10
            self._name = 'Literal'
            self._doc = self.__doc__ = doc

        def __getitem__(self, parameters):
            if False:
                while True:
                    i = 10
            if not isinstance(parameters, tuple):
                parameters = (parameters,)
            parameters = _flatten_literal_params(parameters)
            val_type_pairs = list(_value_and_type_iter(parameters))
            try:
                deduped_pairs = set(val_type_pairs)
            except TypeError:
                pass
            else:
                if len(deduped_pairs) < len(val_type_pairs):
                    new_parameters = []
                    for pair in val_type_pairs:
                        if pair in deduped_pairs:
                            new_parameters.append(pair[0])
                            deduped_pairs.remove(pair)
                    assert not deduped_pairs, deduped_pairs
                    parameters = tuple(new_parameters)
            return _LiteralGenericAlias(self, parameters)
    Literal = _LiteralForm(doc="                           A type that can be used to indicate to type checkers\n                           that the corresponding value has a value literally equivalent\n                           to the provided parameter. For example:\n\n                               var: Literal[4] = 4\n\n                           The type checker understands that 'var' is literally equal to\n                           the value 4 and no other value.\n\n                           Literal[...] cannot be subclassed. There is no runtime\n                           checking verifying that the parameter is actually a value\n                           instead of a type.")
_overload_dummy = typing._overload_dummy
if hasattr(typing, 'get_overloads'):
    overload = typing.overload
    get_overloads = typing.get_overloads
    clear_overloads = typing.clear_overloads
else:
    _overload_registry = collections.defaultdict(functools.partial(collections.defaultdict, dict))

    def overload(func):
        if False:
            i = 10
            return i + 15
        'Decorator for overloaded functions/methods.\n\n        In a stub file, place two or more stub definitions for the same\n        function in a row, each decorated with @overload.  For example:\n\n        @overload\n        def utf8(value: None) -> None: ...\n        @overload\n        def utf8(value: bytes) -> bytes: ...\n        @overload\n        def utf8(value: str) -> bytes: ...\n\n        In a non-stub file (i.e. a regular .py file), do the same but\n        follow it with an implementation.  The implementation should *not*\n        be decorated with @overload.  For example:\n\n        @overload\n        def utf8(value: None) -> None: ...\n        @overload\n        def utf8(value: bytes) -> bytes: ...\n        @overload\n        def utf8(value: str) -> bytes: ...\n        def utf8(value):\n            # implementation goes here\n\n        The overloads for a function can be retrieved at runtime using the\n        get_overloads() function.\n        '
        f = getattr(func, '__func__', func)
        try:
            _overload_registry[f.__module__][f.__qualname__][f.__code__.co_firstlineno] = func
        except AttributeError:
            pass
        return _overload_dummy

    def get_overloads(func):
        if False:
            i = 10
            return i + 15
        'Return all defined overloads for *func* as a sequence.'
        f = getattr(func, '__func__', func)
        if f.__module__ not in _overload_registry:
            return []
        mod_dict = _overload_registry[f.__module__]
        if f.__qualname__ not in mod_dict:
            return []
        return list(mod_dict[f.__qualname__].values())

    def clear_overloads():
        if False:
            print('Hello World!')
        'Clear all overloads in the registry.'
        _overload_registry.clear()
Type = typing.Type
Awaitable = typing.Awaitable
Coroutine = typing.Coroutine
AsyncIterable = typing.AsyncIterable
AsyncIterator = typing.AsyncIterator
Deque = typing.Deque
ContextManager = typing.ContextManager
AsyncContextManager = typing.AsyncContextManager
DefaultDict = typing.DefaultDict
if hasattr(typing, 'OrderedDict'):
    OrderedDict = typing.OrderedDict
else:
    OrderedDict = typing._alias(collections.OrderedDict, (KT, VT))
Counter = typing.Counter
ChainMap = typing.ChainMap
AsyncGenerator = typing.AsyncGenerator
Text = typing.Text
TYPE_CHECKING = typing.TYPE_CHECKING
_PROTO_ALLOWLIST = {'collections.abc': ['Callable', 'Awaitable', 'Iterable', 'Iterator', 'AsyncIterable', 'Hashable', 'Sized', 'Container', 'Collection', 'Reversible', 'Buffer'], 'contextlib': ['AbstractContextManager', 'AbstractAsyncContextManager'], 'typing_extensions': ['Buffer']}
_EXCLUDED_ATTRS = {'__abstractmethods__', '__annotations__', '__weakref__', '_is_protocol', '_is_runtime_protocol', '__dict__', '__slots__', '__parameters__', '__orig_bases__', '__module__', '_MutableMapping__marker', '__doc__', '__subclasshook__', '__orig_class__', '__init__', '__new__', '__protocol_attrs__', '__callable_proto_members_only__'}
if sys.version_info < (3, 8):
    _EXCLUDED_ATTRS |= {'_gorg', '__next_in_mro__', '__extra__', '__tree_hash__', '__args__', '__origin__'}
if sys.version_info >= (3, 9):
    _EXCLUDED_ATTRS.add('__class_getitem__')
if sys.version_info >= (3, 12):
    _EXCLUDED_ATTRS.add('__type_params__')
_EXCLUDED_ATTRS = frozenset(_EXCLUDED_ATTRS)

def _get_protocol_attrs(cls):
    if False:
        for i in range(10):
            print('nop')
    attrs = set()
    for base in cls.__mro__[:-1]:
        if base.__name__ in {'Protocol', 'Generic'}:
            continue
        annotations = getattr(base, '__annotations__', {})
        for attr in (*base.__dict__, *annotations):
            if not attr.startswith('_abc_') and attr not in _EXCLUDED_ATTRS:
                attrs.add(attr)
    return attrs

def _maybe_adjust_parameters(cls):
    if False:
        while True:
            i = 10
    'Helper function used in Protocol.__init_subclass__ and _TypedDictMeta.__new__.\n\n    The contents of this function are very similar\n    to logic found in typing.Generic.__init_subclass__\n    on the CPython main branch.\n    '
    tvars = []
    if '__orig_bases__' in cls.__dict__:
        tvars = _collect_type_vars(cls.__orig_bases__)
        gvars = None
        for base in cls.__orig_bases__:
            if isinstance(base, typing._GenericAlias) and base.__origin__ in (typing.Generic, Protocol):
                the_base = base.__origin__.__name__
                if gvars is not None:
                    raise TypeError('Cannot inherit from Generic[...] and/or Protocol[...] multiple types.')
                gvars = base.__parameters__
        if gvars is None:
            gvars = tvars
        else:
            tvarset = set(tvars)
            gvarset = set(gvars)
            if not tvarset <= gvarset:
                s_vars = ', '.join((str(t) for t in tvars if t not in gvarset))
                s_args = ', '.join((str(g) for g in gvars))
                raise TypeError(f'Some type variables ({s_vars}) are not listed in {the_base}[{s_args}]')
            tvars = gvars
    cls.__parameters__ = tuple(tvars)

def _caller(depth=2):
    if False:
        return 10
    try:
        return sys._getframe(depth).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        return None
if sys.version_info >= (3, 12):
    Protocol = typing.Protocol
else:

    def _allow_reckless_class_checks(depth=3):
        if False:
            return 10
        'Allow instance and class checks for special stdlib modules.\n        The abc and functools modules indiscriminately call isinstance() and\n        issubclass() on the whole MRO of a user class, which may contain protocols.\n        '
        return _caller(depth) in {'abc', 'functools', None}

    def _no_init(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if type(self)._is_protocol:
            raise TypeError('Protocols cannot be instantiated')
    if sys.version_info >= (3, 8):
        _typing_Protocol = typing.Protocol
        _ProtocolMetaBase = type(_typing_Protocol)
    else:
        _typing_Protocol = _marker
        _ProtocolMetaBase = abc.ABCMeta

    class _ProtocolMeta(_ProtocolMetaBase):

        def __new__(mcls, name, bases, namespace, **kwargs):
            if False:
                i = 10
                return i + 15
            if name == 'Protocol' and len(bases) < 2:
                pass
            elif {Protocol, _typing_Protocol} & set(bases):
                for base in bases:
                    if not (base in {object, typing.Generic, Protocol, _typing_Protocol} or base.__name__ in _PROTO_ALLOWLIST.get(base.__module__, []) or is_protocol(base)):
                        raise TypeError(f'Protocols can only inherit from other protocols, got {base!r}')
            return abc.ABCMeta.__new__(mcls, name, bases, namespace, **kwargs)

        def __init__(cls, *args, **kwargs):
            if False:
                return 10
            abc.ABCMeta.__init__(cls, *args, **kwargs)
            if getattr(cls, '_is_protocol', False):
                cls.__protocol_attrs__ = _get_protocol_attrs(cls)
                cls.__callable_proto_members_only__ = all((callable(getattr(cls, attr, None)) for attr in cls.__protocol_attrs__))

        def __subclasscheck__(cls, other):
            if False:
                i = 10
                return i + 15
            if cls is Protocol:
                return type.__subclasscheck__(cls, other)
            if getattr(cls, '_is_protocol', False) and (not _allow_reckless_class_checks()):
                if not isinstance(other, type):
                    raise TypeError('issubclass() arg 1 must be a class')
                if not cls.__callable_proto_members_only__ and cls.__dict__.get('__subclasshook__') is _proto_hook:
                    raise TypeError("Protocols with non-method members don't support issubclass()")
                if not getattr(cls, '_is_runtime_protocol', False):
                    raise TypeError('Instance and class checks can only be used with @runtime_checkable protocols')
            return abc.ABCMeta.__subclasscheck__(cls, other)

        def __instancecheck__(cls, instance):
            if False:
                print('Hello World!')
            if cls is Protocol:
                return type.__instancecheck__(cls, instance)
            if not getattr(cls, '_is_protocol', False):
                return abc.ABCMeta.__instancecheck__(cls, instance)
            if not getattr(cls, '_is_runtime_protocol', False) and (not _allow_reckless_class_checks()):
                raise TypeError('Instance and class checks can only be used with @runtime_checkable protocols')
            if abc.ABCMeta.__instancecheck__(cls, instance):
                return True
            for attr in cls.__protocol_attrs__:
                try:
                    val = inspect.getattr_static(instance, attr)
                except AttributeError:
                    break
                if val is None and callable(getattr(cls, attr, None)):
                    break
            else:
                return True
            return False

        def __eq__(cls, other):
            if False:
                print('Hello World!')
            if abc.ABCMeta.__eq__(cls, other) is True:
                return True
            return cls is Protocol and other is getattr(typing, 'Protocol', object())

        def __hash__(cls) -> int:
            if False:
                for i in range(10):
                    print('nop')
            return type.__hash__(cls)

    @classmethod
    def _proto_hook(cls, other):
        if False:
            i = 10
            return i + 15
        if not cls.__dict__.get('_is_protocol', False):
            return NotImplemented
        for attr in cls.__protocol_attrs__:
            for base in other.__mro__:
                if attr in base.__dict__:
                    if base.__dict__[attr] is None:
                        return NotImplemented
                    break
                annotations = getattr(base, '__annotations__', {})
                if isinstance(annotations, collections.abc.Mapping) and attr in annotations and is_protocol(other):
                    break
            else:
                return NotImplemented
        return True
    if sys.version_info >= (3, 8):

        class Protocol(typing.Generic, metaclass=_ProtocolMeta):
            __doc__ = typing.Protocol.__doc__
            __slots__ = ()
            _is_protocol = True
            _is_runtime_protocol = False

            def __init_subclass__(cls, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init_subclass__(*args, **kwargs)
                if not cls.__dict__.get('_is_protocol', False):
                    cls._is_protocol = any((b is Protocol for b in cls.__bases__))
                if '__subclasshook__' not in cls.__dict__:
                    cls.__subclasshook__ = _proto_hook
                if cls._is_protocol and cls.__init__ is Protocol.__init__:
                    cls.__init__ = _no_init
    else:

        class Protocol(metaclass=_ProtocolMeta):
            """Base class for protocol classes. Protocol classes are defined as::

                class Proto(Protocol):
                    def meth(self) -> int:
                        ...

            Such classes are primarily used with static type checkers that recognize
            structural subtyping (static duck-typing), for example::

                class C:
                    def meth(self) -> int:
                        return 0

                def func(x: Proto) -> int:
                    return x.meth()

                func(C())  # Passes static type check

            See PEP 544 for details. Protocol classes decorated with
            @typing_extensions.runtime_checkable act
            as simple-minded runtime-checkable protocols that check
            only the presence of given attributes, ignoring their type signatures.

            Protocol classes can be generic, they are defined as::

                class GenProto(Protocol[T]):
                    def meth(self) -> T:
                        ...
            """
            __slots__ = ()
            _is_protocol = True
            _is_runtime_protocol = False

            def __new__(cls, *args, **kwds):
                if False:
                    i = 10
                    return i + 15
                if cls is Protocol:
                    raise TypeError('Type Protocol cannot be instantiated; it can only be used as a base class')
                return super().__new__(cls)

            @typing._tp_cache
            def __class_getitem__(cls, params):
                if False:
                    for i in range(10):
                        print('nop')
                if not isinstance(params, tuple):
                    params = (params,)
                if not params and cls is not typing.Tuple:
                    raise TypeError(f'Parameter list to {cls.__qualname__}[...] cannot be empty')
                msg = 'Parameters to generic types must be types.'
                params = tuple((typing._type_check(p, msg) for p in params))
                if cls is Protocol:
                    if not all((isinstance(p, typing.TypeVar) for p in params)):
                        i = 0
                        while isinstance(params[i], typing.TypeVar):
                            i += 1
                        raise TypeError(f'Parameters to Protocol[...] must all be type variables. Parameter {i + 1} is {params[i]}')
                    if len(set(params)) != len(params):
                        raise TypeError('Parameters to Protocol[...] must all be unique')
                else:
                    _check_generic(cls, params, len(cls.__parameters__))
                return typing._GenericAlias(cls, params)

            def __init_subclass__(cls, *args, **kwargs):
                if False:
                    return 10
                if '__orig_bases__' in cls.__dict__:
                    error = typing.Generic in cls.__orig_bases__
                else:
                    error = typing.Generic in cls.__bases__
                if error:
                    raise TypeError('Cannot inherit from plain Generic')
                _maybe_adjust_parameters(cls)
                if not cls.__dict__.get('_is_protocol', None):
                    cls._is_protocol = any((b is Protocol for b in cls.__bases__))
                if '__subclasshook__' not in cls.__dict__:
                    cls.__subclasshook__ = _proto_hook
                if cls._is_protocol and cls.__init__ is Protocol.__init__:
                    cls.__init__ = _no_init
if sys.version_info >= (3, 8):
    runtime_checkable = typing.runtime_checkable
else:

    def runtime_checkable(cls):
        if False:
            for i in range(10):
                print('nop')
        'Mark a protocol class as a runtime protocol, so that it\n        can be used with isinstance() and issubclass(). Raise TypeError\n        if applied to a non-protocol class.\n\n        This allows a simple-minded structural check very similar to the\n        one-offs in collections.abc such as Hashable.\n        '
        if not ((isinstance(cls, _ProtocolMeta) or issubclass(cls, typing.Generic)) and getattr(cls, '_is_protocol', False)):
            raise TypeError(f'@runtime_checkable can be only applied to protocol classes, got {cls!r}')
        cls._is_runtime_protocol = True
        return cls
runtime = runtime_checkable
if sys.version_info >= (3, 12):
    SupportsInt = typing.SupportsInt
    SupportsFloat = typing.SupportsFloat
    SupportsComplex = typing.SupportsComplex
    SupportsBytes = typing.SupportsBytes
    SupportsIndex = typing.SupportsIndex
    SupportsAbs = typing.SupportsAbs
    SupportsRound = typing.SupportsRound
else:

    @runtime_checkable
    class SupportsInt(Protocol):
        """An ABC with one abstract method __int__."""
        __slots__ = ()

        @abc.abstractmethod
        def __int__(self) -> int:
            if False:
                print('Hello World!')
            pass

    @runtime_checkable
    class SupportsFloat(Protocol):
        """An ABC with one abstract method __float__."""
        __slots__ = ()

        @abc.abstractmethod
        def __float__(self) -> float:
            if False:
                return 10
            pass

    @runtime_checkable
    class SupportsComplex(Protocol):
        """An ABC with one abstract method __complex__."""
        __slots__ = ()

        @abc.abstractmethod
        def __complex__(self) -> complex:
            if False:
                while True:
                    i = 10
            pass

    @runtime_checkable
    class SupportsBytes(Protocol):
        """An ABC with one abstract method __bytes__."""
        __slots__ = ()

        @abc.abstractmethod
        def __bytes__(self) -> bytes:
            if False:
                return 10
            pass

    @runtime_checkable
    class SupportsIndex(Protocol):
        __slots__ = ()

        @abc.abstractmethod
        def __index__(self) -> int:
            if False:
                i = 10
                return i + 15
            pass

    @runtime_checkable
    class SupportsAbs(Protocol[T_co]):
        """
        An ABC with one abstract method __abs__ that is covariant in its return type.
        """
        __slots__ = ()

        @abc.abstractmethod
        def __abs__(self) -> T_co:
            if False:
                for i in range(10):
                    print('nop')
            pass

    @runtime_checkable
    class SupportsRound(Protocol[T_co]):
        """
        An ABC with one abstract method __round__ that is covariant in its return type.
        """
        __slots__ = ()

        @abc.abstractmethod
        def __round__(self, ndigits: int=0) -> T_co:
            if False:
                print('Hello World!')
            pass

def _ensure_subclassable(mro_entries):
    if False:
        return 10

    def inner(func):
        if False:
            for i in range(10):
                print('nop')
        if sys.implementation.name == 'pypy' and sys.version_info < (3, 9):
            cls_dict = {'__call__': staticmethod(func), '__mro_entries__': staticmethod(mro_entries)}
            t = type(func.__name__, (), cls_dict)
            return functools.update_wrapper(t(), func)
        else:
            func.__mro_entries__ = mro_entries
            return func
    return inner
if sys.version_info >= (3, 13):
    TypedDict = typing.TypedDict
    _TypedDictMeta = typing._TypedDictMeta
    is_typeddict = typing.is_typeddict
else:
    _TAKES_MODULE = 'module' in inspect.signature(typing._type_check).parameters
    if sys.version_info >= (3, 8):
        _fake_name = 'Protocol'
    else:
        _fake_name = '_Protocol'

    class _TypedDictMeta(type):

        def __new__(cls, name, bases, ns, total=True):
            if False:
                i = 10
                return i + 15
            'Create new typed dict class object.\n\n            This method is called when TypedDict is subclassed,\n            or when TypedDict is instantiated. This way\n            TypedDict supports all three syntax forms described in its docstring.\n            Subclasses and instances of TypedDict return actual dictionaries.\n            '
            for base in bases:
                if type(base) is not _TypedDictMeta and base is not typing.Generic:
                    raise TypeError('cannot inherit from both a TypedDict type and a non-TypedDict base class')
            if any((issubclass(b, typing.Generic) for b in bases)):
                generic_base = (typing.Generic,)
            else:
                generic_base = ()
            tp_dict = type.__new__(_TypedDictMeta, _fake_name, (*generic_base, dict), ns)
            tp_dict.__name__ = name
            if tp_dict.__qualname__ == _fake_name:
                tp_dict.__qualname__ = name
            if not hasattr(tp_dict, '__orig_bases__'):
                tp_dict.__orig_bases__ = bases
            annotations = {}
            own_annotations = ns.get('__annotations__', {})
            msg = "TypedDict('Name', {f0: t0, f1: t1, ...}); each t must be a type"
            if _TAKES_MODULE:
                own_annotations = {n: typing._type_check(tp, msg, module=tp_dict.__module__) for (n, tp) in own_annotations.items()}
            else:
                own_annotations = {n: typing._type_check(tp, msg) for (n, tp) in own_annotations.items()}
            required_keys = set()
            optional_keys = set()
            for base in bases:
                annotations.update(base.__dict__.get('__annotations__', {}))
                required_keys.update(base.__dict__.get('__required_keys__', ()))
                optional_keys.update(base.__dict__.get('__optional_keys__', ()))
            annotations.update(own_annotations)
            for (annotation_key, annotation_type) in own_annotations.items():
                annotation_origin = get_origin(annotation_type)
                if annotation_origin is Annotated:
                    annotation_args = get_args(annotation_type)
                    if annotation_args:
                        annotation_type = annotation_args[0]
                        annotation_origin = get_origin(annotation_type)
                if annotation_origin is Required:
                    required_keys.add(annotation_key)
                elif annotation_origin is NotRequired:
                    optional_keys.add(annotation_key)
                elif total:
                    required_keys.add(annotation_key)
                else:
                    optional_keys.add(annotation_key)
            tp_dict.__annotations__ = annotations
            tp_dict.__required_keys__ = frozenset(required_keys)
            tp_dict.__optional_keys__ = frozenset(optional_keys)
            if not hasattr(tp_dict, '__total__'):
                tp_dict.__total__ = total
            return tp_dict
        __call__ = dict

        def __subclasscheck__(cls, other):
            if False:
                for i in range(10):
                    print('nop')
            raise TypeError('TypedDict does not support instance and class checks')
        __instancecheck__ = __subclasscheck__
    _TypedDict = type.__new__(_TypedDictMeta, 'TypedDict', (), {})

    @_ensure_subclassable(lambda bases: (_TypedDict,))
    def TypedDict(__typename, __fields=_marker, *, total=True, **kwargs):
        if False:
            return 10
        'A simple typed namespace. At runtime it is equivalent to a plain dict.\n\n        TypedDict creates a dictionary type such that a type checker will expect all\n        instances to have a certain set of keys, where each key is\n        associated with a value of a consistent type. This expectation\n        is not checked at runtime.\n\n        Usage::\n\n            class Point2D(TypedDict):\n                x: int\n                y: int\n                label: str\n\n            a: Point2D = {\'x\': 1, \'y\': 2, \'label\': \'good\'}  # OK\n            b: Point2D = {\'z\': 3, \'label\': \'bad\'}           # Fails type check\n\n            assert Point2D(x=1, y=2, label=\'first\') == dict(x=1, y=2, label=\'first\')\n\n        The type info can be accessed via the Point2D.__annotations__ dict, and\n        the Point2D.__required_keys__ and Point2D.__optional_keys__ frozensets.\n        TypedDict supports an additional equivalent form::\n\n            Point2D = TypedDict(\'Point2D\', {\'x\': int, \'y\': int, \'label\': str})\n\n        By default, all keys must be present in a TypedDict. It is possible\n        to override this by specifying totality::\n\n            class Point2D(TypedDict, total=False):\n                x: int\n                y: int\n\n        This means that a Point2D TypedDict can have any of the keys omitted. A type\n        checker is only expected to support a literal False or True as the value of\n        the total argument. True is the default, and makes all items defined in the\n        class body be required.\n\n        The Required and NotRequired special forms can also be used to mark\n        individual keys as being required or not required::\n\n            class Point2D(TypedDict):\n                x: int  # the "x" key must always be present (Required is the default)\n                y: NotRequired[int]  # the "y" key can be omitted\n\n        See PEP 655 for more details on Required and NotRequired.\n        '
        if __fields is _marker or __fields is None:
            if __fields is _marker:
                deprecated_thing = "Failing to pass a value for the 'fields' parameter"
            else:
                deprecated_thing = "Passing `None` as the 'fields' parameter"
            example = f'`{__typename} = TypedDict({__typename!r}, {{}})`'
            deprecation_msg = f'{deprecated_thing} is deprecated and will be disallowed in Python 3.15. To create a TypedDict class with 0 fields using the functional syntax, pass an empty dictionary, e.g. ' + example + '.'
            warnings.warn(deprecation_msg, DeprecationWarning, stacklevel=2)
            __fields = kwargs
        elif kwargs:
            raise TypeError('TypedDict takes either a dict or keyword arguments, but not both')
        if kwargs:
            warnings.warn('The kwargs-based syntax for TypedDict definitions is deprecated in Python 3.11, will be removed in Python 3.13, and may not be understood by third-party type checkers.', DeprecationWarning, stacklevel=2)
        ns = {'__annotations__': dict(__fields)}
        module = _caller()
        if module is not None:
            ns['__module__'] = module
        td = _TypedDictMeta(__typename, (), ns, total=total)
        td.__orig_bases__ = (TypedDict,)
        return td
    if hasattr(typing, '_TypedDictMeta'):
        _TYPEDDICT_TYPES = (typing._TypedDictMeta, _TypedDictMeta)
    else:
        _TYPEDDICT_TYPES = (_TypedDictMeta,)

    def is_typeddict(tp):
        if False:
            for i in range(10):
                print('nop')
        'Check if an annotation is a TypedDict class\n\n        For example::\n            class Film(TypedDict):\n                title: str\n                year: int\n\n            is_typeddict(Film)  # => True\n            is_typeddict(Union[list, str])  # => False\n        '
        if hasattr(typing, 'TypedDict') and tp is typing.TypedDict:
            return False
        return isinstance(tp, _TYPEDDICT_TYPES)
if hasattr(typing, 'assert_type'):
    assert_type = typing.assert_type
else:

    def assert_type(__val, __typ):
        if False:
            i = 10
            return i + 15
        'Assert (to the type checker) that the value is of the given type.\n\n        When the type checker encounters a call to assert_type(), it\n        emits an error if the value is not of the specified type::\n\n            def greet(name: str) -> None:\n                assert_type(name, str)  # ok\n                assert_type(name, int)  # type checker error\n\n        At runtime this returns the first argument unchanged and otherwise\n        does nothing.\n        '
        return __val
if hasattr(typing, 'Required'):
    get_type_hints = typing.get_type_hints
else:

    def _strip_extras(t):
        if False:
            return 10
        'Strips Annotated, Required and NotRequired from a given type.'
        if isinstance(t, _AnnotatedAlias):
            return _strip_extras(t.__origin__)
        if hasattr(t, '__origin__') and t.__origin__ in (Required, NotRequired):
            return _strip_extras(t.__args__[0])
        if isinstance(t, typing._GenericAlias):
            stripped_args = tuple((_strip_extras(a) for a in t.__args__))
            if stripped_args == t.__args__:
                return t
            return t.copy_with(stripped_args)
        if hasattr(_types, 'GenericAlias') and isinstance(t, _types.GenericAlias):
            stripped_args = tuple((_strip_extras(a) for a in t.__args__))
            if stripped_args == t.__args__:
                return t
            return _types.GenericAlias(t.__origin__, stripped_args)
        if hasattr(_types, 'UnionType') and isinstance(t, _types.UnionType):
            stripped_args = tuple((_strip_extras(a) for a in t.__args__))
            if stripped_args == t.__args__:
                return t
            return functools.reduce(operator.or_, stripped_args)
        return t

    def get_type_hints(obj, globalns=None, localns=None, include_extras=False):
        if False:
            i = 10
            return i + 15
        "Return type hints for an object.\n\n        This is often the same as obj.__annotations__, but it handles\n        forward references encoded as string literals, adds Optional[t] if a\n        default value equal to None is set and recursively replaces all\n        'Annotated[T, ...]', 'Required[T]' or 'NotRequired[T]' with 'T'\n        (unless 'include_extras=True').\n\n        The argument may be a module, class, method, or function. The annotations\n        are returned as a dictionary. For classes, annotations include also\n        inherited members.\n\n        TypeError is raised if the argument is not of a type that can contain\n        annotations, and an empty dictionary is returned if no annotations are\n        present.\n\n        BEWARE -- the behavior of globalns and localns is counterintuitive\n        (unless you are familiar with how eval() and exec() work).  The\n        search order is locals first, then globals.\n\n        - If no dict arguments are passed, an attempt is made to use the\n          globals from obj (or the respective module's globals for classes),\n          and these are also used as the locals.  If the object does not appear\n          to have globals, an empty dictionary is used.\n\n        - If one dict argument is passed, it is used for both globals and\n          locals.\n\n        - If two dict arguments are passed, they specify globals and\n          locals, respectively.\n        "
        if hasattr(typing, 'Annotated'):
            hint = typing.get_type_hints(obj, globalns=globalns, localns=localns, include_extras=True)
        else:
            hint = typing.get_type_hints(obj, globalns=globalns, localns=localns)
        if include_extras:
            return hint
        return {k: _strip_extras(t) for (k, t) in hint.items()}
if hasattr(typing, 'Annotated'):
    Annotated = typing.Annotated
    _AnnotatedAlias = typing._AnnotatedAlias
else:

    class _AnnotatedAlias(typing._GenericAlias, _root=True):
        """Runtime representation of an annotated type.

        At its core 'Annotated[t, dec1, dec2, ...]' is an alias for the type 't'
        with extra annotations. The alias behaves like a normal typing alias,
        instantiating is the same as instantiating the underlying type, binding
        it to types is also the same.
        """

        def __init__(self, origin, metadata):
            if False:
                while True:
                    i = 10
            if isinstance(origin, _AnnotatedAlias):
                metadata = origin.__metadata__ + metadata
                origin = origin.__origin__
            super().__init__(origin, origin)
            self.__metadata__ = metadata

        def copy_with(self, params):
            if False:
                i = 10
                return i + 15
            assert len(params) == 1
            new_type = params[0]
            return _AnnotatedAlias(new_type, self.__metadata__)

        def __repr__(self):
            if False:
                while True:
                    i = 10
            return f"typing_extensions.Annotated[{typing._type_repr(self.__origin__)}, {', '.join((repr(a) for a in self.__metadata__))}]"

        def __reduce__(self):
            if False:
                while True:
                    i = 10
            return (operator.getitem, (Annotated, (self.__origin__,) + self.__metadata__))

        def __eq__(self, other):
            if False:
                i = 10
                return i + 15
            if not isinstance(other, _AnnotatedAlias):
                return NotImplemented
            if self.__origin__ != other.__origin__:
                return False
            return self.__metadata__ == other.__metadata__

        def __hash__(self):
            if False:
                return 10
            return hash((self.__origin__, self.__metadata__))

    class Annotated:
        """Add context specific metadata to a type.

        Example: Annotated[int, runtime_check.Unsigned] indicates to the
        hypothetical runtime_check module that this type is an unsigned int.
        Every other consumer of this type can ignore this metadata and treat
        this type as int.

        The first argument to Annotated must be a valid type (and will be in
        the __origin__ field), the remaining arguments are kept as a tuple in
        the __extra__ field.

        Details:

        - It's an error to call `Annotated` with less than two arguments.
        - Nested Annotated are flattened::

            Annotated[Annotated[T, Ann1, Ann2], Ann3] == Annotated[T, Ann1, Ann2, Ann3]

        - Instantiating an annotated type is equivalent to instantiating the
        underlying type::

            Annotated[C, Ann1](5) == C(5)

        - Annotated can be used as a generic type alias::

            Optimized = Annotated[T, runtime.Optimize()]
            Optimized[int] == Annotated[int, runtime.Optimize()]

            OptimizedList = Annotated[List[T], runtime.Optimize()]
            OptimizedList[int] == Annotated[List[int], runtime.Optimize()]
        """
        __slots__ = ()

        def __new__(cls, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            raise TypeError('Type Annotated cannot be instantiated.')

        @typing._tp_cache
        def __class_getitem__(cls, params):
            if False:
                return 10
            if not isinstance(params, tuple) or len(params) < 2:
                raise TypeError('Annotated[...] should be used with at least two arguments (a type and an annotation).')
            allowed_special_forms = (ClassVar, Final)
            if get_origin(params[0]) in allowed_special_forms:
                origin = params[0]
            else:
                msg = 'Annotated[t, ...]: t must be a type.'
                origin = typing._type_check(params[0], msg)
            metadata = tuple(params[1:])
            return _AnnotatedAlias(origin, metadata)

        def __init_subclass__(cls, *args, **kwargs):
            if False:
                return 10
            raise TypeError(f'Cannot subclass {cls.__module__}.Annotated')
if sys.version_info[:2] >= (3, 10):
    get_origin = typing.get_origin
    get_args = typing.get_args
else:
    try:
        from typing import _BaseGenericAlias
    except ImportError:
        _BaseGenericAlias = typing._GenericAlias
    try:
        from typing import GenericAlias as _typing_GenericAlias
    except ImportError:
        _typing_GenericAlias = typing._GenericAlias

    def get_origin(tp):
        if False:
            while True:
                i = 10
        'Get the unsubscripted version of a type.\n\n        This supports generic types, Callable, Tuple, Union, Literal, Final, ClassVar\n        and Annotated. Return None for unsupported types. Examples::\n\n            get_origin(Literal[42]) is Literal\n            get_origin(int) is None\n            get_origin(ClassVar[int]) is ClassVar\n            get_origin(Generic) is Generic\n            get_origin(Generic[T]) is Generic\n            get_origin(Union[T, int]) is Union\n            get_origin(List[Tuple[T, T]][int]) == list\n            get_origin(P.args) is P\n        '
        if isinstance(tp, _AnnotatedAlias):
            return Annotated
        if isinstance(tp, (typing._GenericAlias, _typing_GenericAlias, _BaseGenericAlias, ParamSpecArgs, ParamSpecKwargs)):
            return tp.__origin__
        if tp is typing.Generic:
            return typing.Generic
        return None

    def get_args(tp):
        if False:
            for i in range(10):
                print('nop')
        'Get type arguments with all substitutions performed.\n\n        For unions, basic simplifications used by Union constructor are performed.\n        Examples::\n            get_args(Dict[str, int]) == (str, int)\n            get_args(int) == ()\n            get_args(Union[int, Union[T, int], str][int]) == (int, str)\n            get_args(Union[int, Tuple[T, int]][str]) == (int, Tuple[str, int])\n            get_args(Callable[[], T][int]) == ([], int)\n        '
        if isinstance(tp, _AnnotatedAlias):
            return (tp.__origin__,) + tp.__metadata__
        if isinstance(tp, (typing._GenericAlias, _typing_GenericAlias)):
            if getattr(tp, '_special', False):
                return ()
            res = tp.__args__
            if get_origin(tp) is collections.abc.Callable and res[0] is not Ellipsis:
                res = (list(res[:-1]), res[-1])
            return res
        return ()
if hasattr(typing, 'TypeAlias'):
    TypeAlias = typing.TypeAlias
elif sys.version_info[:2] >= (3, 9):

    @_ExtensionsSpecialForm
    def TypeAlias(self, parameters):
        if False:
            return 10
        "Special marker indicating that an assignment should\n        be recognized as a proper type alias definition by type\n        checkers.\n\n        For example::\n\n            Predicate: TypeAlias = Callable[..., bool]\n\n        It's invalid when used anywhere except as in the example above.\n        "
        raise TypeError(f'{self} is not subscriptable')
else:
    TypeAlias = _ExtensionsSpecialForm('TypeAlias', doc="Special marker indicating that an assignment should\n        be recognized as a proper type alias definition by type\n        checkers.\n\n        For example::\n\n            Predicate: TypeAlias = Callable[..., bool]\n\n        It's invalid when used anywhere except as in the example\n        above.")

def _set_default(type_param, default):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(default, (tuple, list)):
        type_param.__default__ = tuple((typing._type_check(d, 'Default must be a type') for d in default))
    elif default != _marker:
        type_param.__default__ = typing._type_check(default, 'Default must be a type')
    else:
        type_param.__default__ = None

def _set_module(typevarlike):
    if False:
        return 10
    def_mod = _caller(depth=3)
    if def_mod != 'typing_extensions':
        typevarlike.__module__ = def_mod

class _DefaultMixin:
    """Mixin for TypeVarLike defaults."""
    __slots__ = ()
    __init__ = _set_default

class _TypeVarLikeMeta(type):

    def __instancecheck__(cls, __instance: Any) -> bool:
        if False:
            print('Hello World!')
        return isinstance(__instance, cls._backported_typevarlike)

class TypeVar(metaclass=_TypeVarLikeMeta):
    """Type variable."""
    _backported_typevarlike = typing.TypeVar

    def __new__(cls, name, *constraints, bound=None, covariant=False, contravariant=False, default=_marker, infer_variance=False):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(typing, 'TypeAliasType'):
            typevar = typing.TypeVar(name, *constraints, bound=bound, covariant=covariant, contravariant=contravariant, infer_variance=infer_variance)
        else:
            typevar = typing.TypeVar(name, *constraints, bound=bound, covariant=covariant, contravariant=contravariant)
            if infer_variance and (covariant or contravariant):
                raise ValueError('Variance cannot be specified with infer_variance.')
            typevar.__infer_variance__ = infer_variance
        _set_default(typevar, default)
        _set_module(typevar)
        return typevar

    def __init_subclass__(cls) -> None:
        if False:
            return 10
        raise TypeError(f"type '{__name__}.TypeVar' is not an acceptable base type")
if hasattr(typing, 'ParamSpecArgs'):
    ParamSpecArgs = typing.ParamSpecArgs
    ParamSpecKwargs = typing.ParamSpecKwargs
else:

    class _Immutable:
        """Mixin to indicate that object should not be copied."""
        __slots__ = ()

        def __copy__(self):
            if False:
                for i in range(10):
                    print('nop')
            return self

        def __deepcopy__(self, memo):
            if False:
                for i in range(10):
                    print('nop')
            return self

    class ParamSpecArgs(_Immutable):
        """The args for a ParamSpec object.

        Given a ParamSpec object P, P.args is an instance of ParamSpecArgs.

        ParamSpecArgs objects have a reference back to their ParamSpec:

        P.args.__origin__ is P

        This type is meant for runtime introspection and has no special meaning to
        static type checkers.
        """

        def __init__(self, origin):
            if False:
                return 10
            self.__origin__ = origin

        def __repr__(self):
            if False:
                print('Hello World!')
            return f'{self.__origin__.__name__}.args'

        def __eq__(self, other):
            if False:
                while True:
                    i = 10
            if not isinstance(other, ParamSpecArgs):
                return NotImplemented
            return self.__origin__ == other.__origin__

    class ParamSpecKwargs(_Immutable):
        """The kwargs for a ParamSpec object.

        Given a ParamSpec object P, P.kwargs is an instance of ParamSpecKwargs.

        ParamSpecKwargs objects have a reference back to their ParamSpec:

        P.kwargs.__origin__ is P

        This type is meant for runtime introspection and has no special meaning to
        static type checkers.
        """

        def __init__(self, origin):
            if False:
                return 10
            self.__origin__ = origin

        def __repr__(self):
            if False:
                while True:
                    i = 10
            return f'{self.__origin__.__name__}.kwargs'

        def __eq__(self, other):
            if False:
                i = 10
                return i + 15
            if not isinstance(other, ParamSpecKwargs):
                return NotImplemented
            return self.__origin__ == other.__origin__
if hasattr(typing, 'ParamSpec'):

    class ParamSpec(metaclass=_TypeVarLikeMeta):
        """Parameter specification."""
        _backported_typevarlike = typing.ParamSpec

        def __new__(cls, name, *, bound=None, covariant=False, contravariant=False, infer_variance=False, default=_marker):
            if False:
                print('Hello World!')
            if hasattr(typing, 'TypeAliasType'):
                paramspec = typing.ParamSpec(name, bound=bound, covariant=covariant, contravariant=contravariant, infer_variance=infer_variance)
            else:
                paramspec = typing.ParamSpec(name, bound=bound, covariant=covariant, contravariant=contravariant)
                paramspec.__infer_variance__ = infer_variance
            _set_default(paramspec, default)
            _set_module(paramspec)
            return paramspec

        def __init_subclass__(cls) -> None:
            if False:
                print('Hello World!')
            raise TypeError(f"type '{__name__}.ParamSpec' is not an acceptable base type")
else:

    class ParamSpec(list, _DefaultMixin):
        """Parameter specification variable.

        Usage::

           P = ParamSpec('P')

        Parameter specification variables exist primarily for the benefit of static
        type checkers.  They are used to forward the parameter types of one
        callable to another callable, a pattern commonly found in higher order
        functions and decorators.  They are only valid when used in ``Concatenate``,
        or s the first argument to ``Callable``. In Python 3.10 and higher,
        they are also supported in user-defined Generics at runtime.
        See class Generic for more information on generic types.  An
        example for annotating a decorator::

           T = TypeVar('T')
           P = ParamSpec('P')

           def add_logging(f: Callable[P, T]) -> Callable[P, T]:
               '''A type-safe decorator to add logging to a function.'''
               def inner(*args: P.args, **kwargs: P.kwargs) -> T:
                   logging.info(f'{f.__name__} was called')
                   return f(*args, **kwargs)
               return inner

           @add_logging
           def add_two(x: float, y: float) -> float:
               '''Add two numbers together.'''
               return x + y

        Parameter specification variables defined with covariant=True or
        contravariant=True can be used to declare covariant or contravariant
        generic types.  These keyword arguments are valid, but their actual semantics
        are yet to be decided.  See PEP 612 for details.

        Parameter specification variables can be introspected. e.g.:

           P.__name__ == 'T'
           P.__bound__ == None
           P.__covariant__ == False
           P.__contravariant__ == False

        Note that only parameter specification variables defined in global scope can
        be pickled.
        """
        __class__ = typing.TypeVar

        @property
        def args(self):
            if False:
                return 10
            return ParamSpecArgs(self)

        @property
        def kwargs(self):
            if False:
                for i in range(10):
                    print('nop')
            return ParamSpecKwargs(self)

        def __init__(self, name, *, bound=None, covariant=False, contravariant=False, infer_variance=False, default=_marker):
            if False:
                return 10
            super().__init__([self])
            self.__name__ = name
            self.__covariant__ = bool(covariant)
            self.__contravariant__ = bool(contravariant)
            self.__infer_variance__ = bool(infer_variance)
            if bound:
                self.__bound__ = typing._type_check(bound, 'Bound must be a type.')
            else:
                self.__bound__ = None
            _DefaultMixin.__init__(self, default)
            def_mod = _caller()
            if def_mod != 'typing_extensions':
                self.__module__ = def_mod

        def __repr__(self):
            if False:
                print('Hello World!')
            if self.__infer_variance__:
                prefix = ''
            elif self.__covariant__:
                prefix = '+'
            elif self.__contravariant__:
                prefix = '-'
            else:
                prefix = '~'
            return prefix + self.__name__

        def __hash__(self):
            if False:
                while True:
                    i = 10
            return object.__hash__(self)

        def __eq__(self, other):
            if False:
                while True:
                    i = 10
            return self is other

        def __reduce__(self):
            if False:
                print('Hello World!')
            return self.__name__

        def __call__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            pass
if not hasattr(typing, 'Concatenate'):

    class _ConcatenateGenericAlias(list):
        __class__ = typing._GenericAlias
        _special = False

        def __init__(self, origin, args):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(args)
            self.__origin__ = origin
            self.__args__ = args

        def __repr__(self):
            if False:
                for i in range(10):
                    print('nop')
            _type_repr = typing._type_repr
            return f"{_type_repr(self.__origin__)}[{', '.join((_type_repr(arg) for arg in self.__args__))}]"

        def __hash__(self):
            if False:
                while True:
                    i = 10
            return hash((self.__origin__, self.__args__))

        def __call__(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            pass

        @property
        def __parameters__(self):
            if False:
                print('Hello World!')
            return tuple((tp for tp in self.__args__ if isinstance(tp, (typing.TypeVar, ParamSpec))))

@typing._tp_cache
def _concatenate_getitem(self, parameters):
    if False:
        i = 10
        return i + 15
    if parameters == ():
        raise TypeError('Cannot take a Concatenate of no types.')
    if not isinstance(parameters, tuple):
        parameters = (parameters,)
    if not isinstance(parameters[-1], ParamSpec):
        raise TypeError('The last parameter to Concatenate should be a ParamSpec variable.')
    msg = 'Concatenate[arg, ...]: each arg must be a type.'
    parameters = tuple((typing._type_check(p, msg) for p in parameters))
    return _ConcatenateGenericAlias(self, parameters)
if hasattr(typing, 'Concatenate'):
    Concatenate = typing.Concatenate
    _ConcatenateGenericAlias = typing._ConcatenateGenericAlias
elif sys.version_info[:2] >= (3, 9):

    @_ExtensionsSpecialForm
    def Concatenate(self, parameters):
        if False:
            return 10
        'Used in conjunction with ``ParamSpec`` and ``Callable`` to represent a\n        higher order function which adds, removes or transforms parameters of a\n        callable.\n\n        For example::\n\n           Callable[Concatenate[int, P], int]\n\n        See PEP 612 for detailed information.\n        '
        return _concatenate_getitem(self, parameters)
else:

    class _ConcatenateForm(_ExtensionsSpecialForm, _root=True):

        def __getitem__(self, parameters):
            if False:
                return 10
            return _concatenate_getitem(self, parameters)
    Concatenate = _ConcatenateForm('Concatenate', doc='Used in conjunction with ``ParamSpec`` and ``Callable`` to represent a\n        higher order function which adds, removes or transforms parameters of a\n        callable.\n\n        For example::\n\n           Callable[Concatenate[int, P], int]\n\n        See PEP 612 for detailed information.\n        ')
if hasattr(typing, 'TypeGuard'):
    TypeGuard = typing.TypeGuard
elif sys.version_info[:2] >= (3, 9):

    @_ExtensionsSpecialForm
    def TypeGuard(self, parameters):
        if False:
            return 10
        'Special typing form used to annotate the return type of a user-defined\n        type guard function.  ``TypeGuard`` only accepts a single type argument.\n        At runtime, functions marked this way should return a boolean.\n\n        ``TypeGuard`` aims to benefit *type narrowing* -- a technique used by static\n        type checkers to determine a more precise type of an expression within a\n        program\'s code flow.  Usually type narrowing is done by analyzing\n        conditional code flow and applying the narrowing to a block of code.  The\n        conditional expression here is sometimes referred to as a "type guard".\n\n        Sometimes it would be convenient to use a user-defined boolean function\n        as a type guard.  Such a function should use ``TypeGuard[...]`` as its\n        return type to alert static type checkers to this intention.\n\n        Using  ``-> TypeGuard`` tells the static type checker that for a given\n        function:\n\n        1. The return value is a boolean.\n        2. If the return value is ``True``, the type of its argument\n        is the type inside ``TypeGuard``.\n\n        For example::\n\n            def is_str(val: Union[str, float]):\n                # "isinstance" type guard\n                if isinstance(val, str):\n                    # Type of ``val`` is narrowed to ``str``\n                    ...\n                else:\n                    # Else, type of ``val`` is narrowed to ``float``.\n                    ...\n\n        Strict type narrowing is not enforced -- ``TypeB`` need not be a narrower\n        form of ``TypeA`` (it can even be a wider form) and this may lead to\n        type-unsafe results.  The main reason is to allow for things like\n        narrowing ``List[object]`` to ``List[str]`` even though the latter is not\n        a subtype of the former, since ``List`` is invariant.  The responsibility of\n        writing type-safe type guards is left to the user.\n\n        ``TypeGuard`` also works with type variables.  For more information, see\n        PEP 647 (User-Defined Type Guards).\n        '
        item = typing._type_check(parameters, f'{self} accepts only a single type.')
        return typing._GenericAlias(self, (item,))
else:

    class _TypeGuardForm(_ExtensionsSpecialForm, _root=True):

        def __getitem__(self, parameters):
            if False:
                print('Hello World!')
            item = typing._type_check(parameters, f'{self._name} accepts only a single type')
            return typing._GenericAlias(self, (item,))
    TypeGuard = _TypeGuardForm('TypeGuard', doc='Special typing form used to annotate the return type of a user-defined\n        type guard function.  ``TypeGuard`` only accepts a single type argument.\n        At runtime, functions marked this way should return a boolean.\n\n        ``TypeGuard`` aims to benefit *type narrowing* -- a technique used by static\n        type checkers to determine a more precise type of an expression within a\n        program\'s code flow.  Usually type narrowing is done by analyzing\n        conditional code flow and applying the narrowing to a block of code.  The\n        conditional expression here is sometimes referred to as a "type guard".\n\n        Sometimes it would be convenient to use a user-defined boolean function\n        as a type guard.  Such a function should use ``TypeGuard[...]`` as its\n        return type to alert static type checkers to this intention.\n\n        Using  ``-> TypeGuard`` tells the static type checker that for a given\n        function:\n\n        1. The return value is a boolean.\n        2. If the return value is ``True``, the type of its argument\n        is the type inside ``TypeGuard``.\n\n        For example::\n\n            def is_str(val: Union[str, float]):\n                # "isinstance" type guard\n                if isinstance(val, str):\n                    # Type of ``val`` is narrowed to ``str``\n                    ...\n                else:\n                    # Else, type of ``val`` is narrowed to ``float``.\n                    ...\n\n        Strict type narrowing is not enforced -- ``TypeB`` need not be a narrower\n        form of ``TypeA`` (it can even be a wider form) and this may lead to\n        type-unsafe results.  The main reason is to allow for things like\n        narrowing ``List[object]`` to ``List[str]`` even though the latter is not\n        a subtype of the former, since ``List`` is invariant.  The responsibility of\n        writing type-safe type guards is left to the user.\n\n        ``TypeGuard`` also works with type variables.  For more information, see\n        PEP 647 (User-Defined Type Guards).\n        ')

class _SpecialForm(typing._Final, _root=True):
    __slots__ = ('_name', '__doc__', '_getitem')

    def __init__(self, getitem):
        if False:
            return 10
        self._getitem = getitem
        self._name = getitem.__name__
        self.__doc__ = getitem.__doc__

    def __getattr__(self, item):
        if False:
            for i in range(10):
                print('nop')
        if item in {'__name__', '__qualname__'}:
            return self._name
        raise AttributeError(item)

    def __mro_entries__(self, bases):
        if False:
            while True:
                i = 10
        raise TypeError(f'Cannot subclass {self!r}')

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'typing_extensions.{self._name}'

    def __reduce__(self):
        if False:
            while True:
                i = 10
        return self._name

    def __call__(self, *args, **kwds):
        if False:
            i = 10
            return i + 15
        raise TypeError(f'Cannot instantiate {self!r}')

    def __or__(self, other):
        if False:
            while True:
                i = 10
        return typing.Union[self, other]

    def __ror__(self, other):
        if False:
            while True:
                i = 10
        return typing.Union[other, self]

    def __instancecheck__(self, obj):
        if False:
            return 10
        raise TypeError(f'{self} cannot be used with isinstance()')

    def __subclasscheck__(self, cls):
        if False:
            return 10
        raise TypeError(f'{self} cannot be used with issubclass()')

    @typing._tp_cache
    def __getitem__(self, parameters):
        if False:
            while True:
                i = 10
        return self._getitem(self, parameters)
if hasattr(typing, 'LiteralString'):
    LiteralString = typing.LiteralString
else:

    @_SpecialForm
    def LiteralString(self, params):
        if False:
            i = 10
            return i + 15
        'Represents an arbitrary literal string.\n\n        Example::\n\n          from pip._vendor.typing_extensions import LiteralString\n\n          def query(sql: LiteralString) -> ...:\n              ...\n\n          query("SELECT * FROM table")  # ok\n          query(f"SELECT * FROM {input()}")  # not ok\n\n        See PEP 675 for details.\n\n        '
        raise TypeError(f'{self} is not subscriptable')
if hasattr(typing, 'Self'):
    Self = typing.Self
else:

    @_SpecialForm
    def Self(self, params):
        if False:
            for i in range(10):
                print('nop')
        'Used to spell the type of "self" in classes.\n\n        Example::\n\n          from typing import Self\n\n          class ReturnsSelf:\n              def parse(self, data: bytes) -> Self:\n                  ...\n                  return self\n\n        '
        raise TypeError(f'{self} is not subscriptable')
if hasattr(typing, 'Never'):
    Never = typing.Never
else:

    @_SpecialForm
    def Never(self, params):
        if False:
            i = 10
            return i + 15
        'The bottom type, a type that has no members.\n\n        This can be used to define a function that should never be\n        called, or a function that never returns::\n\n            from pip._vendor.typing_extensions import Never\n\n            def never_call_me(arg: Never) -> None:\n                pass\n\n            def int_or_str(arg: int | str) -> None:\n                never_call_me(arg)  # type checker error\n                match arg:\n                    case int():\n                        print("It\'s an int")\n                    case str():\n                        print("It\'s a str")\n                    case _:\n                        never_call_me(arg)  # ok, arg is of type Never\n\n        '
        raise TypeError(f'{self} is not subscriptable')
if hasattr(typing, 'Required'):
    Required = typing.Required
    NotRequired = typing.NotRequired
elif sys.version_info[:2] >= (3, 9):

    @_ExtensionsSpecialForm
    def Required(self, parameters):
        if False:
            while True:
                i = 10
        "A special typing construct to mark a key of a total=False TypedDict\n        as required. For example:\n\n            class Movie(TypedDict, total=False):\n                title: Required[str]\n                year: int\n\n            m = Movie(\n                title='The Matrix',  # typechecker error if key is omitted\n                year=1999,\n            )\n\n        There is no runtime checking that a required key is actually provided\n        when instantiating a related TypedDict.\n        "
        item = typing._type_check(parameters, f'{self._name} accepts only a single type.')
        return typing._GenericAlias(self, (item,))

    @_ExtensionsSpecialForm
    def NotRequired(self, parameters):
        if False:
            for i in range(10):
                print('nop')
        "A special typing construct to mark a key of a TypedDict as\n        potentially missing. For example:\n\n            class Movie(TypedDict):\n                title: str\n                year: NotRequired[int]\n\n            m = Movie(\n                title='The Matrix',  # typechecker error if key is omitted\n                year=1999,\n            )\n        "
        item = typing._type_check(parameters, f'{self._name} accepts only a single type.')
        return typing._GenericAlias(self, (item,))
else:

    class _RequiredForm(_ExtensionsSpecialForm, _root=True):

        def __getitem__(self, parameters):
            if False:
                i = 10
                return i + 15
            item = typing._type_check(parameters, f'{self._name} accepts only a single type.')
            return typing._GenericAlias(self, (item,))
    Required = _RequiredForm('Required', doc="A special typing construct to mark a key of a total=False TypedDict\n        as required. For example:\n\n            class Movie(TypedDict, total=False):\n                title: Required[str]\n                year: int\n\n            m = Movie(\n                title='The Matrix',  # typechecker error if key is omitted\n                year=1999,\n            )\n\n        There is no runtime checking that a required key is actually provided\n        when instantiating a related TypedDict.\n        ")
    NotRequired = _RequiredForm('NotRequired', doc="A special typing construct to mark a key of a TypedDict as\n        potentially missing. For example:\n\n            class Movie(TypedDict):\n                title: str\n                year: NotRequired[int]\n\n            m = Movie(\n                title='The Matrix',  # typechecker error if key is omitted\n                year=1999,\n            )\n        ")
_UNPACK_DOC = "Type unpack operator.\n\nThe type unpack operator takes the child types from some container type,\nsuch as `tuple[int, str]` or a `TypeVarTuple`, and 'pulls them out'. For\nexample:\n\n  # For some generic class `Foo`:\n  Foo[Unpack[tuple[int, str]]]  # Equivalent to Foo[int, str]\n\n  Ts = TypeVarTuple('Ts')\n  # Specifies that `Bar` is generic in an arbitrary number of types.\n  # (Think of `Ts` as a tuple of an arbitrary number of individual\n  #  `TypeVar`s, which the `Unpack` is 'pulling out' directly into the\n  #  `Generic[]`.)\n  class Bar(Generic[Unpack[Ts]]): ...\n  Bar[int]  # Valid\n  Bar[int, str]  # Also valid\n\nFrom Python 3.11, this can also be done using the `*` operator:\n\n    Foo[*tuple[int, str]]\n    class Bar(Generic[*Ts]): ...\n\nThe operator can also be used along with a `TypedDict` to annotate\n`**kwargs` in a function signature. For instance:\n\n  class Movie(TypedDict):\n    name: str\n    year: int\n\n  # This function expects two keyword arguments - *name* of type `str` and\n  # *year* of type `int`.\n  def foo(**kwargs: Unpack[Movie]): ...\n\nNote that there is only some runtime checking of this operator. Not\neverything the runtime allows may be accepted by static type checkers.\n\nFor more information, see PEP 646 and PEP 692.\n"
if sys.version_info >= (3, 12):
    Unpack = typing.Unpack

    def _is_unpack(obj):
        if False:
            print('Hello World!')
        return get_origin(obj) is Unpack
elif sys.version_info[:2] >= (3, 9):

    class _UnpackSpecialForm(_ExtensionsSpecialForm, _root=True):

        def __init__(self, getitem):
            if False:
                return 10
            super().__init__(getitem)
            self.__doc__ = _UNPACK_DOC

    class _UnpackAlias(typing._GenericAlias, _root=True):
        __class__ = typing.TypeVar

    @_UnpackSpecialForm
    def Unpack(self, parameters):
        if False:
            while True:
                i = 10
        item = typing._type_check(parameters, f'{self._name} accepts only a single type.')
        return _UnpackAlias(self, (item,))

    def _is_unpack(obj):
        if False:
            while True:
                i = 10
        return isinstance(obj, _UnpackAlias)
else:

    class _UnpackAlias(typing._GenericAlias, _root=True):
        __class__ = typing.TypeVar

    class _UnpackForm(_ExtensionsSpecialForm, _root=True):

        def __getitem__(self, parameters):
            if False:
                i = 10
                return i + 15
            item = typing._type_check(parameters, f'{self._name} accepts only a single type.')
            return _UnpackAlias(self, (item,))
    Unpack = _UnpackForm('Unpack', doc=_UNPACK_DOC)

    def _is_unpack(obj):
        if False:
            i = 10
            return i + 15
        return isinstance(obj, _UnpackAlias)
if hasattr(typing, 'TypeVarTuple'):

    class TypeVarTuple(metaclass=_TypeVarLikeMeta):
        """Type variable tuple."""
        _backported_typevarlike = typing.TypeVarTuple

        def __new__(cls, name, *, default=_marker):
            if False:
                return 10
            tvt = typing.TypeVarTuple(name)
            _set_default(tvt, default)
            _set_module(tvt)
            return tvt

        def __init_subclass__(self, *args, **kwds):
            if False:
                print('Hello World!')
            raise TypeError('Cannot subclass special typing classes')
else:

    class TypeVarTuple(_DefaultMixin):
        """Type variable tuple.

        Usage::

            Ts = TypeVarTuple('Ts')

        In the same way that a normal type variable is a stand-in for a single
        type such as ``int``, a type variable *tuple* is a stand-in for a *tuple*
        type such as ``Tuple[int, str]``.

        Type variable tuples can be used in ``Generic`` declarations.
        Consider the following example::

            class Array(Generic[*Ts]): ...

        The ``Ts`` type variable tuple here behaves like ``tuple[T1, T2]``,
        where ``T1`` and ``T2`` are type variables. To use these type variables
        as type parameters of ``Array``, we must *unpack* the type variable tuple using
        the star operator: ``*Ts``. The signature of ``Array`` then behaves
        as if we had simply written ``class Array(Generic[T1, T2]): ...``.
        In contrast to ``Generic[T1, T2]``, however, ``Generic[*Shape]`` allows
        us to parameterise the class with an *arbitrary* number of type parameters.

        Type variable tuples can be used anywhere a normal ``TypeVar`` can.
        This includes class definitions, as shown above, as well as function
        signatures and variable annotations::

            class Array(Generic[*Ts]):

                def __init__(self, shape: Tuple[*Ts]):
                    self._shape: Tuple[*Ts] = shape

                def get_shape(self) -> Tuple[*Ts]:
                    return self._shape

            shape = (Height(480), Width(640))
            x: Array[Height, Width] = Array(shape)
            y = abs(x)  # Inferred type is Array[Height, Width]
            z = x + x   #        ...    is Array[Height, Width]
            x.get_shape()  #     ...    is tuple[Height, Width]

        """
        __class__ = typing.TypeVar

        def __iter__(self):
            if False:
                i = 10
                return i + 15
            yield self.__unpacked__

        def __init__(self, name, *, default=_marker):
            if False:
                for i in range(10):
                    print('nop')
            self.__name__ = name
            _DefaultMixin.__init__(self, default)
            def_mod = _caller()
            if def_mod != 'typing_extensions':
                self.__module__ = def_mod
            self.__unpacked__ = Unpack[self]

        def __repr__(self):
            if False:
                i = 10
                return i + 15
            return self.__name__

        def __hash__(self):
            if False:
                return 10
            return object.__hash__(self)

        def __eq__(self, other):
            if False:
                while True:
                    i = 10
            return self is other

        def __reduce__(self):
            if False:
                print('Hello World!')
            return self.__name__

        def __init_subclass__(self, *args, **kwds):
            if False:
                while True:
                    i = 10
            if '_root' not in kwds:
                raise TypeError('Cannot subclass special typing classes')
if hasattr(typing, 'reveal_type'):
    reveal_type = typing.reveal_type
else:

    def reveal_type(__obj: T) -> T:
        if False:
            print('Hello World!')
        'Reveal the inferred type of a variable.\n\n        When a static type checker encounters a call to ``reveal_type()``,\n        it will emit the inferred type of the argument::\n\n            x: int = 1\n            reveal_type(x)\n\n        Running a static type checker (e.g., ``mypy``) on this example\n        will produce output similar to \'Revealed type is "builtins.int"\'.\n\n        At runtime, the function prints the runtime type of the\n        argument and returns it unchanged.\n\n        '
        print(f'Runtime type is {type(__obj).__name__!r}', file=sys.stderr)
        return __obj
if hasattr(typing, 'assert_never'):
    assert_never = typing.assert_never
else:

    def assert_never(__arg: Never) -> Never:
        if False:
            print('Hello World!')
        'Assert to the type checker that a line of code is unreachable.\n\n        Example::\n\n            def int_or_str(arg: int | str) -> None:\n                match arg:\n                    case int():\n                        print("It\'s an int")\n                    case str():\n                        print("It\'s a str")\n                    case _:\n                        assert_never(arg)\n\n        If a type checker finds that a call to assert_never() is\n        reachable, it will emit an error.\n\n        At runtime, this throws an exception when called.\n\n        '
        raise AssertionError('Expected code to be unreachable')
if sys.version_info >= (3, 12):
    dataclass_transform = typing.dataclass_transform
else:

    def dataclass_transform(*, eq_default: bool=True, order_default: bool=False, kw_only_default: bool=False, frozen_default: bool=False, field_specifiers: typing.Tuple[typing.Union[typing.Type[typing.Any], typing.Callable[..., typing.Any]], ...]=(), **kwargs: typing.Any) -> typing.Callable[[T], T]:
        if False:
            print('Hello World!')
        'Decorator that marks a function, class, or metaclass as providing\n        dataclass-like behavior.\n\n        Example:\n\n            from pip._vendor.typing_extensions import dataclass_transform\n\n            _T = TypeVar("_T")\n\n            # Used on a decorator function\n            @dataclass_transform()\n            def create_model(cls: type[_T]) -> type[_T]:\n                ...\n                return cls\n\n            @create_model\n            class CustomerModel:\n                id: int\n                name: str\n\n            # Used on a base class\n            @dataclass_transform()\n            class ModelBase: ...\n\n            class CustomerModel(ModelBase):\n                id: int\n                name: str\n\n            # Used on a metaclass\n            @dataclass_transform()\n            class ModelMeta(type): ...\n\n            class ModelBase(metaclass=ModelMeta): ...\n\n            class CustomerModel(ModelBase):\n                id: int\n                name: str\n\n        Each of the ``CustomerModel`` classes defined in this example will now\n        behave similarly to a dataclass created with the ``@dataclasses.dataclass``\n        decorator. For example, the type checker will synthesize an ``__init__``\n        method.\n\n        The arguments to this decorator can be used to customize this behavior:\n        - ``eq_default`` indicates whether the ``eq`` parameter is assumed to be\n          True or False if it is omitted by the caller.\n        - ``order_default`` indicates whether the ``order`` parameter is\n          assumed to be True or False if it is omitted by the caller.\n        - ``kw_only_default`` indicates whether the ``kw_only`` parameter is\n          assumed to be True or False if it is omitted by the caller.\n        - ``frozen_default`` indicates whether the ``frozen`` parameter is\n          assumed to be True or False if it is omitted by the caller.\n        - ``field_specifiers`` specifies a static list of supported classes\n          or functions that describe fields, similar to ``dataclasses.field()``.\n\n        At runtime, this decorator records its arguments in the\n        ``__dataclass_transform__`` attribute on the decorated object.\n\n        See PEP 681 for details.\n\n        '

        def decorator(cls_or_fn):
            if False:
                while True:
                    i = 10
            cls_or_fn.__dataclass_transform__ = {'eq_default': eq_default, 'order_default': order_default, 'kw_only_default': kw_only_default, 'frozen_default': frozen_default, 'field_specifiers': field_specifiers, 'kwargs': kwargs}
            return cls_or_fn
        return decorator
if hasattr(typing, 'override'):
    override = typing.override
else:
    _F = typing.TypeVar('_F', bound=typing.Callable[..., typing.Any])

    def override(__arg: _F) -> _F:
        if False:
            i = 10
            return i + 15
        'Indicate that a method is intended to override a method in a base class.\n\n        Usage:\n\n            class Base:\n                def method(self) -> None: ...\n                    pass\n\n            class Child(Base):\n                @override\n                def method(self) -> None:\n                    super().method()\n\n        When this decorator is applied to a method, the type checker will\n        validate that it overrides a method with the same name on a base class.\n        This helps prevent bugs that may occur when a base class is changed\n        without an equivalent change to a child class.\n\n        There is no runtime checking of these properties. The decorator\n        sets the ``__override__`` attribute to ``True`` on the decorated object\n        to allow runtime introspection.\n\n        See PEP 698 for details.\n\n        '
        try:
            __arg.__override__ = True
        except (AttributeError, TypeError):
            pass
        return __arg
if hasattr(typing, 'deprecated'):
    deprecated = typing.deprecated
else:
    _T = typing.TypeVar('_T')

    def deprecated(__msg: str, *, category: typing.Optional[typing.Type[Warning]]=DeprecationWarning, stacklevel: int=1) -> typing.Callable[[_T], _T]:
        if False:
            return 10
        'Indicate that a class, function or overload is deprecated.\n\n        Usage:\n\n            @deprecated("Use B instead")\n            class A:\n                pass\n\n            @deprecated("Use g instead")\n            def f():\n                pass\n\n            @overload\n            @deprecated("int support is deprecated")\n            def g(x: int) -> int: ...\n            @overload\n            def g(x: str) -> int: ...\n\n        When this decorator is applied to an object, the type checker\n        will generate a diagnostic on usage of the deprecated object.\n\n        The warning specified by ``category`` will be emitted on use\n        of deprecated objects. For functions, that happens on calls;\n        for classes, on instantiation. If the ``category`` is ``None``,\n        no warning is emitted. The ``stacklevel`` determines where the\n        warning is emitted. If it is ``1`` (the default), the warning\n        is emitted at the direct caller of the deprecated object; if it\n        is higher, it is emitted further up the stack.\n\n        The decorator sets the ``__deprecated__``\n        attribute on the decorated object to the deprecation message\n        passed to the decorator. If applied to an overload, the decorator\n        must be after the ``@overload`` decorator for the attribute to\n        exist on the overload as returned by ``get_overloads()``.\n\n        See PEP 702 for details.\n\n        '

        def decorator(__arg: _T) -> _T:
            if False:
                while True:
                    i = 10
            if category is None:
                __arg.__deprecated__ = __msg
                return __arg
            elif isinstance(__arg, type):
                original_new = __arg.__new__
                has_init = __arg.__init__ is not object.__init__

                @functools.wraps(original_new)
                def __new__(cls, *args, **kwargs):
                    if False:
                        return 10
                    warnings.warn(__msg, category=category, stacklevel=stacklevel + 1)
                    if original_new is not object.__new__:
                        return original_new(cls, *args, **kwargs)
                    elif not has_init and (args or kwargs):
                        raise TypeError(f'{cls.__name__}() takes no arguments')
                    else:
                        return original_new(cls)
                __arg.__new__ = staticmethod(__new__)
                __arg.__deprecated__ = __new__.__deprecated__ = __msg
                return __arg
            elif callable(__arg):

                @functools.wraps(__arg)
                def wrapper(*args, **kwargs):
                    if False:
                        while True:
                            i = 10
                    warnings.warn(__msg, category=category, stacklevel=stacklevel + 1)
                    return __arg(*args, **kwargs)
                __arg.__deprecated__ = wrapper.__deprecated__ = __msg
                return wrapper
            else:
                raise TypeError(f'@deprecated decorator with non-None category must be applied to a class or callable, not {__arg!r}')
        return decorator
if not hasattr(typing, 'TypeVarTuple'):
    typing._collect_type_vars = _collect_type_vars
    typing._check_generic = _check_generic
if sys.version_info >= (3, 13):
    NamedTuple = typing.NamedTuple
else:

    def _make_nmtuple(name, types, module, defaults=()):
        if False:
            for i in range(10):
                print('nop')
        fields = [n for (n, t) in types]
        annotations = {n: typing._type_check(t, f'field {n} annotation must be a type') for (n, t) in types}
        nm_tpl = collections.namedtuple(name, fields, defaults=defaults, module=module)
        nm_tpl.__annotations__ = nm_tpl.__new__.__annotations__ = annotations
        if sys.version_info < (3, 9):
            nm_tpl._field_types = annotations
        return nm_tpl
    _prohibited_namedtuple_fields = typing._prohibited
    _special_namedtuple_fields = frozenset({'__module__', '__name__', '__annotations__'})

    class _NamedTupleMeta(type):

        def __new__(cls, typename, bases, ns):
            if False:
                print('Hello World!')
            assert _NamedTuple in bases
            for base in bases:
                if base is not _NamedTuple and base is not typing.Generic:
                    raise TypeError('can only inherit from a NamedTuple type and Generic')
            bases = tuple((tuple if base is _NamedTuple else base for base in bases))
            types = ns.get('__annotations__', {})
            default_names = []
            for field_name in types:
                if field_name in ns:
                    default_names.append(field_name)
                elif default_names:
                    raise TypeError(f"Non-default namedtuple field {field_name} cannot follow default field{('s' if len(default_names) > 1 else '')} {', '.join(default_names)}")
            nm_tpl = _make_nmtuple(typename, types.items(), defaults=[ns[n] for n in default_names], module=ns['__module__'])
            nm_tpl.__bases__ = bases
            if typing.Generic in bases:
                if hasattr(typing, '_generic_class_getitem'):
                    nm_tpl.__class_getitem__ = classmethod(typing._generic_class_getitem)
                else:
                    class_getitem = typing.Generic.__class_getitem__.__func__
                    nm_tpl.__class_getitem__ = classmethod(class_getitem)
            for key in ns:
                if key in _prohibited_namedtuple_fields:
                    raise AttributeError('Cannot overwrite NamedTuple attribute ' + key)
                elif key not in _special_namedtuple_fields and key not in nm_tpl._fields:
                    setattr(nm_tpl, key, ns[key])
            if typing.Generic in bases:
                nm_tpl.__init_subclass__()
            return nm_tpl
    _NamedTuple = type.__new__(_NamedTupleMeta, 'NamedTuple', (), {})

    def _namedtuple_mro_entries(bases):
        if False:
            for i in range(10):
                print('nop')
        assert NamedTuple in bases
        return (_NamedTuple,)

    @_ensure_subclassable(_namedtuple_mro_entries)
    def NamedTuple(__typename, __fields=_marker, **kwargs):
        if False:
            i = 10
            return i + 15
        "Typed version of namedtuple.\n\n        Usage::\n\n            class Employee(NamedTuple):\n                name: str\n                id: int\n\n        This is equivalent to::\n\n            Employee = collections.namedtuple('Employee', ['name', 'id'])\n\n        The resulting class has an extra __annotations__ attribute, giving a\n        dict that maps field names to types.  (The field names are also in\n        the _fields attribute, which is part of the namedtuple API.)\n        An alternative equivalent functional syntax is also accepted::\n\n            Employee = NamedTuple('Employee', [('name', str), ('id', int)])\n        "
        if __fields is _marker:
            if kwargs:
                deprecated_thing = 'Creating NamedTuple classes using keyword arguments'
                deprecation_msg = '{name} is deprecated and will be disallowed in Python {remove}. Use the class-based or functional syntax instead.'
            else:
                deprecated_thing = "Failing to pass a value for the 'fields' parameter"
                example = f'`{__typename} = NamedTuple({__typename!r}, [])`'
                deprecation_msg = '{name} is deprecated and will be disallowed in Python {remove}. To create a NamedTuple class with 0 fields using the functional syntax, pass an empty list, e.g. ' + example + '.'
        elif __fields is None:
            if kwargs:
                raise TypeError("Cannot pass `None` as the 'fields' parameter and also specify fields using keyword arguments")
            else:
                deprecated_thing = "Passing `None` as the 'fields' parameter"
                example = f'`{__typename} = NamedTuple({__typename!r}, [])`'
                deprecation_msg = '{name} is deprecated and will be disallowed in Python {remove}. To create a NamedTuple class with 0 fields using the functional syntax, pass an empty list, e.g. ' + example + '.'
        elif kwargs:
            raise TypeError('Either list of fields or keywords can be provided to NamedTuple, not both')
        if __fields is _marker or __fields is None:
            warnings.warn(deprecation_msg.format(name=deprecated_thing, remove='3.15'), DeprecationWarning, stacklevel=2)
            __fields = kwargs.items()
        nt = _make_nmtuple(__typename, __fields, module=_caller())
        nt.__orig_bases__ = (NamedTuple,)
        return nt
    if sys.version_info >= (3, 8):
        _new_signature = '(typename, fields=None, /, **kwargs)'
        if isinstance(NamedTuple, _types.FunctionType):
            NamedTuple.__text_signature__ = _new_signature
        else:
            NamedTuple.__call__.__text_signature__ = _new_signature
if hasattr(collections.abc, 'Buffer'):
    Buffer = collections.abc.Buffer
else:

    class Buffer(abc.ABC):
        """Base class for classes that implement the buffer protocol.

        The buffer protocol allows Python objects to expose a low-level
        memory buffer interface. Before Python 3.12, it is not possible
        to implement the buffer protocol in pure Python code, or even
        to check whether a class implements the buffer protocol. In
        Python 3.12 and higher, the ``__buffer__`` method allows access
        to the buffer protocol from Python code, and the
        ``collections.abc.Buffer`` ABC allows checking whether a class
        implements the buffer protocol.

        To indicate support for the buffer protocol in earlier versions,
        inherit from this ABC, either in a stub file or at runtime,
        or use ABC registration. This ABC provides no methods, because
        there is no Python-accessible methods shared by pre-3.12 buffer
        classes. It is useful primarily for static checks.

        """
    Buffer.register(memoryview)
    Buffer.register(bytearray)
    Buffer.register(bytes)
if hasattr(_types, 'get_original_bases'):
    get_original_bases = _types.get_original_bases
else:

    def get_original_bases(__cls):
        if False:
            while True:
                i = 10
        'Return the class\'s "original" bases prior to modification by `__mro_entries__`.\n\n        Examples::\n\n            from typing import TypeVar, Generic\n            from pip._vendor.typing_extensions import NamedTuple, TypedDict\n\n            T = TypeVar("T")\n            class Foo(Generic[T]): ...\n            class Bar(Foo[int], float): ...\n            class Baz(list[str]): ...\n            Eggs = NamedTuple("Eggs", [("a", int), ("b", str)])\n            Spam = TypedDict("Spam", {"a": int, "b": str})\n\n            assert get_original_bases(Bar) == (Foo[int], float)\n            assert get_original_bases(Baz) == (list[str],)\n            assert get_original_bases(Eggs) == (NamedTuple,)\n            assert get_original_bases(Spam) == (TypedDict,)\n            assert get_original_bases(int) == (object,)\n        '
        try:
            return __cls.__orig_bases__
        except AttributeError:
            try:
                return __cls.__bases__
            except AttributeError:
                raise TypeError(f'Expected an instance of type, not {type(__cls).__name__!r}') from None
if sys.version_info >= (3, 11):
    NewType = typing.NewType
else:

    class NewType:
        """NewType creates simple unique types with almost zero
        runtime overhead. NewType(name, tp) is considered a subtype of tp
        by static type checkers. At runtime, NewType(name, tp) returns
        a dummy callable that simply returns its argument. Usage::
            UserId = NewType('UserId', int)
            def name_by_id(user_id: UserId) -> str:
                ...
            UserId('user')          # Fails type check
            name_by_id(42)          # Fails type check
            name_by_id(UserId(42))  # OK
            num = UserId(5) + 1     # type: int
        """

        def __call__(self, obj):
            if False:
                print('Hello World!')
            return obj

        def __init__(self, name, tp):
            if False:
                print('Hello World!')
            self.__qualname__ = name
            if '.' in name:
                name = name.rpartition('.')[-1]
            self.__name__ = name
            self.__supertype__ = tp
            def_mod = _caller()
            if def_mod != 'typing_extensions':
                self.__module__ = def_mod

        def __mro_entries__(self, bases):
            if False:
                while True:
                    i = 10
            supercls_name = self.__name__

            class Dummy:

                def __init_subclass__(cls):
                    if False:
                        for i in range(10):
                            print('nop')
                    subcls_name = cls.__name__
                    raise TypeError(f'Cannot subclass an instance of NewType. Perhaps you were looking for: `{subcls_name} = NewType({subcls_name!r}, {supercls_name})`')
            return (Dummy,)

        def __repr__(self):
            if False:
                i = 10
                return i + 15
            return f'{self.__module__}.{self.__qualname__}'

        def __reduce__(self):
            if False:
                print('Hello World!')
            return self.__qualname__
        if sys.version_info >= (3, 10):

            def __or__(self, other):
                if False:
                    while True:
                        i = 10
                return typing.Union[self, other]

            def __ror__(self, other):
                if False:
                    while True:
                        i = 10
                return typing.Union[other, self]
if hasattr(typing, 'TypeAliasType'):
    TypeAliasType = typing.TypeAliasType
else:

    def _is_unionable(obj):
        if False:
            i = 10
            return i + 15
        'Corresponds to is_unionable() in unionobject.c in CPython.'
        return obj is None or isinstance(obj, (type, _types.GenericAlias, _types.UnionType, TypeAliasType))

    class TypeAliasType:
        """Create named, parameterized type aliases.

        This provides a backport of the new `type` statement in Python 3.12:

            type ListOrSet[T] = list[T] | set[T]

        is equivalent to:

            T = TypeVar("T")
            ListOrSet = TypeAliasType("ListOrSet", list[T] | set[T], type_params=(T,))

        The name ListOrSet can then be used as an alias for the type it refers to.

        The type_params argument should contain all the type parameters used
        in the value of the type alias. If the alias is not generic, this
        argument is omitted.

        Static type checkers should only support type aliases declared using
        TypeAliasType that follow these rules:

        - The first argument (the name) must be a string literal.
        - The TypeAliasType instance must be immediately assigned to a variable
          of the same name. (For example, 'X = TypeAliasType("Y", int)' is invalid,
          as is 'X, Y = TypeAliasType("X", int), TypeAliasType("Y", int)').

        """

        def __init__(self, name: str, value, *, type_params=()):
            if False:
                return 10
            if not isinstance(name, str):
                raise TypeError('TypeAliasType name must be a string')
            self.__value__ = value
            self.__type_params__ = type_params
            parameters = []
            for type_param in type_params:
                if isinstance(type_param, TypeVarTuple):
                    parameters.extend(type_param)
                else:
                    parameters.append(type_param)
            self.__parameters__ = tuple(parameters)
            def_mod = _caller()
            if def_mod != 'typing_extensions':
                self.__module__ = def_mod
            self.__name__ = name

        def __setattr__(self, __name: str, __value: object) -> None:
            if False:
                while True:
                    i = 10
            if hasattr(self, '__name__'):
                self._raise_attribute_error(__name)
            super().__setattr__(__name, __value)

        def __delattr__(self, __name: str) -> Never:
            if False:
                return 10
            self._raise_attribute_error(__name)

        def _raise_attribute_error(self, name: str) -> Never:
            if False:
                i = 10
                return i + 15
            if name == '__name__':
                raise AttributeError('readonly attribute')
            elif name in {'__value__', '__type_params__', '__parameters__', '__module__'}:
                raise AttributeError(f"attribute '{name}' of 'typing.TypeAliasType' objects is not writable")
            else:
                raise AttributeError(f"'typing.TypeAliasType' object has no attribute '{name}'")

        def __repr__(self) -> str:
            if False:
                return 10
            return self.__name__

        def __getitem__(self, parameters):
            if False:
                for i in range(10):
                    print('nop')
            if not isinstance(parameters, tuple):
                parameters = (parameters,)
            parameters = [typing._type_check(item, f'Subscripting {self.__name__} requires a type.') for item in parameters]
            return typing._GenericAlias(self, tuple(parameters))

        def __reduce__(self):
            if False:
                while True:
                    i = 10
            return self.__name__

        def __init_subclass__(cls, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            raise TypeError("type 'typing_extensions.TypeAliasType' is not an acceptable base type")

        def __call__(self):
            if False:
                i = 10
                return i + 15
            raise TypeError('Type alias is not callable')
        if sys.version_info >= (3, 10):

            def __or__(self, right):
                if False:
                    print('Hello World!')
                if not _is_unionable(right):
                    return NotImplemented
                return typing.Union[self, right]

            def __ror__(self, left):
                if False:
                    return 10
                if not _is_unionable(left):
                    return NotImplemented
                return typing.Union[left, self]
if hasattr(typing, 'is_protocol'):
    is_protocol = typing.is_protocol
    get_protocol_members = typing.get_protocol_members
else:

    def is_protocol(__tp: type) -> bool:
        if False:
            while True:
                i = 10
        'Return True if the given type is a Protocol.\n\n        Example::\n\n            >>> from typing_extensions import Protocol, is_protocol\n            >>> class P(Protocol):\n            ...     def a(self) -> str: ...\n            ...     b: int\n            >>> is_protocol(P)\n            True\n            >>> is_protocol(int)\n            False\n        '
        return isinstance(__tp, type) and getattr(__tp, '_is_protocol', False) and (__tp is not Protocol) and (__tp is not getattr(typing, 'Protocol', object()))

    def get_protocol_members(__tp: type) -> typing.FrozenSet[str]:
        if False:
            while True:
                i = 10
        "Return the set of members defined in a Protocol.\n\n        Example::\n\n            >>> from typing_extensions import Protocol, get_protocol_members\n            >>> class P(Protocol):\n            ...     def a(self) -> str: ...\n            ...     b: int\n            >>> get_protocol_members(P)\n            frozenset({'a', 'b'})\n\n        Raise a TypeError for arguments that are not Protocols.\n        "
        if not is_protocol(__tp):
            raise TypeError(f'{__tp!r} is not a Protocol')
        if hasattr(__tp, '__protocol_attrs__'):
            return frozenset(__tp.__protocol_attrs__)
        return frozenset(_get_protocol_attrs(__tp))
AbstractSet = typing.AbstractSet
AnyStr = typing.AnyStr
BinaryIO = typing.BinaryIO
Callable = typing.Callable
Collection = typing.Collection
Container = typing.Container
Dict = typing.Dict
ForwardRef = typing.ForwardRef
FrozenSet = typing.FrozenSet
Generator = typing.Generator
Generic = typing.Generic
Hashable = typing.Hashable
IO = typing.IO
ItemsView = typing.ItemsView
Iterable = typing.Iterable
Iterator = typing.Iterator
KeysView = typing.KeysView
List = typing.List
Mapping = typing.Mapping
MappingView = typing.MappingView
Match = typing.Match
MutableMapping = typing.MutableMapping
MutableSequence = typing.MutableSequence
MutableSet = typing.MutableSet
Optional = typing.Optional
Pattern = typing.Pattern
Reversible = typing.Reversible
Sequence = typing.Sequence
Set = typing.Set
Sized = typing.Sized
TextIO = typing.TextIO
Tuple = typing.Tuple
Union = typing.Union
ValuesView = typing.ValuesView
cast = typing.cast
no_type_check = typing.no_type_check
no_type_check_decorator = typing.no_type_check_decorator