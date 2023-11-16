"""Abstract Base Classes (ABCs) for collections, according to PEP 3119.

Unit tests are in test_collections.
"""
from abc import ABCMeta, abstractmethod
import sys
GenericAlias = type(list[int])
EllipsisType = type(...)

def _f():
    if False:
        return 10
    pass
FunctionType = type(_f)
del _f
__all__ = ['Awaitable', 'Coroutine', 'AsyncIterable', 'AsyncIterator', 'AsyncGenerator', 'Hashable', 'Iterable', 'Iterator', 'Generator', 'Reversible', 'Sized', 'Container', 'Callable', 'Collection', 'Set', 'MutableSet', 'Mapping', 'MutableMapping', 'MappingView', 'KeysView', 'ItemsView', 'ValuesView', 'Sequence', 'MutableSequence', 'ByteString']
__name__ = 'collections.abc'
bytes_iterator = type(iter(b''))
bytearray_iterator = type(iter(bytearray()))
dict_keyiterator = type(iter({}.keys()))
dict_valueiterator = type(iter({}.values()))
dict_itemiterator = type(iter({}.items()))
list_iterator = type(iter([]))
list_reverseiterator = type(iter(reversed([])))
range_iterator = type(iter(range(0)))
longrange_iterator = type(iter(range(1 << 1000)))
set_iterator = type(iter(set()))
str_iterator = type(iter(''))
tuple_iterator = type(iter(()))
zip_iterator = type(iter(zip()))
dict_keys = type({}.keys())
dict_values = type({}.values())
dict_items = type({}.items())
mappingproxy = type(type.__dict__)
generator = type((lambda : (yield))())

async def _coro():
    pass
_coro = _coro()
coroutine = type(_coro)
_coro.close()
del _coro

async def _ag():
    yield
_ag = _ag()
async_generator = type(_ag)
del _ag

def _check_methods(C, *methods):
    if False:
        for i in range(10):
            print('nop')
    mro = C.__mro__
    for method in methods:
        for B in mro:
            if method in B.__dict__:
                if B.__dict__[method] is None:
                    return NotImplemented
                break
        else:
            return NotImplemented
    return True

class Hashable(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def __hash__(self):
        if False:
            while True:
                i = 10
        return 0

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            i = 10
            return i + 15
        if cls is Hashable:
            return _check_methods(C, '__hash__')
        return NotImplemented

class Awaitable(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def __await__(self):
        if False:
            print('Hello World!')
        yield

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            print('Hello World!')
        if cls is Awaitable:
            return _check_methods(C, '__await__')
        return NotImplemented
    __class_getitem__ = classmethod(GenericAlias)

class Coroutine(Awaitable):
    __slots__ = ()

    @abstractmethod
    def send(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Send a value into the coroutine.\n        Return next yielded value or raise StopIteration.\n        '
        raise StopIteration

    @abstractmethod
    def throw(self, typ, val=None, tb=None):
        if False:
            for i in range(10):
                print('nop')
        'Raise an exception in the coroutine.\n        Return next yielded value or raise StopIteration.\n        '
        if val is None:
            if tb is None:
                raise typ
            val = typ()
        if tb is not None:
            val = val.with_traceback(tb)
        raise val

    def close(self):
        if False:
            return 10
        'Raise GeneratorExit inside coroutine.\n        '
        try:
            self.throw(GeneratorExit)
        except (GeneratorExit, StopIteration):
            pass
        else:
            raise RuntimeError('coroutine ignored GeneratorExit')

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            return 10
        if cls is Coroutine:
            return _check_methods(C, '__await__', 'send', 'throw', 'close')
        return NotImplemented
Coroutine.register(coroutine)

class AsyncIterable(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def __aiter__(self):
        if False:
            while True:
                i = 10
        return AsyncIterator()

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            for i in range(10):
                print('nop')
        if cls is AsyncIterable:
            return _check_methods(C, '__aiter__')
        return NotImplemented
    __class_getitem__ = classmethod(GenericAlias)

class AsyncIterator(AsyncIterable):
    __slots__ = ()

    @abstractmethod
    async def __anext__(self):
        """Return the next item or raise StopAsyncIteration when exhausted."""
        raise StopAsyncIteration

    def __aiter__(self):
        if False:
            print('Hello World!')
        return self

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            i = 10
            return i + 15
        if cls is AsyncIterator:
            return _check_methods(C, '__anext__', '__aiter__')
        return NotImplemented

class AsyncGenerator(AsyncIterator):
    __slots__ = ()

    async def __anext__(self):
        """Return the next item from the asynchronous generator.
        When exhausted, raise StopAsyncIteration.
        """
        return await self.asend(None)

    @abstractmethod
    async def asend(self, value):
        """Send a value into the asynchronous generator.
        Return next yielded value or raise StopAsyncIteration.
        """
        raise StopAsyncIteration

    @abstractmethod
    async def athrow(self, typ, val=None, tb=None):
        """Raise an exception in the asynchronous generator.
        Return next yielded value or raise StopAsyncIteration.
        """
        if val is None:
            if tb is None:
                raise typ
            val = typ()
        if tb is not None:
            val = val.with_traceback(tb)
        raise val

    async def aclose(self):
        """Raise GeneratorExit inside coroutine.
        """
        try:
            await self.athrow(GeneratorExit)
        except (GeneratorExit, StopAsyncIteration):
            pass
        else:
            raise RuntimeError('asynchronous generator ignored GeneratorExit')

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            return 10
        if cls is AsyncGenerator:
            return _check_methods(C, '__aiter__', '__anext__', 'asend', 'athrow', 'aclose')
        return NotImplemented
AsyncGenerator.register(async_generator)

class Iterable(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def __iter__(self):
        if False:
            i = 10
            return i + 15
        while False:
            yield None

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            while True:
                i = 10
        if cls is Iterable:
            return _check_methods(C, '__iter__')
        return NotImplemented
    __class_getitem__ = classmethod(GenericAlias)

class Iterator(Iterable):
    __slots__ = ()

    @abstractmethod
    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the next item from the iterator. When exhausted, raise StopIteration'
        raise StopIteration

    def __iter__(self):
        if False:
            print('Hello World!')
        return self

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            for i in range(10):
                print('nop')
        if cls is Iterator:
            return _check_methods(C, '__iter__', '__next__')
        return NotImplemented
Iterator.register(bytes_iterator)
Iterator.register(bytearray_iterator)
Iterator.register(dict_keyiterator)
Iterator.register(dict_valueiterator)
Iterator.register(dict_itemiterator)
Iterator.register(list_iterator)
Iterator.register(list_reverseiterator)
Iterator.register(range_iterator)
Iterator.register(longrange_iterator)
Iterator.register(set_iterator)
Iterator.register(str_iterator)
Iterator.register(tuple_iterator)
Iterator.register(zip_iterator)

class Reversible(Iterable):
    __slots__ = ()

    @abstractmethod
    def __reversed__(self):
        if False:
            print('Hello World!')
        while False:
            yield None

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            print('Hello World!')
        if cls is Reversible:
            return _check_methods(C, '__reversed__', '__iter__')
        return NotImplemented

class Generator(Iterator):
    __slots__ = ()

    def __next__(self):
        if False:
            return 10
        'Return the next item from the generator.\n        When exhausted, raise StopIteration.\n        '
        return self.send(None)

    @abstractmethod
    def send(self, value):
        if False:
            return 10
        'Send a value into the generator.\n        Return next yielded value or raise StopIteration.\n        '
        raise StopIteration

    @abstractmethod
    def throw(self, typ, val=None, tb=None):
        if False:
            while True:
                i = 10
        'Raise an exception in the generator.\n        Return next yielded value or raise StopIteration.\n        '
        if val is None:
            if tb is None:
                raise typ
            val = typ()
        if tb is not None:
            val = val.with_traceback(tb)
        raise val

    def close(self):
        if False:
            while True:
                i = 10
        'Raise GeneratorExit inside generator.\n        '
        try:
            self.throw(GeneratorExit)
        except (GeneratorExit, StopIteration):
            pass
        else:
            raise RuntimeError('generator ignored GeneratorExit')

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            while True:
                i = 10
        if cls is Generator:
            return _check_methods(C, '__iter__', '__next__', 'send', 'throw', 'close')
        return NotImplemented
Generator.register(generator)

class Sized(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def __len__(self):
        if False:
            return 10
        return 0

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            while True:
                i = 10
        if cls is Sized:
            return _check_methods(C, '__len__')
        return NotImplemented

class Container(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def __contains__(self, x):
        if False:
            while True:
                i = 10
        return False

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            i = 10
            return i + 15
        if cls is Container:
            return _check_methods(C, '__contains__')
        return NotImplemented
    __class_getitem__ = classmethod(GenericAlias)

class Collection(Sized, Iterable, Container):
    __slots__ = ()

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            print('Hello World!')
        if cls is Collection:
            return _check_methods(C, '__len__', '__iter__', '__contains__')
        return NotImplemented

class _CallableGenericAlias(GenericAlias):
    """ Represent `Callable[argtypes, resulttype]`.

    This sets ``__args__`` to a tuple containing the flattened ``argtypes``
    followed by ``resulttype``.

    Example: ``Callable[[int, str], float]`` sets ``__args__`` to
    ``(int, str, float)``.
    """
    __slots__ = ()

    def __new__(cls, origin, args):
        if False:
            return 10
        if not (isinstance(args, tuple) and len(args) == 2):
            raise TypeError('Callable must be used as Callable[[arg, ...], result].')
        (t_args, t_result) = args
        if isinstance(t_args, list):
            args = (*t_args, t_result)
        elif not _is_param_expr(t_args):
            raise TypeError(f'Expected a list of types, an ellipsis, ParamSpec, or Concatenate. Got {t_args}')
        return super().__new__(cls, origin, args)

    @property
    def __parameters__(self):
        if False:
            print('Hello World!')
        params = []
        for arg in self.__args__:
            if hasattr(arg, '__parameters__') and isinstance(arg.__parameters__, tuple):
                params.extend(arg.__parameters__)
            elif _is_typevarlike(arg):
                params.append(arg)
        return tuple(dict.fromkeys(params))

    def __repr__(self):
        if False:
            print('Hello World!')
        if len(self.__args__) == 2 and _is_param_expr(self.__args__[0]):
            return super().__repr__()
        return f"collections.abc.Callable[[{', '.join([_type_repr(a) for a in self.__args__[:-1]])}], {_type_repr(self.__args__[-1])}]"

    def __reduce__(self):
        if False:
            return 10
        args = self.__args__
        if not (len(args) == 2 and _is_param_expr(args[0])):
            args = (list(args[:-1]), args[-1])
        return (_CallableGenericAlias, (Callable, args))

    def __getitem__(self, item):
        if False:
            return 10
        param_len = len(self.__parameters__)
        if param_len == 0:
            raise TypeError(f'{self} is not a generic class')
        if not isinstance(item, tuple):
            item = (item,)
        if param_len == 1 and _is_param_expr(self.__parameters__[0]) and item and (not _is_param_expr(item[0])):
            item = (list(item),)
        item_len = len(item)
        if item_len != param_len:
            raise TypeError(f"Too {('many' if item_len > param_len else 'few')} arguments for {self}; actual {item_len}, expected {param_len}")
        subst = dict(zip(self.__parameters__, item))
        new_args = []
        for arg in self.__args__:
            if _is_typevarlike(arg):
                if _is_param_expr(arg):
                    arg = subst[arg]
                    if not _is_param_expr(arg):
                        raise TypeError(f'Expected a list of types, an ellipsis, ParamSpec, or Concatenate. Got {arg}')
                else:
                    arg = subst[arg]
            elif hasattr(arg, '__parameters__') and isinstance(arg.__parameters__, tuple):
                subparams = arg.__parameters__
                if subparams:
                    subargs = tuple((subst[x] for x in subparams))
                    arg = arg[subargs]
            if isinstance(arg, tuple):
                new_args.extend(arg)
            else:
                new_args.append(arg)
        if not isinstance(new_args[0], list):
            t_result = new_args[-1]
            t_args = new_args[:-1]
            new_args = (t_args, t_result)
        return _CallableGenericAlias(Callable, tuple(new_args))

def _is_typevarlike(arg):
    if False:
        for i in range(10):
            print('nop')
    obj = type(arg)
    return obj.__module__ == 'typing' and obj.__name__ in {'ParamSpec', 'TypeVar'}

def _is_param_expr(obj):
    if False:
        while True:
            i = 10
    'Checks if obj matches either a list of types, ``...``, ``ParamSpec`` or\n    ``_ConcatenateGenericAlias`` from typing.py\n    '
    if obj is Ellipsis:
        return True
    if isinstance(obj, list):
        return True
    obj = type(obj)
    names = ('ParamSpec', '_ConcatenateGenericAlias')
    return obj.__module__ == 'typing' and any((obj.__name__ == name for name in names))

def _type_repr(obj):
    if False:
        while True:
            i = 10
    "Return the repr() of an object, special-casing types (internal helper).\n\n    Copied from :mod:`typing` since collections.abc\n    shouldn't depend on that module.\n    "
    if isinstance(obj, GenericAlias):
        return repr(obj)
    if isinstance(obj, type):
        if obj.__module__ == 'builtins':
            return obj.__qualname__
        return f'{obj.__module__}.{obj.__qualname__}'
    if obj is Ellipsis:
        return '...'
    if isinstance(obj, FunctionType):
        return obj.__name__
    return repr(obj)

class Callable(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def __call__(self, *args, **kwds):
        if False:
            return 10
        return False

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            while True:
                i = 10
        if cls is Callable:
            return _check_methods(C, '__call__')
        return NotImplemented
    __class_getitem__ = classmethod(_CallableGenericAlias)

class Set(Collection):
    """A set is a finite, iterable container.

    This class provides concrete generic implementations of all
    methods except for __contains__, __iter__ and __len__.

    To override the comparisons (presumably for speed, as the
    semantics are fixed), redefine __le__ and __ge__,
    then the other operations will automatically follow suit.
    """
    __slots__ = ()

    def __le__(self, other):
        if False:
            return 10
        if not isinstance(other, Set):
            return NotImplemented
        if len(self) > len(other):
            return False
        for elem in self:
            if elem not in other:
                return False
        return True

    def __lt__(self, other):
        if False:
            return 10
        if not isinstance(other, Set):
            return NotImplemented
        return len(self) < len(other) and self.__le__(other)

    def __gt__(self, other):
        if False:
            return 10
        if not isinstance(other, Set):
            return NotImplemented
        return len(self) > len(other) and self.__ge__(other)

    def __ge__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, Set):
            return NotImplemented
        if len(self) < len(other):
            return False
        for elem in other:
            if elem not in self:
                return False
        return True

    def __eq__(self, other):
        if False:
            return 10
        if not isinstance(other, Set):
            return NotImplemented
        return len(self) == len(other) and self.__le__(other)

    @classmethod
    def _from_iterable(cls, it):
        if False:
            return 10
        'Construct an instance of the class from any iterable input.\n\n        Must override this method if the class constructor signature\n        does not accept an iterable for an input.\n        '
        return cls(it)

    def __and__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, Iterable):
            return NotImplemented
        return self._from_iterable((value for value in other if value in self))
    __rand__ = __and__

    def isdisjoint(self, other):
        if False:
            print('Hello World!')
        'Return True if two sets have a null intersection.'
        for value in other:
            if value in self:
                return False
        return True

    def __or__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, Iterable):
            return NotImplemented
        chain = (e for s in (self, other) for e in s)
        return self._from_iterable(chain)
    __ror__ = __or__

    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, Set):
            if not isinstance(other, Iterable):
                return NotImplemented
            other = self._from_iterable(other)
        return self._from_iterable((value for value in self if value not in other))

    def __rsub__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, Set):
            if not isinstance(other, Iterable):
                return NotImplemented
            other = self._from_iterable(other)
        return self._from_iterable((value for value in other if value not in self))

    def __xor__(self, other):
        if False:
            return 10
        if not isinstance(other, Set):
            if not isinstance(other, Iterable):
                return NotImplemented
            other = self._from_iterable(other)
        return self - other | other - self
    __rxor__ = __xor__

    def _hash(self):
        if False:
            while True:
                i = 10
        "Compute the hash value of a set.\n\n        Note that we don't define __hash__: not all sets are hashable.\n        But if you define a hashable set type, its __hash__ should\n        call this function.\n\n        This must be compatible __eq__.\n\n        All sets ought to compare equal if they contain the same\n        elements, regardless of how they are implemented, and\n        regardless of the order of the elements; so there's not much\n        freedom for __eq__ or __hash__.  We match the algorithm used\n        by the built-in frozenset type.\n        "
        MAX = sys.maxsize
        MASK = 2 * MAX + 1
        n = len(self)
        h = 1927868237 * (n + 1)
        h &= MASK
        for x in self:
            hx = hash(x)
            h ^= (hx ^ hx << 16 ^ 89869747) * 3644798167
            h &= MASK
        h ^= h >> 11 ^ h >> 25
        h = h * 69069 + 907133923
        h &= MASK
        if h > MAX:
            h -= MASK + 1
        if h == -1:
            h = 590923713
        return h
Set.register(frozenset)

class MutableSet(Set):
    """A mutable set is a finite, iterable container.

    This class provides concrete generic implementations of all
    methods except for __contains__, __iter__, __len__,
    add(), and discard().

    To override the comparisons (presumably for speed, as the
    semantics are fixed), all you have to do is redefine __le__ and
    then the other operations will automatically follow suit.
    """
    __slots__ = ()

    @abstractmethod
    def add(self, value):
        if False:
            return 10
        'Add an element.'
        raise NotImplementedError

    @abstractmethod
    def discard(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Remove an element.  Do not raise an exception if absent.'
        raise NotImplementedError

    def remove(self, value):
        if False:
            print('Hello World!')
        'Remove an element. If not a member, raise a KeyError.'
        if value not in self:
            raise KeyError(value)
        self.discard(value)

    def pop(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the popped value.  Raise KeyError if empty.'
        it = iter(self)
        try:
            value = next(it)
        except StopIteration:
            raise KeyError from None
        self.discard(value)
        return value

    def clear(self):
        if False:
            return 10
        'This is slow (creates N new iterators!) but effective.'
        try:
            while True:
                self.pop()
        except KeyError:
            pass

    def __ior__(self, it):
        if False:
            return 10
        for value in it:
            self.add(value)
        return self

    def __iand__(self, it):
        if False:
            return 10
        for value in self - it:
            self.discard(value)
        return self

    def __ixor__(self, it):
        if False:
            while True:
                i = 10
        if it is self:
            self.clear()
        else:
            if not isinstance(it, Set):
                it = self._from_iterable(it)
            for value in it:
                if value in self:
                    self.discard(value)
                else:
                    self.add(value)
        return self

    def __isub__(self, it):
        if False:
            return 10
        if it is self:
            self.clear()
        else:
            for value in it:
                self.discard(value)
        return self
MutableSet.register(set)

class Mapping(Collection):
    """A Mapping is a generic container for associating key/value
    pairs.

    This class provides concrete generic implementations of all
    methods except for __getitem__, __iter__, and __len__.
    """
    __slots__ = ()
    __abc_tpflags__ = 1 << 6

    @abstractmethod
    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        raise KeyError

    def get(self, key, default=None):
        if False:
            print('Hello World!')
        'D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.'
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        if False:
            i = 10
            return i + 15
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def keys(self):
        if False:
            i = 10
            return i + 15
        "D.keys() -> a set-like object providing a view on D's keys"
        return KeysView(self)

    def items(self):
        if False:
            i = 10
            return i + 15
        "D.items() -> a set-like object providing a view on D's items"
        return ItemsView(self)

    def values(self):
        if False:
            return 10
        "D.values() -> an object providing a view on D's values"
        return ValuesView(self)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())
    __reversed__ = None
Mapping.register(mappingproxy)

class MappingView(Sized):
    __slots__ = ('_mapping',)

    def __init__(self, mapping):
        if False:
            while True:
                i = 10
        self._mapping = mapping

    def __len__(self):
        if False:
            return 10
        return len(self._mapping)

    def __repr__(self):
        if False:
            return 10
        return '{0.__class__.__name__}({0._mapping!r})'.format(self)
    __class_getitem__ = classmethod(GenericAlias)

class KeysView(MappingView, Set):
    __slots__ = ()

    @classmethod
    def _from_iterable(cls, it):
        if False:
            for i in range(10):
                print('nop')
        return set(it)

    def __contains__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return key in self._mapping

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        yield from self._mapping
KeysView.register(dict_keys)

class ItemsView(MappingView, Set):
    __slots__ = ()

    @classmethod
    def _from_iterable(cls, it):
        if False:
            return 10
        return set(it)

    def __contains__(self, item):
        if False:
            i = 10
            return i + 15
        (key, value) = item
        try:
            v = self._mapping[key]
        except KeyError:
            return False
        else:
            return v is value or v == value

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        for key in self._mapping:
            yield (key, self._mapping[key])
ItemsView.register(dict_items)

class ValuesView(MappingView, Collection):
    __slots__ = ()

    def __contains__(self, value):
        if False:
            return 10
        for key in self._mapping:
            v = self._mapping[key]
            if v is value or v == value:
                return True
        return False

    def __iter__(self):
        if False:
            while True:
                i = 10
        for key in self._mapping:
            yield self._mapping[key]
ValuesView.register(dict_values)

class MutableMapping(Mapping):
    """A MutableMapping is a generic container for associating
    key/value pairs.

    This class provides concrete generic implementations of all
    methods except for __getitem__, __setitem__, __delitem__,
    __iter__, and __len__.
    """
    __slots__ = ()

    @abstractmethod
    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        raise KeyError

    @abstractmethod
    def __delitem__(self, key):
        if False:
            i = 10
            return i + 15
        raise KeyError
    __marker = object()

    def pop(self, key, default=__marker):
        if False:
            print('Hello World!')
        'D.pop(k[,d]) -> v, remove specified key and return the corresponding value.\n          If key is not found, d is returned if given, otherwise KeyError is raised.\n        '
        try:
            value = self[key]
        except KeyError:
            if default is self.__marker:
                raise
            return default
        else:
            del self[key]
            return value

    def popitem(self):
        if False:
            while True:
                i = 10
        'D.popitem() -> (k, v), remove and return some (key, value) pair\n           as a 2-tuple; but raise KeyError if D is empty.\n        '
        try:
            key = next(iter(self))
        except StopIteration:
            raise KeyError from None
        value = self[key]
        del self[key]
        return (key, value)

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        'D.clear() -> None.  Remove all items from D.'
        try:
            while True:
                self.popitem()
        except KeyError:
            pass

    def update(self, other=(), /, **kwds):
        if False:
            for i in range(10):
                print('nop')
        ' D.update([E, ]**F) -> None.  Update D from mapping/iterable E and F.\n            If E present and has a .keys() method, does:     for k in E: D[k] = E[k]\n            If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v\n            In either case, this is followed by: for k, v in F.items(): D[k] = v\n        '
        if isinstance(other, Mapping):
            for key in other:
                self[key] = other[key]
        elif hasattr(other, 'keys'):
            for key in other.keys():
                self[key] = other[key]
        else:
            for (key, value) in other:
                self[key] = value
        for (key, value) in kwds.items():
            self[key] = value

    def setdefault(self, key, default=None):
        if False:
            print('Hello World!')
        'D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D'
        try:
            return self[key]
        except KeyError:
            self[key] = default
        return default
MutableMapping.register(dict)

class Sequence(Reversible, Collection):
    """All the operations on a read-only sequence.

    Concrete subclasses must override __new__ or __init__,
    __getitem__, and __len__.
    """
    __slots__ = ()
    __abc_tpflags__ = 1 << 5

    @abstractmethod
    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        raise IndexError

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        i = 0
        try:
            while True:
                v = self[i]
                yield v
                i += 1
        except IndexError:
            return

    def __contains__(self, value):
        if False:
            for i in range(10):
                print('nop')
        for v in self:
            if v is value or v == value:
                return True
        return False

    def __reversed__(self):
        if False:
            for i in range(10):
                print('nop')
        for i in reversed(range(len(self))):
            yield self[i]

    def index(self, value, start=0, stop=None):
        if False:
            i = 10
            return i + 15
        'S.index(value, [start, [stop]]) -> integer -- return first index of value.\n           Raises ValueError if the value is not present.\n\n           Supporting start and stop arguments is optional, but\n           recommended.\n        '
        if start is not None and start < 0:
            start = max(len(self) + start, 0)
        if stop is not None and stop < 0:
            stop += len(self)
        i = start
        while stop is None or i < stop:
            try:
                v = self[i]
                if v is value or v == value:
                    return i
            except IndexError:
                break
            i += 1
        raise ValueError

    def count(self, value):
        if False:
            i = 10
            return i + 15
        'S.count(value) -> integer -- return number of occurrences of value'
        return sum((1 for v in self if v is value or v == value))
Sequence.register(tuple)
Sequence.register(str)
Sequence.register(range)
Sequence.register(memoryview)

class ByteString(Sequence):
    """This unifies bytes and bytearray.

    XXX Should add all their methods.
    """
    __slots__ = ()
ByteString.register(bytes)
ByteString.register(bytearray)

class MutableSequence(Sequence):
    """All the operations on a read-write sequence.

    Concrete subclasses must provide __new__ or __init__,
    __getitem__, __setitem__, __delitem__, __len__, and insert().
    """
    __slots__ = ()

    @abstractmethod
    def __setitem__(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        raise IndexError

    @abstractmethod
    def __delitem__(self, index):
        if False:
            i = 10
            return i + 15
        raise IndexError

    @abstractmethod
    def insert(self, index, value):
        if False:
            return 10
        'S.insert(index, value) -- insert value before index'
        raise IndexError

    def append(self, value):
        if False:
            while True:
                i = 10
        'S.append(value) -- append value to the end of the sequence'
        self.insert(len(self), value)

    def clear(self):
        if False:
            print('Hello World!')
        'S.clear() -> None -- remove all items from S'
        try:
            while True:
                self.pop()
        except IndexError:
            pass

    def reverse(self):
        if False:
            while True:
                i = 10
        'S.reverse() -- reverse *IN PLACE*'
        n = len(self)
        for i in range(n // 2):
            (self[i], self[n - i - 1]) = (self[n - i - 1], self[i])

    def extend(self, values):
        if False:
            i = 10
            return i + 15
        'S.extend(iterable) -- extend sequence by appending elements from the iterable'
        if values is self:
            values = list(values)
        for v in values:
            self.append(v)

    def pop(self, index=-1):
        if False:
            return 10
        'S.pop([index]) -> item -- remove and return item at index (default last).\n           Raise IndexError if list is empty or index is out of range.\n        '
        v = self[index]
        del self[index]
        return v

    def remove(self, value):
        if False:
            print('Hello World!')
        'S.remove(value) -- remove first occurrence of value.\n           Raise ValueError if the value is not present.\n        '
        del self[self.index(value)]

    def __iadd__(self, values):
        if False:
            print('Hello World!')
        self.extend(values)
        return self
MutableSequence.register(list)
MutableSequence.register(bytearray)