from abc import ABCMeta
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generator, Iterator, NoReturn, Optional, TypeVar, Union, final
from typing_extensions import ParamSpec
from returns.interfaces.specific.maybe import MaybeBased2
from returns.primitives.container import BaseContainer, container_equality
from returns.primitives.exceptions import UnwrapFailedError
from returns.primitives.hkt import Kind1, SupportsKind1
_ValueType = TypeVar('_ValueType', covariant=True)
_NewValueType = TypeVar('_NewValueType')
_FuncParams = ParamSpec('_FuncParams')

class Maybe(BaseContainer, SupportsKind1['Maybe', _ValueType], MaybeBased2[_ValueType, None], metaclass=ABCMeta):
    """
    Represents a result of a series of computations that can return ``None``.

    An alternative to using exceptions or constant ``is None`` checks.
    ``Maybe`` is an abstract type and should not be instantiated directly.
    Instead use ``Some`` and ``Nothing``.

    See also:
        - https://github.com/gcanti/fp-ts/blob/master/docs/modules/Option.ts.md

    """
    __slots__ = ()
    _inner_value: Optional[_ValueType]
    __match_args__ = ('_inner_value',)
    empty: ClassVar['Maybe[Any]']
    equals = container_equality

    def map(self, function: Callable[[_ValueType], _NewValueType]) -> 'Maybe[_NewValueType]':
        if False:
            print('Hello World!')
        "\n        Composes successful container with a pure function.\n\n        .. code:: python\n\n          >>> from returns.maybe import Some, Nothing\n          >>> def mappable(string: str) -> str:\n          ...      return string + 'b'\n\n          >>> assert Some('a').map(mappable) == Some('ab')\n          >>> assert Nothing.map(mappable) == Nothing\n\n        "

    def apply(self, function: Kind1['Maybe', Callable[[_ValueType], _NewValueType]]) -> 'Maybe[_NewValueType]':
        if False:
            i = 10
            return i + 15
        "\n        Calls a wrapped function in a container on this container.\n\n        .. code:: python\n\n          >>> from returns.maybe import Some, Nothing\n\n          >>> def appliable(string: str) -> str:\n          ...      return string + 'b'\n\n          >>> assert Some('a').apply(Some(appliable)) == Some('ab')\n          >>> assert Some('a').apply(Nothing) == Nothing\n          >>> assert Nothing.apply(Some(appliable)) == Nothing\n          >>> assert Nothing.apply(Nothing) == Nothing\n\n        "

    def bind(self, function: Callable[[_ValueType], Kind1['Maybe', _NewValueType]]) -> 'Maybe[_NewValueType]':
        if False:
            while True:
                i = 10
        "\n        Composes successful container with a function that returns a container.\n\n        .. code:: python\n\n          >>> from returns.maybe import Nothing, Maybe, Some\n          >>> def bindable(string: str) -> Maybe[str]:\n          ...      return Some(string + 'b')\n\n          >>> assert Some('a').bind(bindable) == Some('ab')\n          >>> assert Nothing.bind(bindable) == Nothing\n\n        "

    def bind_optional(self, function: Callable[[_ValueType], Optional[_NewValueType]]) -> 'Maybe[_NewValueType]':
        if False:
            for i in range(10):
                print('nop')
        "\n        Binds a function returning an optional value over a container.\n\n        .. code:: python\n\n          >>> from returns.maybe import Some, Nothing\n          >>> from typing import Optional\n\n          >>> def bindable(arg: str) -> Optional[int]:\n          ...     return len(arg) if arg else None\n\n          >>> assert Some('a').bind_optional(bindable) == Some(1)\n          >>> assert Some('').bind_optional(bindable) == Nothing\n\n        "

    def lash(self, function: Callable[[Any], Kind1['Maybe', _ValueType]]) -> 'Maybe[_ValueType]':
        if False:
            print('Hello World!')
        "\n        Composes failed container with a function that returns a container.\n\n        .. code:: python\n\n          >>> from returns.maybe import Maybe, Some, Nothing\n\n          >>> def lashable(arg=None) -> Maybe[str]:\n          ...      return Some('b')\n\n          >>> assert Some('a').lash(lashable) == Some('a')\n          >>> assert Nothing.lash(lashable) == Some('b')\n\n        We need this feature to make ``Maybe`` compatible\n        with different ``Result`` like operations.\n\n        "

    def __iter__(self) -> Iterator[_ValueType]:
        if False:
            while True:
                i = 10
        'API for :ref:`do-notation`.'
        yield self.unwrap()

    @classmethod
    def do(cls, expr: Generator[_NewValueType, None, None]) -> 'Maybe[_NewValueType]':
        if False:
            return 10
        '\n        Allows working with unwrapped values of containers in a safe way.\n\n        .. code:: python\n\n          >>> from returns.maybe import Maybe, Some, Nothing\n\n          >>> assert Maybe.do(\n          ...     first + second\n          ...     for first in Some(2)\n          ...     for second in Some(3)\n          ... ) == Some(5)\n\n          >>> assert Maybe.do(\n          ...     first + second\n          ...     for first in Some(2)\n          ...     for second in Nothing\n          ... ) == Nothing\n\n        See :ref:`do-notation` to learn more.\n\n        '
        try:
            return Maybe.from_value(next(expr))
        except UnwrapFailedError as exc:
            return exc.halted_container

    def value_or(self, default_value: _NewValueType) -> Union[_ValueType, _NewValueType]:
        if False:
            print('Hello World!')
        '\n        Get value from successful container or default value from failed one.\n\n        .. code:: python\n\n          >>> from returns.maybe import Nothing, Some\n          >>> assert Some(0).value_or(1) == 0\n          >>> assert Nothing.value_or(1) == 1\n\n        '

    def or_else_call(self, function: Callable[[], _NewValueType]) -> Union[_ValueType, _NewValueType]:
        if False:
            i = 10
            return i + 15
        "\n        Get value from successful container or default value from failed one.\n\n        Really close to :meth:`~Maybe.value_or` but works with lazy values.\n        This method is unique to ``Maybe`` container, because other containers\n        do have ``.alt`` method.\n\n        But, ``Maybe`` does not have this method.\n        There's nothing to ``alt`` in ``Nothing``.\n\n        Instead, it has this method to execute\n        some function if called on a failed container:\n\n        .. code:: pycon\n\n          >>> from returns.maybe import Some, Nothing\n          >>> assert Some(1).or_else_call(lambda: 2) == 1\n          >>> assert Nothing.or_else_call(lambda: 2) == 2\n\n        It might be useful to work with exceptions as well:\n\n        .. code:: pycon\n\n          >>> def fallback() -> NoReturn:\n          ...    raise ValueError('Nothing!')\n\n          >>> Nothing.or_else_call(fallback)\n          Traceback (most recent call last):\n            ...\n          ValueError: Nothing!\n\n        "

    def unwrap(self) -> _ValueType:
        if False:
            return 10
        '\n        Get value from successful container or raise exception for failed one.\n\n        .. code:: pycon\n          :force:\n\n          >>> from returns.maybe import Nothing, Some\n          >>> assert Some(1).unwrap() == 1\n\n          >>> Nothing.unwrap()\n          Traceback (most recent call last):\n            ...\n          returns.primitives.exceptions.UnwrapFailedError\n\n        '

    def failure(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Get failed value from failed container or raise exception from success.\n\n        .. code:: pycon\n          :force:\n\n          >>> from returns.maybe import Nothing, Some\n          >>> assert Nothing.failure() is None\n\n          >>> Some(1).failure()\n          Traceback (most recent call last):\n            ...\n          returns.primitives.exceptions.UnwrapFailedError\n\n        '

    @classmethod
    def from_value(cls, inner_value: _NewValueType) -> 'Maybe[_NewValueType]':
        if False:
            i = 10
            return i + 15
        '\n        Creates new instance of ``Maybe`` container based on a value.\n\n        .. code:: python\n\n          >>> from returns.maybe import Maybe, Some\n          >>> assert Maybe.from_value(1) == Some(1)\n          >>> assert Maybe.from_value(None) == Some(None)\n\n        '
        return Some(inner_value)

    @classmethod
    def from_optional(cls, inner_value: Optional[_NewValueType]) -> 'Maybe[_NewValueType]':
        if False:
            while True:
                i = 10
        '\n        Creates new instance of ``Maybe`` container based on an optional value.\n\n        .. code:: python\n\n          >>> from returns.maybe import Maybe, Some, Nothing\n          >>> assert Maybe.from_optional(1) == Some(1)\n          >>> assert Maybe.from_optional(None) == Nothing\n\n        '
        if inner_value is None:
            return _Nothing(inner_value)
        return Some(inner_value)

@final
class _Nothing(Maybe[Any]):
    """Represents an empty state."""
    __slots__ = ()
    _inner_value: None
    _instance: Optional['_Nothing'] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> '_Nothing':
        if False:
            return 10
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, inner_value: None=None) -> None:
        if False:
            return 10
        '\n        Private constructor for ``_Nothing`` type.\n\n        Use :attr:`~Nothing` instead.\n        Wraps the given value in the ``_Nothing`` container.\n\n        ``inner_value`` can only be ``None``.\n        '
        super().__init__(None)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Custom ``str`` definition without the state inside.\n\n        .. code:: python\n\n          >>> from returns.maybe import Nothing\n          >>> assert str(Nothing) == '<Nothing>'\n          >>> assert repr(Nothing) == '<Nothing>'\n\n        "
        return '<Nothing>'

    def map(self, function):
        if False:
            for i in range(10):
                print('nop')
        'Does nothing for ``Nothing``.'
        return self

    def apply(self, container):
        if False:
            return 10
        'Does nothing for ``Nothing``.'
        return self

    def bind(self, function):
        if False:
            while True:
                i = 10
        'Does nothing for ``Nothing``.'
        return self

    def bind_optional(self, function):
        if False:
            for i in range(10):
                print('nop')
        'Does nothing.'
        return self

    def lash(self, function):
        if False:
            print('Hello World!')
        'Composes this container with a function returning container.'
        return function(None)

    def value_or(self, default_value):
        if False:
            print('Hello World!')
        'Returns default value.'
        return default_value

    def or_else_call(self, function):
        if False:
            while True:
                i = 10
        'Returns the result of a passed function.'
        return function()

    def unwrap(self):
        if False:
            return 10
        'Raises an exception, since it does not have a value inside.'
        raise UnwrapFailedError(self)

    def failure(self) -> None:
        if False:
            i = 10
            return i + 15
        'Returns failed value.'
        return self._inner_value

@final
class Some(Maybe[_ValueType]):
    """
    Represents a calculation which has succeeded and contains the value.

    Quite similar to ``Success`` type.
    """
    __slots__ = ()
    _inner_value: _ValueType

    def __init__(self, inner_value: _ValueType) -> None:
        if False:
            return 10
        'Some constructor.'
        super().__init__(inner_value)
    if not TYPE_CHECKING:

        def bind(self, function):
            if False:
                return 10
            'Binds current container to a function that returns container.'
            return function(self._inner_value)

        def bind_optional(self, function):
            if False:
                i = 10
                return i + 15
            'Binds a function returning an optional value over a container.'
            return Maybe.from_optional(function(self._inner_value))

        def unwrap(self):
            if False:
                i = 10
                return i + 15
            'Returns inner value for successful container.'
            return self._inner_value

    def map(self, function):
        if False:
            return 10
        'Composes current container with a pure function.'
        return Some(function(self._inner_value))

    def apply(self, container):
        if False:
            return 10
        'Calls a wrapped function in a container on this container.'
        if isinstance(container, Some):
            return self.map(container.unwrap())
        return container

    def lash(self, function):
        if False:
            for i in range(10):
                print('nop')
        'Does nothing for ``Some``.'
        return self

    def value_or(self, default_value):
        if False:
            print('Hello World!')
        'Returns inner value for successful container.'
        return self._inner_value

    def or_else_call(self, function):
        if False:
            print('Hello World!')
        'Returns inner value for successful container.'
        return self._inner_value

    def failure(self):
        if False:
            while True:
                i = 10
        'Raises exception for successful container.'
        raise UnwrapFailedError(self)
Nothing: Maybe[NoReturn] = _Nothing()
Maybe.empty = Nothing

def maybe(function: Callable[_FuncParams, Optional[_ValueType]]) -> Callable[_FuncParams, Maybe[_ValueType]]:
    if False:
        while True:
            i = 10
    '\n    Decorator to convert ``None``-returning function to ``Maybe`` container.\n\n    This decorator works with sync functions only. Example:\n\n    .. code:: python\n\n      >>> from typing import Optional\n      >>> from returns.maybe import Nothing, Some, maybe\n\n      >>> @maybe\n      ... def might_be_none(arg: int) -> Optional[int]:\n      ...     if arg == 0:\n      ...         return None\n      ...     return 1 / arg\n\n      >>> assert might_be_none(0) == Nothing\n      >>> assert might_be_none(1) == Some(1.0)\n\n    '

    @wraps(function)
    def decorator(*args: _FuncParams.args, **kwargs: _FuncParams.kwargs) -> Maybe[_ValueType]:
        if False:
            while True:
                i = 10
        return Maybe.from_optional(function(*args, **kwargs))
    return decorator