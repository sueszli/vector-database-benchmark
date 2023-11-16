from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, ClassVar, TypeVar, final
from returns.functions import identity
from returns.future import FutureResult
from returns.interfaces.specific import reader
from returns.io import IOResult
from returns.primitives.container import BaseContainer
from returns.primitives.hkt import Kind2, SupportsKind2, dekind
from returns.result import Result
if TYPE_CHECKING:
    from returns.context.requires_context_future_result import RequiresContextFutureResult
    from returns.context.requires_context_ioresult import RequiresContextIOResult
    from returns.context.requires_context_result import RequiresContextResult
_EnvType = TypeVar('_EnvType', contravariant=True)
_NewEnvType = TypeVar('_NewEnvType')
_ReturnType = TypeVar('_ReturnType', covariant=True)
_NewReturnType = TypeVar('_NewReturnType')
_ValueType = TypeVar('_ValueType')
_ErrorType = TypeVar('_ErrorType')
_FirstType = TypeVar('_FirstType')
NoDeps = Any

@final
class RequiresContext(BaseContainer, SupportsKind2['RequiresContext', _ReturnType, _EnvType], reader.ReaderBased2[_ReturnType, _EnvType]):
    """
    The ``RequiresContext`` container.

    It's main purpose is to wrap some specific function
    and to provide tools to compose other functions around it
    without actually calling it.

    The ``RequiresContext`` container passes the state
    you want to share between functions.
    Functions may read that state, but can't change it.
    The ``RequiresContext`` container
    lets us access shared immutable state within a specific context.

    It can be used for lazy evaluation and typed dependency injection.

    ``RequiresContext`` is used with functions that never fail.
    If you want to use ``RequiresContext`` with returns ``Result``
    then consider using ``RequiresContextResult`` instead.

    Note:
        This container does not wrap ANY value. It wraps only functions.
        You won't be able to supply arbitrary types!

    See also:
        - https://dev.to/gcanti/getting-started-with-fp-ts-reader-1ie5
        - https://en.wikipedia.org/wiki/Lazy_evaluation
        - https://bit.ly/2R8l4WK

    """
    __slots__ = ()
    _inner_value: Callable[[_EnvType], _ReturnType]
    no_args: ClassVar[NoDeps] = object()

    def __init__(self, inner_value: Callable[[_EnvType], _ReturnType]) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Public constructor for this type. Also required for typing.\n\n        Only allows functions of kind ``* -> *``.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n          >>> str(RequiresContext(lambda deps: deps + 1))\n          '<RequiresContext: <function <lambda> at ...>>'\n\n        "
        super().__init__(inner_value)

    def __call__(self, deps: _EnvType) -> _ReturnType:
        if False:
            print('Hello World!')
        '\n        Evaluates the wrapped function.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n\n          >>> def first(lg: bool) -> RequiresContext[int, float]:\n          ...     # `deps` has `float` type here:\n          ...     return RequiresContext(\n          ...         lambda deps: deps if lg else -deps,\n          ...     )\n\n          >>> instance = first(False)  # creating `RequiresContext` instance\n          >>> assert instance(3.5) == -3.5 # calling it with `__call__`\n\n          >>> # Example with another logic:\n          >>> assert first(True)(3.5) == 3.5\n\n        In other things, it is a regular python magic method.\n        '
        return self._inner_value(deps)

    def map(self, function: Callable[[_ReturnType], _NewReturnType]) -> RequiresContext[_NewReturnType, _EnvType]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Allows to compose functions inside the wrapped container.\n\n        Here's how it works:\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n          >>> def first(lg: bool) -> RequiresContext[int, float]:\n          ...     # `deps` has `float` type here:\n          ...     return RequiresContext(\n          ...         lambda deps: deps if lg else -deps,\n          ...     )\n\n          >>> assert first(True).map(lambda number: number * 10)(2.5) == 25.0\n          >>> assert first(False).map(lambda number: number * 10)(0.1) -1.0\n\n        "
        return RequiresContext(lambda deps: function(self(deps)))

    def apply(self, container: Kind2[RequiresContext, Callable[[_ReturnType], _NewReturnType], _EnvType]) -> RequiresContext[_NewReturnType, _EnvType]:
        if False:
            while True:
                i = 10
        "\n        Calls a wrapped function in a container on this container.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n          >>> assert RequiresContext.from_value('a').apply(\n          ...    RequiresContext.from_value(lambda inner: inner + 'b')\n          ... )(...) == 'ab'\n\n        "
        return RequiresContext(lambda deps: self.map(dekind(container)(deps))(deps))

    def bind(self, function: Callable[[_ReturnType], Kind2[RequiresContext, _NewReturnType, _EnvType]]) -> RequiresContext[_NewReturnType, _EnvType]:
        if False:
            i = 10
            return i + 15
        "\n        Composes a container with a function returning another container.\n\n        This is useful when you do several computations that rely on the\n        same context.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n\n          >>> def first(lg: bool) -> RequiresContext[int, float]:\n          ...     # `deps` has `float` type here:\n          ...     return RequiresContext(\n          ...         lambda deps: deps if lg else -deps,\n          ...     )\n\n          >>> def second(number: int) -> RequiresContext[str, float]:\n          ...     # `deps` has `float` type here:\n          ...     return RequiresContext(\n          ...         lambda deps: '>=' if number >= deps else '<',\n          ...     )\n\n          >>> assert first(True).bind(second)(1) == '>='\n          >>> assert first(False).bind(second)(2) == '<'\n\n        "
        return RequiresContext(lambda deps: dekind(function(self(deps)))(deps))
    bind_context = bind

    def modify_env(self, function: Callable[[_NewEnvType], _EnvType]) -> RequiresContext[_ReturnType, _NewEnvType]:
        if False:
            i = 10
            return i + 15
        "\n        Allows to modify the environment type.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n\n          >>> def mul(arg: int) -> RequiresContext[float, int]:\n          ...     return RequiresContext(lambda deps: arg * deps)\n\n          >>> assert mul(3).modify_env(int)('2') == 6\n\n        "
        return RequiresContext(lambda deps: self(function(deps)))

    @classmethod
    def ask(cls) -> RequiresContext[_EnvType, _EnvType]:
        if False:
            i = 10
            return i + 15
        "\n        Get current context to use the dependencies.\n\n        It is a common scenario when you need to use the environment.\n        For example, you want to do some context-related computation,\n        but you don't have the context instance at your disposal.\n        That's where ``.ask()`` becomes useful!\n\n        .. code:: python\n\n          >>> from typing_extensions import TypedDict\n          >>> class Deps(TypedDict):\n          ...     message: str\n\n          >>> def first(lg: bool) -> RequiresContext[int, Deps]:\n          ...     # `deps` has `Deps` type here:\n          ...     return RequiresContext(\n          ...         lambda deps: deps['message'] if lg else 'error',\n          ...     )\n\n          >>> def second(text: str) -> RequiresContext[int, Deps]:\n          ...     return first(len(text) > 3)\n\n          >>> assert second('abc')({'message': 'ok'}) == 'error'\n          >>> assert second('abcd')({'message': 'ok'}) == 'ok'\n\n        And now imagine that you have to change this ``3`` limit.\n        And you want to be able to set it via environment as well.\n        Ok, let's fix it with the power of ``RequiresContext.ask()``!\n\n        .. code:: python\n\n          >>> from typing_extensions import TypedDict\n          >>> class Deps(TypedDict):\n          ...     message: str\n          ...     limit: int   # note this new field!\n\n          >>> def new_first(lg: bool) -> RequiresContext[int, Deps]:\n          ...     # `deps` has `Deps` type here:\n          ...     return RequiresContext(\n          ...         lambda deps: deps['message'] if lg else 'err',\n          ...     )\n\n          >>> def new_second(text: str) -> RequiresContext[int, Deps]:\n          ...     return RequiresContext[int, Deps].ask().bind(\n          ...         lambda deps: new_first(len(text) > deps.get('limit', 3)),\n          ...     )\n\n          >>> assert new_second('abc')({'message': 'ok', 'limit': 2}) == 'ok'\n          >>> assert new_second('abcd')({'message': 'ok'}) == 'ok'\n          >>> assert new_second('abcd')({'message': 'ok', 'limit': 5}) == 'err'\n\n        That's how ``ask`` works.\n\n        This class contains methods that require\n        to explicitly set type annotations. Why?\n        Because it is impossible to figure out the type without them.\n        So, here's how you should use them:\n\n        .. code:: python\n\n            RequiresContext[int, Dict[str, str]].ask()\n\n        Otherwise, your ``.ask()`` method\n        will return ``RequiresContext[<nothing>, <nothing>]``,\n        which is unusable:\n\n        .. code:: python\n\n            env = RequiresContext.ask()\n            env(some_deps)\n\n        And ``mypy`` will warn you: ``error: Need type annotation for '...'``\n\n        See also:\n            - https://dev.to/gcanti/getting-started-with-fp-ts-reader-1ie5\n\n        "
        return RequiresContext(identity)

    @classmethod
    def from_value(cls, inner_value: _FirstType) -> RequiresContext[_FirstType, NoDeps]:
        if False:
            return 10
        '\n        Used to return some specific value from the container.\n\n        Consider this method as some kind of factory.\n        Passed value will be a return type.\n        Make sure to use :attr:`~RequiresContext.no_args`\n        for getting the unit value.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n          >>> unit = RequiresContext.from_value(5)\n          >>> assert unit(RequiresContext.no_args) == 5\n\n        Might be used with or without direct type hint.\n        '
        return RequiresContext(lambda _: inner_value)

    @classmethod
    def from_context(cls, inner_value: RequiresContext[_NewReturnType, _NewEnvType]) -> RequiresContext[_NewReturnType, _NewEnvType]:
        if False:
            return 10
        '\n        Used to create new containers from existing ones.\n\n        Used as a part of ``ReaderBased2`` interface.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n          >>> unit = RequiresContext.from_value(5)\n          >>> assert RequiresContext.from_context(unit)(...) == unit(...)\n\n        '
        return inner_value

    @classmethod
    def from_requires_context_result(cls, inner_value: RequiresContextResult[_ValueType, _ErrorType, _EnvType]) -> RequiresContext[Result[_ValueType, _ErrorType], _EnvType]:
        if False:
            return 10
        '\n        Typecasts ``RequiresContextResult`` to ``RequiresContext`` instance.\n\n        Breaks ``RequiresContextResult[a, b, e]``\n        into ``RequiresContext[Result[a, b], e]``.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.result import Success\n          >>> assert RequiresContext.from_requires_context_result(\n          ...    RequiresContextResult.from_value(1),\n          ... )(...) == Success(1)\n\n        Can be reverted with ``RequiresContextResult.from_typecast``.\n\n        '
        return RequiresContext(inner_value)

    @classmethod
    def from_requires_context_ioresult(cls, inner_value: RequiresContextIOResult[_ValueType, _ErrorType, _EnvType]) -> RequiresContext[IOResult[_ValueType, _ErrorType], _EnvType]:
        if False:
            while True:
                i = 10
        '\n        Typecasts ``RequiresContextIOResult`` to ``RequiresContext`` instance.\n\n        Breaks ``RequiresContextIOResult[a, b, e]``\n        into ``RequiresContext[IOResult[a, b], e]``.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IOSuccess\n          >>> assert RequiresContext.from_requires_context_ioresult(\n          ...    RequiresContextIOResult.from_value(1),\n          ... )(...) == IOSuccess(1)\n\n        Can be reverted with ``RequiresContextIOResult.from_typecast``.\n\n        '
        return RequiresContext(inner_value)

    @classmethod
    def from_requires_context_future_result(cls, inner_value: RequiresContextFutureResult[_ValueType, _ErrorType, _EnvType]) -> RequiresContext[FutureResult[_ValueType, _ErrorType], _EnvType]:
        if False:
            while True:
                i = 10
        '\n        Typecasts ``RequiresContextIOResult`` to ``RequiresContext`` instance.\n\n        Breaks ``RequiresContextIOResult[a, b, e]``\n        into ``RequiresContext[IOResult[a, b], e]``.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContext\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.io import IOSuccess\n\n          >>> container = RequiresContext.from_requires_context_future_result(\n          ...    RequiresContextFutureResult.from_value(1),\n          ... )\n          >>> assert anyio.run(\n          ...     container, RequiresContext.no_args,\n          ... ) == IOSuccess(1)\n\n        Can be reverted with ``RequiresContextFutureResult.from_typecast``.\n\n        '
        return RequiresContext(inner_value)
Reader = RequiresContext