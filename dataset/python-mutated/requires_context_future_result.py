from __future__ import annotations
from typing import TYPE_CHECKING, Any, Awaitable, Callable, ClassVar, TypeVar, final
from returns._internal.futures import _reader_future_result
from returns.context import NoDeps
from returns.future import Future, FutureResult
from returns.interfaces.specific import future_result, reader_future_result
from returns.io import IO, IOResult
from returns.primitives.container import BaseContainer
from returns.primitives.hkt import Kind3, SupportsKind3, dekind
from returns.result import Result
if TYPE_CHECKING:
    from returns.context.requires_context import RequiresContext
    from returns.context.requires_context_ioresult import ReaderIOResult, RequiresContextIOResult
    from returns.context.requires_context_result import RequiresContextResult
_EnvType = TypeVar('_EnvType', contravariant=True)
_NewEnvType = TypeVar('_NewEnvType')
_ValueType = TypeVar('_ValueType', covariant=True)
_NewValueType = TypeVar('_NewValueType')
_ErrorType = TypeVar('_ErrorType', covariant=True)
_NewErrorType = TypeVar('_NewErrorType')
_FirstType = TypeVar('_FirstType')

@final
class RequiresContextFutureResult(BaseContainer, SupportsKind3['RequiresContextFutureResult', _ValueType, _ErrorType, _EnvType], reader_future_result.ReaderFutureResultBasedN[_ValueType, _ErrorType, _EnvType], future_result.FutureResultLike3[_ValueType, _ErrorType, _EnvType]):
    """
    The ``RequiresContextFutureResult`` combinator.

    This probably the main type people are going to use in ``async`` programs.

    See :class:`returns.context.requires_context.RequiresContext`,
    :class:`returns.context.requires_context_result.RequiresContextResult`,
    and
    :class:`returns.context.requires_context_result.RequiresContextIOResult`
    for more docs.

    This is just a handy wrapper around
    ``RequiresContext[FutureResult[a, b], env]``
    which represents a context-dependent impure async operation that might fail.

    So, this is a thin wrapper, without any changes in logic.
    Why do we need this wrapper? That's just for better usability!

    This way ``RequiresContextIOResult`` allows to simply work with:

    - raw values and pure functions
    - ``RequiresContext`` values and pure functions returning it
    - ``RequiresContextResult`` values and pure functions returning it
    - ``RequiresContextIOResult`` values and pure functions returning it
    - ``Result`` and pure functions returning it
    - ``IOResult`` and functions returning it
    - ``FutureResult`` and functions returning it
    - other ``RequiresContextFutureResult`` related functions and values

    This is a complex type for complex tasks!

    .. rubric:: Important implementation details

    Due it is meaning, ``RequiresContextFutureResult``
    cannot have ``Success`` and ``Failure`` subclasses.

    We only have just one type. That's by design.

    Different converters are also not supported for this type.
    Use converters inside the ``RequiresContext`` context, not outside.

    See also:
        - https://dev.to/gcanti/getting-started-with-fp-ts-reader-1ie5
        - https://en.wikipedia.org/wiki/Lazy_evaluation
        - https://bit.ly/2R8l4WK
        - https://bit.ly/2RwP4fp

    """
    __slots__ = ()
    _inner_value: Callable[[_EnvType], FutureResult[_ValueType, _ErrorType]]
    no_args: ClassVar[NoDeps] = object()

    def __init__(self, inner_value: Callable[[_EnvType], FutureResult[_ValueType, _ErrorType]]) -> None:
        if False:
            print('Hello World!')
        "\n        Public constructor for this type. Also required for typing.\n\n        Only allows functions of kind ``* -> *``\n        and returning :class:`returns.result.Result` instances.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.future import FutureResult\n\n          >>> instance = RequiresContextFutureResult(\n          ...    lambda deps: FutureResult.from_value(1),\n          ... )\n          >>> str(instance)\n          '<RequiresContextFutureResult: <function <lambda> at ...>>'\n\n        "
        super().__init__(inner_value)

    def __call__(self, deps: _EnvType) -> FutureResult[_ValueType, _ErrorType]:
        if False:
            i = 10
            return i + 15
        '\n        Evaluates the wrapped function.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.future import FutureResult\n          >>> from returns.io import IOSuccess\n\n          >>> def first(lg: bool) -> RequiresContextFutureResult[int, str, int]:\n          ...     # `deps` has `int` type here:\n          ...     return RequiresContextFutureResult(\n          ...         lambda deps: FutureResult.from_value(\n          ...             deps if lg else -deps,\n          ...         ),\n          ...     )\n\n          >>> instance = first(False)\n          >>> assert anyio.run(instance(3).awaitable) == IOSuccess(-3)\n\n          >>> instance = first(True)\n          >>> assert anyio.run(instance(3).awaitable) == IOSuccess(3)\n\n        In other things, it is a regular Python magic method.\n\n        '
        return self._inner_value(deps)

    def swap(self) -> RequiresContextFutureResult[_ErrorType, _ValueType, _EnvType]:
        if False:
            return 10
        '\n        Swaps value and error types.\n\n        So, values become errors and errors become values.\n        It is useful when you have to work with errors a lot.\n        And since we have a lot of ``.bind_`` related methods\n        and only a single ``.lash`` - it is easier to work with values.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> success = RequiresContextFutureResult.from_value(1)\n          >>> failure = RequiresContextFutureResult.from_failure(1)\n\n          >>> assert anyio.run(success.swap(), ...) == IOFailure(1)\n          >>> assert anyio.run(failure.swap(), ...) == IOSuccess(1)\n\n        '
        return RequiresContextFutureResult(lambda deps: self(deps).swap())

    def map(self, function: Callable[[_ValueType], _NewValueType]) -> RequiresContextFutureResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            i = 10
            return i + 15
        '\n        Composes successful container with a pure function.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> assert anyio.run(RequiresContextFutureResult.from_value(1).map(\n          ...     lambda x: x + 1,\n          ... )(...).awaitable) == IOSuccess(2)\n\n          >>> assert anyio.run(RequiresContextFutureResult.from_failure(1).map(\n          ...     lambda x: x + 1,\n          ... )(...).awaitable) == IOFailure(1)\n\n        '
        return RequiresContextFutureResult(lambda deps: self(deps).map(function))

    def apply(self, container: Kind3[RequiresContextFutureResult, Callable[[_ValueType], _NewValueType], _ErrorType, _EnvType]) -> RequiresContextFutureResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Calls a wrapped function in a container on this container.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> def transform(arg: str) -> str:\n          ...     return arg + 'b'\n\n          >>> assert anyio.run(\n          ...    RequiresContextFutureResult.from_value('a').apply(\n          ...        RequiresContextFutureResult.from_value(transform),\n          ...    ),\n          ...    RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess('ab')\n\n          >>> assert anyio.run(\n          ...    RequiresContextFutureResult.from_failure('a').apply(\n          ...        RequiresContextFutureResult.from_value(transform),\n          ...    ),\n          ...    RequiresContextFutureResult.no_args,\n          ... ) == IOFailure('a')\n\n        "
        return RequiresContextFutureResult(lambda deps: self(deps).apply(dekind(container)(deps)))

    def bind(self, function: Callable[[_ValueType], Kind3[RequiresContextFutureResult, _NewValueType, _ErrorType, _EnvType]]) -> RequiresContextFutureResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            print('Hello World!')
        "\n        Composes this container with a function returning the same type.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.future import FutureResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> def function(\n          ...     number: int,\n          ... ) -> RequiresContextFutureResult[str, int, int]:\n          ...     # `deps` has `int` type here:\n          ...     return RequiresContextFutureResult(\n          ...         lambda deps: FutureResult.from_value(str(number + deps)),\n          ...     )\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_value(2).bind(function),\n          ...     3,\n          ... ) == IOSuccess('5')\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_failure(2).bind(function),\n          ...     3,\n          ... ) == IOFailure(2)\n\n        "
        return RequiresContextFutureResult(lambda deps: self(deps).bind(lambda inner: dekind(function(inner))(deps)))
    bind_context_future_result = bind

    def bind_async(self, function: Callable[[_ValueType], Awaitable[Kind3[RequiresContextFutureResult, _NewValueType, _ErrorType, _EnvType],]]) -> RequiresContextFutureResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            print('Hello World!')
        '\n        Composes this container with a async function returning the same type.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> async def function(\n          ...     number: int,\n          ... ) -> RequiresContextFutureResult[str, int, int]:\n          ...     return RequiresContextFutureResult.from_value(number + 1)\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_value(1).bind_async(\n          ...        function,\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(2)\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_failure(1).bind_async(\n          ...        function,\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(1)\n\n        '
        return RequiresContextFutureResult(lambda deps: FutureResult(_reader_future_result.async_bind_async(function, self, deps)))
    bind_async_context_future_result = bind_async

    def bind_awaitable(self, function: Callable[[_ValueType], Awaitable[_NewValueType]]) -> RequiresContextFutureResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Allows to compose a container and a regular ``async`` function.\n\n        This function should return plain, non-container value.\n        See :meth:`~RequiresContextFutureResult.bind_async`\n        to bind ``async`` function that returns a container.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> async def coroutine(x: int) -> int:\n          ...    return x + 1\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_value(1).bind_awaitable(\n          ...         coroutine,\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(2)\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_failure(1).bind_awaitable(\n          ...         coroutine,\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(1)\n\n        '
        return RequiresContextFutureResult(lambda deps: self(deps).bind_awaitable(function))

    def bind_result(self, function: Callable[[_ValueType], Result[_NewValueType, _ErrorType]]) -> RequiresContextFutureResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Binds ``Result`` returning function to the current container.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.result import Success, Result\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> def function(num: int) -> Result[int, str]:\n          ...     return Success(num + 1)\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_value(1).bind_result(\n          ...         function,\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(2)\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_failure(':(').bind_result(\n          ...         function,\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(':(')\n\n        "
        return RequiresContextFutureResult(lambda deps: self(deps).bind_result(function))

    def bind_context(self, function: Callable[[_ValueType], RequiresContext[_NewValueType, _EnvType]]) -> RequiresContextFutureResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            return 10
        "\n        Binds ``RequiresContext`` returning function to current container.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContext\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> def function(arg: int) -> RequiresContext[int, str]:\n          ...     return RequiresContext(lambda deps: len(deps) + arg)\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_value(2).bind_context(\n          ...         function,\n          ...     ),\n          ...     'abc',\n          ... ) == IOSuccess(5)\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_failure(0).bind_context(\n          ...         function,\n          ...     ),\n          ...     'abc',\n          ... ) == IOFailure(0)\n\n        "
        return RequiresContextFutureResult(lambda deps: self(deps).map(lambda inner: function(inner)(deps)))

    def bind_context_result(self, function: Callable[[_ValueType], RequiresContextResult[_NewValueType, _ErrorType, _EnvType]]) -> RequiresContextFutureResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            print('Hello World!')
        "\n        Binds ``RequiresContextResult`` returning function to the current one.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.io import IOSuccess, IOFailure\n          >>> from returns.result import Success\n\n          >>> def function(arg: int) -> RequiresContextResult[int, int, str]:\n          ...     return RequiresContextResult(\n          ...         lambda deps: Success(len(deps) + arg),\n          ...     )\n\n          >>> instance = RequiresContextFutureResult.from_value(\n          ...    2,\n          ... ).bind_context_result(\n          ...     function,\n          ... )('abc')\n          >>> assert anyio.run(instance.awaitable) == IOSuccess(5)\n\n          >>> instance = RequiresContextFutureResult.from_failure(\n          ...    2,\n          ... ).bind_context_result(\n          ...     function,\n          ... )('abc')\n          >>> assert anyio.run(instance.awaitable) == IOFailure(2)\n\n        "
        return RequiresContextFutureResult(lambda deps: self(deps).bind_result(lambda inner: function(inner)(deps)))

    def bind_context_ioresult(self, function: Callable[[_ValueType], RequiresContextIOResult[_NewValueType, _ErrorType, _EnvType]]) -> RequiresContextFutureResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            print('Hello World!')
        "\n        Binds ``RequiresContextIOResult`` returning function to the current one.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> def function(arg: int) -> RequiresContextIOResult[int, int, str]:\n          ...     return RequiresContextIOResult(\n          ...         lambda deps: IOSuccess(len(deps) + arg),\n          ...     )\n\n          >>> instance = RequiresContextFutureResult.from_value(\n          ...    2,\n          ... ).bind_context_ioresult(\n          ...     function,\n          ... )('abc')\n          >>> assert anyio.run(instance.awaitable) == IOSuccess(5)\n\n          >>> instance = RequiresContextFutureResult.from_failure(\n          ...    2,\n          ... ).bind_context_ioresult(\n          ...     function,\n          ... )('abc')\n          >>> assert anyio.run(instance.awaitable) == IOFailure(2)\n\n        "
        return RequiresContextFutureResult(lambda deps: self(deps).bind_ioresult(lambda inner: function(inner)(deps)))

    def bind_io(self, function: Callable[[_ValueType], IO[_NewValueType]]) -> RequiresContextFutureResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            print('Hello World!')
        "\n        Binds ``IO`` returning function to the current container.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.io import IO, IOSuccess, IOFailure\n\n          >>> def do_io(number: int) -> IO[str]:\n          ...     return IO(str(number))  # not IO operation actually\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_value(1).bind_io(do_io),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess('1')\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_failure(1).bind_io(do_io),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(1)\n\n        "
        return RequiresContextFutureResult(lambda deps: self(deps).bind_io(function))

    def bind_ioresult(self, function: Callable[[_ValueType], IOResult[_NewValueType, _ErrorType]]) -> RequiresContextFutureResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            i = 10
            return i + 15
        "\n        Binds ``IOResult`` returning function to the current container.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.io import IOResult, IOSuccess, IOFailure\n\n          >>> def function(num: int) -> IOResult[int, str]:\n          ...     return IOSuccess(num + 1)\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_value(1).bind_ioresult(\n          ...         function,\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(2)\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_failure(':(').bind_ioresult(\n          ...         function,\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(':(')\n\n        "
        return RequiresContextFutureResult(lambda deps: self(deps).bind_ioresult(function))

    def bind_future(self, function: Callable[[_ValueType], Future[_NewValueType]]) -> RequiresContextFutureResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            print('Hello World!')
        "\n        Binds ``Future`` returning function to the current container.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.future import Future\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> def function(num: int) -> Future[int]:\n          ...     return Future.from_value(num + 1)\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_value(1).bind_future(\n          ...         function,\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(2)\n\n          >>> failed = RequiresContextFutureResult.from_failure(':(')\n          >>> assert anyio.run(\n          ...     failed.bind_future(function),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(':(')\n\n        "
        return RequiresContextFutureResult(lambda deps: self(deps).bind_future(function))

    def bind_future_result(self, function: Callable[[_ValueType], FutureResult[_NewValueType, _ErrorType]]) -> RequiresContextFutureResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            while True:
                i = 10
        "\n        Binds ``FutureResult`` returning function to the current container.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.future import FutureResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> def function(num: int) -> FutureResult[int, str]:\n          ...     return FutureResult.from_value(num + 1)\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_value(1).bind_future_result(\n          ...         function,\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(2)\n\n          >>> failed = RequiresContextFutureResult.from_failure(':(')\n          >>> assert anyio.run(\n          ...     failed.bind_future_result(function),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(':(')\n\n        "
        return RequiresContextFutureResult(lambda deps: self(deps).bind(function))

    def bind_async_future(self, function: Callable[[_ValueType], Awaitable[Future[_NewValueType]]]) -> RequiresContextFutureResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            print('Hello World!')
        "\n        Binds ``Future`` returning async function to the current container.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.future import Future\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> async def function(num: int) -> Future[int]:\n          ...     return Future.from_value(num + 1)\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_value(1).bind_async_future(\n          ...         function,\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(2)\n\n          >>> failed = RequiresContextFutureResult.from_failure(':(')\n          >>> assert anyio.run(\n          ...     failed.bind_async_future(function),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(':(')\n\n        "
        return RequiresContextFutureResult(lambda deps: self(deps).bind_async_future(function))

    def bind_async_future_result(self, function: Callable[[_ValueType], Awaitable[FutureResult[_NewValueType, _ErrorType]]]) -> RequiresContextFutureResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Bind ``FutureResult`` returning async function to the current container.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.future import FutureResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> async def function(num: int) -> FutureResult[int, str]:\n          ...     return FutureResult.from_value(num + 1)\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_value(\n          ...         1,\n          ...     ).bind_async_future_result(\n          ...         function,\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(2)\n\n          >>> failed = RequiresContextFutureResult.from_failure(':(')\n          >>> assert anyio.run(\n          ...     failed.bind_async_future_result(function),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(':(')\n\n        "
        return RequiresContextFutureResult(lambda deps: self(deps).bind_async(function))

    def alt(self, function: Callable[[_ErrorType], _NewErrorType]) -> RequiresContextFutureResult[_ValueType, _NewErrorType, _EnvType]:
        if False:
            print('Hello World!')
        '\n        Composes failed container with a pure function.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_value(1).alt(\n          ...        lambda x: x + 1,\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(1)\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_failure(1).alt(\n          ...        lambda x: x + 1,\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(2)\n\n        '
        return RequiresContextFutureResult(lambda deps: self(deps).alt(function))

    def lash(self, function: Callable[[_ErrorType], Kind3[RequiresContextFutureResult, _ValueType, _NewErrorType, _EnvType]]) -> RequiresContextFutureResult[_ValueType, _NewErrorType, _EnvType]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Composes this container with a function returning the same type.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.future import FutureResult\n          >>> from returns.io import IOSuccess\n\n          >>> def lashable(\n          ...     arg: str,\n          ... ) -> RequiresContextFutureResult[str, str, str]:\n          ...      return RequiresContextFutureResult(\n          ...          lambda deps: FutureResult.from_value(\n          ...              deps + arg,\n          ...          ),\n          ...      )\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_value('a').lash(lashable),\n          ...     'c',\n          ... ) == IOSuccess('a')\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_failure('aa').lash(\n          ...         lashable,\n          ...     ),\n          ...     'b',\n          ... ) == IOSuccess('baa')\n\n        "
        return RequiresContextFutureResult(lambda deps: self(deps).lash(lambda inner: function(inner)(deps)))

    def compose_result(self, function: Callable[[Result[_ValueType, _ErrorType]], Kind3[RequiresContextFutureResult, _NewValueType, _ErrorType, _EnvType]]) -> RequiresContextFutureResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Composes inner ``Result`` with ``ReaderFutureResult`` returning func.\n\n        Can be useful when you need an access to both states of the result.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import ReaderFutureResult, NoDeps\n          >>> from returns.io import IOSuccess, IOFailure\n          >>> from returns.result import Result\n\n          >>> def count(\n          ...    container: Result[int, int],\n          ... ) -> ReaderFutureResult[int, int, NoDeps]:\n          ...     return ReaderFutureResult.from_result(\n          ...         container.map(lambda x: x + 1).alt(abs),\n          ...     )\n\n          >>> success = ReaderFutureResult.from_value(1)\n          >>> failure = ReaderFutureResult.from_failure(-1)\n\n          >>> assert anyio.run(\n          ...     success.compose_result(count), ReaderFutureResult.no_args,\n          ... ) == IOSuccess(2)\n          >>> assert anyio.run(\n          ...     failure.compose_result(count), ReaderFutureResult.no_args,\n          ... ) == IOFailure(1)\n\n        '
        return RequiresContextFutureResult(lambda deps: FutureResult(_reader_future_result.async_compose_result(function, self, deps)))

    def modify_env(self, function: Callable[[_NewEnvType], _EnvType]) -> RequiresContextFutureResult[_ValueType, _ErrorType, _NewEnvType]:
        if False:
            print('Hello World!')
        "\n        Allows to modify the environment type.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.future import future_safe, asyncify\n          >>> from returns.context import RequiresContextFutureResultE\n          >>> from returns.io import IOSuccess\n\n          >>> def div(arg: int) -> RequiresContextFutureResultE[float, int]:\n          ...     return RequiresContextFutureResultE(\n          ...         future_safe(asyncify(lambda deps: arg / deps)),\n          ...     )\n\n          >>> assert anyio.run(div(3).modify_env(int), '2') == IOSuccess(1.5)\n          >>> assert anyio.run(div(3).modify_env(int), '0').failure()\n\n        "
        return RequiresContextFutureResult(lambda deps: self(function(deps)))

    @classmethod
    def ask(cls) -> RequiresContextFutureResult[_EnvType, _ErrorType, _EnvType]:
        if False:
            print('Hello World!')
        "\n        Is used to get the current dependencies inside the call stack.\n\n        Similar to\n        :meth:`returns.context.requires_context.RequiresContext.ask`,\n        but returns ``FutureResult`` instead of a regular value.\n\n        Please, refer to the docs there to learn how to use it.\n\n        One important note that is worth duplicating here:\n        you might need to provide type annotations explicitly,\n        so ``mypy`` will know about it statically.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResultE\n          >>> from returns.io import IOSuccess\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResultE[int, int].ask().map(str),\n          ...     1,\n          ... ) == IOSuccess('1')\n\n        "
        return RequiresContextFutureResult(FutureResult.from_value)

    @classmethod
    def from_result(cls, inner_value: Result[_NewValueType, _NewErrorType]) -> RequiresContextFutureResult[_NewValueType, _NewErrorType, NoDeps]:
        if False:
            return 10
        '\n        Creates new container with ``Result`` as a unit value.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.result import Success, Failure\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> assert anyio.run(\n          ...    RequiresContextFutureResult.from_result(Success(1)),\n          ...    RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(1)\n\n          >>> assert anyio.run(\n          ...    RequiresContextFutureResult.from_result(Failure(1)),\n          ...    RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(1)\n\n        '
        return RequiresContextFutureResult(lambda _: FutureResult.from_result(inner_value))

    @classmethod
    def from_io(cls, inner_value: IO[_NewValueType]) -> RequiresContextFutureResult[_NewValueType, Any, NoDeps]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates new container from successful ``IO`` value.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.io import IO, IOSuccess\n          >>> from returns.context import RequiresContextFutureResult\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_io(IO(1)),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(1)\n\n        '
        return RequiresContextFutureResult(lambda deps: FutureResult.from_io(inner_value))

    @classmethod
    def from_failed_io(cls, inner_value: IO[_NewErrorType]) -> RequiresContextFutureResult[Any, _NewErrorType, NoDeps]:
        if False:
            print('Hello World!')
        '\n        Creates a new container from failed ``IO`` value.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.io import IO, IOFailure\n          >>> from returns.context import RequiresContextFutureResult\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_failed_io(IO(1)),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(1)\n\n        '
        return RequiresContextFutureResult(lambda deps: FutureResult.from_failed_io(inner_value))

    @classmethod
    def from_ioresult(cls, inner_value: IOResult[_NewValueType, _NewErrorType]) -> RequiresContextFutureResult[_NewValueType, _NewErrorType, NoDeps]:
        if False:
            return 10
        '\n        Creates new container with ``IOResult`` as a unit value.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> assert anyio.run(\n          ...    RequiresContextFutureResult.from_ioresult(IOSuccess(1)),\n          ...    RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(1)\n\n          >>> assert anyio.run(\n          ...    RequiresContextFutureResult.from_ioresult(IOFailure(1)),\n          ...    RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(1)\n\n        '
        return RequiresContextFutureResult(lambda _: FutureResult.from_ioresult(inner_value))

    @classmethod
    def from_future(cls, inner_value: Future[_NewValueType]) -> RequiresContextFutureResult[_NewValueType, Any, NoDeps]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates new container with successful ``Future`` as a unit value.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.future import Future\n          >>> from returns.io import IOSuccess\n\n          >>> assert anyio.run(\n          ...    RequiresContextFutureResult.from_future(Future.from_value(1)),\n          ...    RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(1)\n\n        '
        return RequiresContextFutureResult(lambda _: FutureResult.from_future(inner_value))

    @classmethod
    def from_failed_future(cls, inner_value: Future[_NewErrorType]) -> RequiresContextFutureResult[Any, _NewErrorType, NoDeps]:
        if False:
            print('Hello World!')
        '\n        Creates new container with failed ``Future`` as a unit value.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.future import Future\n          >>> from returns.io import IOFailure\n\n          >>> assert anyio.run(\n          ...    RequiresContextFutureResult.from_failed_future(\n          ...        Future.from_value(1),\n          ...    ),\n          ...    RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(1)\n\n        '
        return RequiresContextFutureResult(lambda _: FutureResult.from_failed_future(inner_value))

    @classmethod
    def from_future_result_context(cls, inner_value: ReaderFutureResult[_NewValueType, _NewErrorType, _NewEnvType]) -> ReaderFutureResult[_NewValueType, _NewErrorType, _NewEnvType]:
        if False:
            return 10
        '\n        Creates new container with ``ReaderFutureResult`` as a unit value.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> assert anyio.run(\n          ...    RequiresContextFutureResult.from_future_result_context(\n          ...        RequiresContextFutureResult.from_value(1),\n          ...    ),\n          ...    RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(1)\n\n          >>> assert anyio.run(\n          ...    RequiresContextFutureResult.from_future_result_context(\n          ...        RequiresContextFutureResult.from_failure(1),\n          ...    ),\n          ...    RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(1)\n\n        '
        return inner_value

    @classmethod
    def from_future_result(cls, inner_value: FutureResult[_NewValueType, _NewErrorType]) -> RequiresContextFutureResult[_NewValueType, _NewErrorType, NoDeps]:
        if False:
            print('Hello World!')
        '\n        Creates new container with ``FutureResult`` as a unit value.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.future import FutureResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> assert anyio.run(\n          ...    RequiresContextFutureResult.from_future_result(\n          ...        FutureResult.from_value(1),\n          ...    ),\n          ...    RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(1)\n\n          >>> assert anyio.run(\n          ...    RequiresContextFutureResult.from_future_result(\n          ...        FutureResult.from_failure(1),\n          ...    ),\n          ...    RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(1)\n\n        '
        return RequiresContextFutureResult(lambda _: inner_value)

    @classmethod
    def from_typecast(cls, inner_value: RequiresContext[FutureResult[_NewValueType, _NewErrorType], _EnvType]) -> RequiresContextFutureResult[_NewValueType, _NewErrorType, _EnvType]:
        if False:
            return 10
        '\n        You might end up with ``RequiresContext[FutureResult]`` as a value.\n\n        This method is designed to turn it into ``RequiresContextFutureResult``.\n        It will save all the typing information.\n\n        It is just more useful!\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContext\n          >>> from returns.future import FutureResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_typecast(\n          ...         RequiresContext.from_value(FutureResult.from_value(1)),\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(1)\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_typecast(\n          ...         RequiresContext.from_value(FutureResult.from_failure(1)),\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(1)\n\n        '
        return RequiresContextFutureResult(inner_value)

    @classmethod
    def from_context(cls, inner_value: RequiresContext[_NewValueType, _NewEnvType]) -> RequiresContextFutureResult[_NewValueType, Any, _NewEnvType]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates new container from ``RequiresContext`` as a success unit.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContext\n          >>> from returns.io import IOSuccess\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_context(\n          ...         RequiresContext.from_value(1),\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(1)\n\n        '
        return RequiresContextFutureResult(lambda deps: FutureResult.from_value(inner_value(deps)))

    @classmethod
    def from_failed_context(cls, inner_value: RequiresContext[_NewValueType, _NewEnvType]) -> RequiresContextFutureResult[Any, _NewValueType, _NewEnvType]:
        if False:
            print('Hello World!')
        '\n        Creates new container from ``RequiresContext`` as a failure unit.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContext\n          >>> from returns.io import IOFailure\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_failed_context(\n          ...         RequiresContext.from_value(1),\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(1)\n\n        '
        return RequiresContextFutureResult(lambda deps: FutureResult.from_failure(inner_value(deps)))

    @classmethod
    def from_result_context(cls, inner_value: RequiresContextResult[_NewValueType, _NewErrorType, _NewEnvType]) -> ReaderFutureResult[_NewValueType, _NewErrorType, _NewEnvType]:
        if False:
            print('Hello World!')
        '\n        Creates new container from ``RequiresContextResult`` as a unit value.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_result_context(\n          ...         RequiresContextResult.from_value(1),\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(1)\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_result_context(\n          ...         RequiresContextResult.from_failure(1),\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(1)\n\n        '
        return RequiresContextFutureResult(lambda deps: FutureResult.from_result(inner_value(deps)))

    @classmethod
    def from_ioresult_context(cls, inner_value: ReaderIOResult[_NewValueType, _NewErrorType, _NewEnvType]) -> ReaderFutureResult[_NewValueType, _NewErrorType, _NewEnvType]:
        if False:
            i = 10
            return i + 15
        '\n        Creates new container from ``RequiresContextIOResult`` as a unit value.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_ioresult_context(\n          ...         RequiresContextIOResult.from_value(1),\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOSuccess(1)\n\n          >>> assert anyio.run(\n          ...     RequiresContextFutureResult.from_ioresult_context(\n          ...         RequiresContextIOResult.from_failure(1),\n          ...     ),\n          ...     RequiresContextFutureResult.no_args,\n          ... ) == IOFailure(1)\n\n        '
        return RequiresContextFutureResult(lambda deps: FutureResult.from_ioresult(inner_value(deps)))

    @classmethod
    def from_value(cls, inner_value: _FirstType) -> RequiresContextFutureResult[_FirstType, Any, NoDeps]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates new container with successful ``FutureResult`` as a unit value.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.io import IOSuccess\n\n          >>> assert anyio.run(RequiresContextFutureResult.from_value(1)(\n          ...    RequiresContextFutureResult.no_args,\n          ... ).awaitable) == IOSuccess(1)\n\n        '
        return RequiresContextFutureResult(lambda _: FutureResult.from_value(inner_value))

    @classmethod
    def from_failure(cls, inner_value: _FirstType) -> RequiresContextFutureResult[Any, _FirstType, NoDeps]:
        if False:
            return 10
        '\n        Creates new container with failed ``FutureResult`` as a unit value.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.context import RequiresContextFutureResult\n          >>> from returns.io import IOFailure\n\n          >>> assert anyio.run(RequiresContextFutureResult.from_failure(1)(\n          ...     RequiresContextFutureResult.no_args,\n          ... ).awaitable) == IOFailure(1)\n\n        '
        return RequiresContextFutureResult(lambda _: FutureResult.from_failure(inner_value))
RequiresContextFutureResultE = RequiresContextFutureResult[_ValueType, Exception, _EnvType]
ReaderFutureResult = RequiresContextFutureResult
ReaderFutureResultE = RequiresContextFutureResult[_ValueType, Exception, _EnvType]