from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, ClassVar, TypeVar, final
from returns.context import NoDeps
from returns.interfaces.specific import reader_ioresult
from returns.io import IO, IOFailure, IOResult, IOSuccess
from returns.primitives.container import BaseContainer
from returns.primitives.hkt import Kind3, SupportsKind3, dekind
from returns.result import Result
if TYPE_CHECKING:
    from returns.context.requires_context import RequiresContext
    from returns.context.requires_context_result import RequiresContextResult
_EnvType = TypeVar('_EnvType', contravariant=True)
_NewEnvType = TypeVar('_NewEnvType')
_ValueType = TypeVar('_ValueType', covariant=True)
_NewValueType = TypeVar('_NewValueType')
_ErrorType = TypeVar('_ErrorType')
_NewErrorType = TypeVar('_NewErrorType')
_FirstType = TypeVar('_FirstType')

@final
class RequiresContextIOResult(BaseContainer, SupportsKind3['RequiresContextIOResult', _ValueType, _ErrorType, _EnvType], reader_ioresult.ReaderIOResultBasedN[_ValueType, _ErrorType, _EnvType]):
    """
    The ``RequiresContextIOResult`` combinator.

    See :class:`returns.context.requires_context.RequiresContext`
    and :class:`returns.context.requires_context_result.RequiresContextResult`
    for more docs.

    This is just a handy wrapper around
    ``RequiresContext[IOResult[a, b], env]``
    which represents a context-dependent impure operation that might fail.

    It has several important differences from the regular ``Result`` classes.
    It does not have ``Success`` and ``Failure`` subclasses.
    Because, the computation is not yet performed.
    And we cannot know the type in advance.

    So, this is a thin wrapper, without any changes in logic.

    Why do we need this wrapper? That's just for better usability!

    .. code:: python

      >>> from returns.context import RequiresContext
      >>> from returns.io import IOSuccess, IOResult

      >>> def function(arg: int) -> IOResult[int, str]:
      ...      return IOSuccess(arg + 1)

      >>> # Without wrapper:
      >>> assert RequiresContext.from_value(IOSuccess(1)).map(
      ...     lambda ioresult: ioresult.bind(function),
      ... )(...) == IOSuccess(2)

      >>> # With wrapper:
      >>> assert RequiresContextIOResult.from_value(1).bind_ioresult(
      ...     function,
      ... )(...) == IOSuccess(2)

    This way ``RequiresContextIOResult`` allows to simply work with:

    - raw values and pure functions
    - ``RequiresContext`` values and pure functions returning it
    - ``RequiresContextResult`` values and pure functions returning it
    - ``Result`` and pure functions returning it
    - ``IOResult`` and functions returning it
    - other ``RequiresContextIOResult`` related functions and values

    This is a complex type for complex tasks!

    .. rubric:: Important implementation details

    Due it is meaning, ``RequiresContextIOResult``
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
    _inner_value: Callable[[_EnvType], IOResult[_ValueType, _ErrorType]]
    no_args: ClassVar[NoDeps] = object()

    def __init__(self, inner_value: Callable[[_EnvType], IOResult[_ValueType, _ErrorType]]) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Public constructor for this type. Also required for typing.\n\n        Only allows functions of kind ``* -> *``\n        and returning :class:`returns.result.Result` instances.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IOSuccess\n          >>> str(RequiresContextIOResult(lambda deps: IOSuccess(deps + 1)))\n          '<RequiresContextIOResult: <function <lambda> at ...>>'\n\n        "
        super().__init__(inner_value)

    def __call__(self, deps: _EnvType) -> IOResult[_ValueType, _ErrorType]:
        if False:
            while True:
                i = 10
        '\n        Evaluates the wrapped function.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IOSuccess\n\n          >>> def first(lg: bool) -> RequiresContextIOResult[int, str, float]:\n          ...     # `deps` has `float` type here:\n          ...     return RequiresContextIOResult(\n          ...         lambda deps: IOSuccess(deps if lg else -deps),\n          ...     )\n\n          >>> instance = first(False)\n          >>> assert instance(3.5) == IOSuccess(-3.5)\n\n        In other things, it is a regular Python magic method.\n\n        '
        return self._inner_value(deps)

    def swap(self) -> RequiresContextIOResult[_ErrorType, _ValueType, _EnvType]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Swaps value and error types.\n\n        So, values become errors and errors become values.\n        It is useful when you have to work with errors a lot.\n        And since we have a lot of ``.bind_`` related methods\n        and only a single ``.lash`` - it is easier to work with values.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> success = RequiresContextIOResult.from_value(1)\n          >>> failure = RequiresContextIOResult.from_failure(1)\n\n          >>> assert success.swap()(...) == IOFailure(1)\n          >>> assert failure.swap()(...) == IOSuccess(1)\n\n        '
        return RequiresContextIOResult(lambda deps: self(deps).swap())

    def map(self, function: Callable[[_ValueType], _NewValueType]) -> RequiresContextIOResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            print('Hello World!')
        '\n        Composes successful container with a pure function.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> assert RequiresContextIOResult.from_value(1).map(\n          ...     lambda x: x + 1,\n          ... )(...) == IOSuccess(2)\n\n          >>> assert RequiresContextIOResult.from_failure(1).map(\n          ...     lambda x: x + 1,\n          ... )(...) == IOFailure(1)\n\n        '
        return RequiresContextIOResult(lambda deps: self(deps).map(function))

    def apply(self, container: Kind3[RequiresContextIOResult, Callable[[_ValueType], _NewValueType], _ErrorType, _EnvType]) -> RequiresContextIOResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            return 10
        "\n        Calls a wrapped function in a container on this container.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> def transform(arg: str) -> str:\n          ...     return arg + 'b'\n\n          >>> assert RequiresContextIOResult.from_value('a').apply(\n          ...    RequiresContextIOResult.from_value(transform),\n          ... )(...) == IOSuccess('ab')\n\n          >>> assert RequiresContextIOResult.from_value('a').apply(\n          ...    RequiresContextIOResult.from_failure(1),\n          ... )(...) == IOFailure(1)\n\n          >>> assert RequiresContextIOResult.from_failure('a').apply(\n          ...    RequiresContextIOResult.from_value(transform),\n          ... )(...) == IOFailure('a')\n\n          >>> assert RequiresContextIOResult.from_failure('a').apply(\n          ...    RequiresContextIOResult.from_failure('b'),\n          ... )(...) == IOFailure('a')\n\n        "
        return RequiresContextIOResult(lambda deps: self(deps).apply(dekind(container)(deps)))

    def bind(self, function: Callable[[_ValueType], Kind3[RequiresContextIOResult, _NewValueType, _ErrorType, _EnvType]]) -> RequiresContextIOResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            i = 10
            return i + 15
        "\n        Composes this container with a function returning the same type.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> def first(lg: bool) -> RequiresContextIOResult[int, int, float]:\n          ...     # `deps` has `float` type here:\n          ...     return RequiresContextIOResult(\n          ...         lambda deps: IOSuccess(deps) if lg else IOFailure(-deps),\n          ...     )\n\n          >>> def second(\n          ...     number: int,\n          ... ) -> RequiresContextIOResult[str, int, float]:\n          ...     # `deps` has `float` type here:\n          ...     return RequiresContextIOResult(\n          ...         lambda deps: IOSuccess('>=' if number >= deps else '<'),\n          ...     )\n\n          >>> assert first(True).bind(second)(1) == IOSuccess('>=')\n          >>> assert first(False).bind(second)(2) == IOFailure(-2)\n\n        "
        return RequiresContextIOResult(lambda deps: self(deps).bind(lambda inner: dekind(function(inner))(deps)))
    bind_context_ioresult = bind

    def bind_result(self, function: Callable[[_ValueType], Result[_NewValueType, _ErrorType]]) -> RequiresContextIOResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            return 10
        "\n        Binds ``Result`` returning function to the current container.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.result import Failure, Result, Success\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> def function(num: int) -> Result[int, str]:\n          ...     return Success(num + 1) if num > 0 else Failure('<0')\n\n          >>> assert RequiresContextIOResult.from_value(1).bind_result(\n          ...     function,\n          ... )(RequiresContextIOResult.no_args) == IOSuccess(2)\n\n          >>> assert RequiresContextIOResult.from_value(0).bind_result(\n          ...     function,\n          ... )(RequiresContextIOResult.no_args) == IOFailure('<0')\n\n          >>> assert RequiresContextIOResult.from_failure(':(').bind_result(\n          ...     function,\n          ... )(RequiresContextIOResult.no_args) == IOFailure(':(')\n\n        "
        return RequiresContextIOResult(lambda deps: self(deps).bind_result(function))

    def bind_context(self, function: Callable[[_ValueType], RequiresContext[_NewValueType, _EnvType]]) -> RequiresContextIOResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Binds ``RequiresContext`` returning function to current container.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> def function(arg: int) -> RequiresContext[int, str]:\n          ...     return RequiresContext(lambda deps: len(deps) + arg)\n\n          >>> assert function(2)('abc') == 5\n\n          >>> assert RequiresContextIOResult.from_value(2).bind_context(\n          ...     function,\n          ... )('abc') == IOSuccess(5)\n\n          >>> assert RequiresContextIOResult.from_failure(2).bind_context(\n          ...     function,\n          ... )('abc') == IOFailure(2)\n\n        "
        return RequiresContextIOResult(lambda deps: self(deps).map(lambda inner: function(inner)(deps)))

    def bind_context_result(self, function: Callable[[_ValueType], RequiresContextResult[_NewValueType, _ErrorType, _EnvType]]) -> RequiresContextIOResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            return 10
        "\n        Binds ``RequiresContextResult`` returning function to the current one.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.io import IOSuccess, IOFailure\n          >>> from returns.result import Success, Failure\n\n          >>> def function(arg: int) -> RequiresContextResult[int, int, str]:\n          ...     if arg > 0:\n          ...         return RequiresContextResult(\n          ...             lambda deps: Success(len(deps) + arg),\n          ...         )\n          ...     return RequiresContextResult(\n          ...         lambda deps: Failure(len(deps) + arg),\n          ...     )\n\n          >>> assert function(2)('abc') == Success(5)\n          >>> assert function(-1)('abc') == Failure(2)\n\n          >>> assert RequiresContextIOResult.from_value(\n          ...    2,\n          ... ).bind_context_result(\n          ...     function,\n          ... )('abc') == IOSuccess(5)\n\n          >>> assert RequiresContextIOResult.from_value(\n          ...    -1,\n          ... ).bind_context_result(\n          ...     function,\n          ... )('abc') == IOFailure(2)\n\n          >>> assert RequiresContextIOResult.from_failure(\n          ...    2,\n          ... ).bind_context_result(\n          ...     function,\n          ... )('abc') == IOFailure(2)\n\n        "
        return RequiresContextIOResult(lambda deps: self(deps).bind_result(lambda inner: function(inner)(deps)))

    def bind_io(self, function: Callable[[_ValueType], IO[_NewValueType]]) -> RequiresContextIOResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Binds ``IO`` returning function to the current container.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IO, IOSuccess, IOFailure\n\n          >>> def function(number: int) -> IO[str]:\n          ...     return IO(str(number))\n\n          >>> assert RequiresContextIOResult.from_value(1).bind_io(\n          ...     function,\n          ... )(RequiresContextIOResult.no_args) == IOSuccess('1')\n\n          >>> assert RequiresContextIOResult.from_failure(1).bind_io(\n          ...     function,\n          ... )(RequiresContextIOResult.no_args) == IOFailure(1)\n\n        "
        return RequiresContextIOResult(lambda deps: self(deps).bind_io(function))

    def bind_ioresult(self, function: Callable[[_ValueType], IOResult[_NewValueType, _ErrorType]]) -> RequiresContextIOResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Binds ``IOResult`` returning function to the current container.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IOResult, IOSuccess, IOFailure\n\n          >>> def function(num: int) -> IOResult[int, str]:\n          ...     return IOSuccess(num + 1) if num > 0 else IOFailure('<0')\n\n          >>> assert RequiresContextIOResult.from_value(1).bind_ioresult(\n          ...     function,\n          ... )(RequiresContextIOResult.no_args) == IOSuccess(2)\n\n          >>> assert RequiresContextIOResult.from_value(0).bind_ioresult(\n          ...     function,\n          ... )(RequiresContextIOResult.no_args) == IOFailure('<0')\n\n          >>> assert RequiresContextIOResult.from_failure(':(').bind_ioresult(\n          ...     function,\n          ... )(RequiresContextIOResult.no_args) == IOFailure(':(')\n\n        "
        return RequiresContextIOResult(lambda deps: self(deps).bind(function))

    def alt(self, function: Callable[[_ErrorType], _NewErrorType]) -> RequiresContextIOResult[_ValueType, _NewErrorType, _EnvType]:
        if False:
            while True:
                i = 10
        '\n        Composes failed container with a pure function.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> assert RequiresContextIOResult.from_value(1).alt(\n          ...     lambda x: x + 1,\n          ... )(...) == IOSuccess(1)\n\n          >>> assert RequiresContextIOResult.from_failure(1).alt(\n          ...     lambda x: x + 1,\n          ... )(...) == IOFailure(2)\n\n        '
        return RequiresContextIOResult(lambda deps: self(deps).alt(function))

    def lash(self, function: Callable[[_ErrorType], Kind3[RequiresContextIOResult, _ValueType, _NewErrorType, _EnvType]]) -> RequiresContextIOResult[_ValueType, _NewErrorType, _EnvType]:
        if False:
            return 10
        "\n        Composes this container with a function returning the same type.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> def lashable(\n          ...     arg: str,\n          ... ) -> RequiresContextIOResult[str, str, str]:\n          ...      if len(arg) > 1:\n          ...          return RequiresContextIOResult(\n          ...              lambda deps: IOSuccess(deps + arg),\n          ...          )\n          ...      return RequiresContextIOResult(\n          ...          lambda deps: IOFailure(arg + deps),\n          ...      )\n\n          >>> assert RequiresContextIOResult.from_value('a').lash(\n          ...     lashable,\n          ... )('c') == IOSuccess('a')\n          >>> assert RequiresContextIOResult.from_failure('a').lash(\n          ...     lashable,\n          ... )('c') == IOFailure('ac')\n          >>> assert RequiresContextIOResult.from_failure('aa').lash(\n          ...     lashable,\n          ... )('b') == IOSuccess('baa')\n\n        "
        return RequiresContextIOResult(lambda deps: self(deps).lash(lambda inner: function(inner)(deps)))

    def compose_result(self, function: Callable[[Result[_ValueType, _ErrorType]], Kind3[RequiresContextIOResult, _NewValueType, _ErrorType, _EnvType]]) -> RequiresContextIOResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            return 10
        '\n        Composes inner ``Result`` with ``ReaderIOResult`` returning function.\n\n        Can be useful when you need an access to both states of the result.\n\n        .. code:: python\n\n          >>> from returns.context import ReaderIOResult, NoDeps\n          >>> from returns.io import IOSuccess, IOFailure\n          >>> from returns.result import Result\n\n          >>> def count(\n          ...    container: Result[int, int],\n          ... ) -> ReaderIOResult[int, int, NoDeps]:\n          ...     return ReaderIOResult.from_result(\n          ...         container.map(lambda x: x + 1).alt(abs),\n          ...     )\n\n          >>> success = ReaderIOResult.from_value(1)\n          >>> failure = ReaderIOResult.from_failure(-1)\n          >>> assert success.compose_result(count)(...) == IOSuccess(2)\n          >>> assert failure.compose_result(count)(...) == IOFailure(1)\n\n        '
        return RequiresContextIOResult(lambda deps: dekind(function(self(deps)._inner_value))(deps))

    def modify_env(self, function: Callable[[_NewEnvType], _EnvType]) -> RequiresContextIOResult[_ValueType, _ErrorType, _NewEnvType]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Allows to modify the environment type.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResultE\n          >>> from returns.io import IOSuccess, impure_safe\n\n          >>> def div(arg: int) -> RequiresContextIOResultE[float, int]:\n          ...     return RequiresContextIOResultE(\n          ...         impure_safe(lambda deps: arg / deps),\n          ...     )\n\n          >>> assert div(3).modify_env(int)('2') == IOSuccess(1.5)\n          >>> assert div(3).modify_env(int)('0').failure()\n\n        "
        return RequiresContextIOResult(lambda deps: self(function(deps)))

    @classmethod
    def ask(cls) -> RequiresContextIOResult[_EnvType, _ErrorType, _EnvType]:
        if False:
            return 10
        "\n        Is used to get the current dependencies inside the call stack.\n\n        Similar to :meth:`returns.context.requires_context.RequiresContext.ask`,\n        but returns ``IOResult`` instead of a regular value.\n\n        Please, refer to the docs there to learn how to use it.\n\n        One important note that is worth duplicating here:\n        you might need to provide ``_EnvType`` explicitly,\n        so ``mypy`` will know about it statically.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResultE\n          >>> from returns.io import IOSuccess\n          >>> assert RequiresContextIOResultE[int, int].ask().map(\n          ...     str,\n          ... )(1) == IOSuccess('1')\n\n        "
        return RequiresContextIOResult(IOSuccess)

    @classmethod
    def from_result(cls, inner_value: Result[_NewValueType, _NewErrorType]) -> RequiresContextIOResult[_NewValueType, _NewErrorType, NoDeps]:
        if False:
            i = 10
            return i + 15
        '\n        Creates new container with ``Result`` as a unit value.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.result import Success, Failure\n          >>> from returns.io import IOSuccess, IOFailure\n          >>> deps = RequiresContextIOResult.no_args\n\n          >>> assert RequiresContextIOResult.from_result(\n          ...    Success(1),\n          ... )(deps) == IOSuccess(1)\n\n          >>> assert RequiresContextIOResult.from_result(\n          ...    Failure(1),\n          ... )(deps) == IOFailure(1)\n\n        '
        return RequiresContextIOResult(lambda _: IOResult.from_result(inner_value))

    @classmethod
    def from_io(cls, inner_value: IO[_NewValueType]) -> RequiresContextIOResult[_NewValueType, Any, NoDeps]:
        if False:
            i = 10
            return i + 15
        '\n        Creates new container from successful ``IO`` value.\n\n        .. code:: python\n\n          >>> from returns.io import IO, IOSuccess\n          >>> from returns.context import RequiresContextIOResult\n\n          >>> assert RequiresContextIOResult.from_io(IO(1))(\n          ...     RequiresContextIOResult.no_args,\n          ... ) == IOSuccess(1)\n\n        '
        return RequiresContextIOResult(lambda deps: IOResult.from_io(inner_value))

    @classmethod
    def from_failed_io(cls, inner_value: IO[_NewErrorType]) -> RequiresContextIOResult[Any, _NewErrorType, NoDeps]:
        if False:
            print('Hello World!')
        '\n        Creates a new container from failed ``IO`` value.\n\n        .. code:: python\n\n          >>> from returns.io import IO, IOFailure\n          >>> from returns.context import RequiresContextIOResult\n\n          >>> assert RequiresContextIOResult.from_failed_io(IO(1))(\n          ...     RequiresContextIOResult.no_args,\n          ... ) == IOFailure(1)\n\n        '
        return RequiresContextIOResult(lambda deps: IOResult.from_failed_io(inner_value))

    @classmethod
    def from_ioresult(cls, inner_value: IOResult[_NewValueType, _NewErrorType]) -> RequiresContextIOResult[_NewValueType, _NewErrorType, NoDeps]:
        if False:
            print('Hello World!')
        '\n        Creates new container with ``IOResult`` as a unit value.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IOSuccess, IOFailure\n          >>> deps = RequiresContextIOResult.no_args\n\n          >>> assert RequiresContextIOResult.from_ioresult(\n          ...    IOSuccess(1),\n          ... )(deps) == IOSuccess(1)\n\n          >>> assert RequiresContextIOResult.from_ioresult(\n          ...    IOFailure(1),\n          ... )(deps) == IOFailure(1)\n\n        '
        return RequiresContextIOResult(lambda _: inner_value)

    @classmethod
    def from_ioresult_context(cls, inner_value: ReaderIOResult[_NewValueType, _NewErrorType, _NewEnvType]) -> ReaderIOResult[_NewValueType, _NewErrorType, _NewEnvType]:
        if False:
            return 10
        '\n        Creates new container with ``ReaderIOResult`` as a unit value.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> assert RequiresContextIOResult.from_ioresult_context(\n          ...    RequiresContextIOResult.from_value(1),\n          ... )(...) == IOSuccess(1)\n\n          >>> assert RequiresContextIOResult.from_ioresult_context(\n          ...    RequiresContextIOResult.from_failure(1),\n          ... )(...) == IOFailure(1)\n\n        '
        return inner_value

    @classmethod
    def from_typecast(cls, inner_value: RequiresContext[IOResult[_NewValueType, _NewErrorType], _EnvType]) -> RequiresContextIOResult[_NewValueType, _NewErrorType, _EnvType]:
        if False:
            i = 10
            return i + 15
        '\n        You might end up with ``RequiresContext[IOResult]`` as a value.\n\n        This method is designed to turn it into ``RequiresContextIOResult``.\n        It will save all the typing information.\n\n        It is just more useful!\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> assert RequiresContextIOResult.from_typecast(\n          ...     RequiresContext.from_value(IOSuccess(1)),\n          ... )(RequiresContextIOResult.no_args) == IOSuccess(1)\n\n          >>> assert RequiresContextIOResult.from_typecast(\n          ...     RequiresContext.from_value(IOFailure(1)),\n          ... )(RequiresContextIOResult.no_args) == IOFailure(1)\n\n        '
        return RequiresContextIOResult(inner_value)

    @classmethod
    def from_context(cls, inner_value: RequiresContext[_NewValueType, _NewEnvType]) -> RequiresContextIOResult[_NewValueType, Any, _NewEnvType]:
        if False:
            i = 10
            return i + 15
        '\n        Creates new container from ``RequiresContext`` as a success unit.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n          >>> from returns.io import IOSuccess\n\n          >>> assert RequiresContextIOResult.from_context(\n          ...     RequiresContext.from_value(1),\n          ... )(...) == IOSuccess(1)\n\n        '
        return RequiresContextIOResult(lambda deps: IOSuccess(inner_value(deps)))

    @classmethod
    def from_failed_context(cls, inner_value: RequiresContext[_NewValueType, _NewEnvType]) -> RequiresContextIOResult[Any, _NewValueType, _NewEnvType]:
        if False:
            print('Hello World!')
        '\n        Creates new container from ``RequiresContext`` as a failure unit.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n          >>> from returns.io import IOFailure\n\n          >>> assert RequiresContextIOResult.from_failed_context(\n          ...     RequiresContext.from_value(1),\n          ... )(...) == IOFailure(1)\n\n        '
        return RequiresContextIOResult(lambda deps: IOFailure(inner_value(deps)))

    @classmethod
    def from_result_context(cls, inner_value: RequiresContextResult[_NewValueType, _NewErrorType, _NewEnvType]) -> RequiresContextIOResult[_NewValueType, _NewErrorType, _NewEnvType]:
        if False:
            while True:
                i = 10
        '\n        Creates new container from ``RequiresContextResult`` as a unit value.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.io import IOSuccess, IOFailure\n\n          >>> assert RequiresContextIOResult.from_result_context(\n          ...     RequiresContextResult.from_value(1),\n          ... )(...) == IOSuccess(1)\n\n          >>> assert RequiresContextIOResult.from_result_context(\n          ...     RequiresContextResult.from_failure(1),\n          ... )(...) == IOFailure(1)\n\n        '
        return RequiresContextIOResult(lambda deps: IOResult.from_result(inner_value(deps)))

    @classmethod
    def from_value(cls, inner_value: _NewValueType) -> RequiresContextIOResult[_NewValueType, Any, NoDeps]:
        if False:
            i = 10
            return i + 15
        '\n        Creates new container with ``IOSuccess(inner_value)`` as a unit value.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IOSuccess\n\n          >>> assert RequiresContextIOResult.from_value(1)(\n          ...    RequiresContextIOResult.no_args,\n          ... ) == IOSuccess(1)\n\n        '
        return RequiresContextIOResult(lambda _: IOSuccess(inner_value))

    @classmethod
    def from_failure(cls, inner_value: _NewErrorType) -> RequiresContextIOResult[Any, _NewErrorType, NoDeps]:
        if False:
            print('Hello World!')
        '\n        Creates new container with ``IOFailure(inner_value)`` as a unit value.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextIOResult\n          >>> from returns.io import IOFailure\n\n          >>> assert RequiresContextIOResult.from_failure(1)(\n          ...     RequiresContextIOResult.no_args,\n          ... ) == IOFailure(1)\n\n        '
        return RequiresContextIOResult(lambda _: IOFailure(inner_value))
RequiresContextIOResultE = RequiresContextIOResult[_ValueType, Exception, _EnvType]
ReaderIOResult = RequiresContextIOResult
ReaderIOResultE = RequiresContextIOResult[_ValueType, Exception, _EnvType]