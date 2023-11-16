from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, ClassVar, TypeVar, final
from returns.context import NoDeps
from returns.interfaces.specific import reader_result
from returns.primitives.container import BaseContainer
from returns.primitives.hkt import Kind3, SupportsKind3, dekind
from returns.result import Failure, Result, Success
if TYPE_CHECKING:
    from returns.context.requires_context import RequiresContext
_EnvType = TypeVar('_EnvType', contravariant=True)
_NewEnvType = TypeVar('_NewEnvType')
_ValueType = TypeVar('_ValueType', covariant=True)
_NewValueType = TypeVar('_NewValueType')
_ErrorType = TypeVar('_ErrorType', covariant=True)
_NewErrorType = TypeVar('_NewErrorType')
_FirstType = TypeVar('_FirstType')

@final
class RequiresContextResult(BaseContainer, SupportsKind3['RequiresContextResult', _ValueType, _ErrorType, _EnvType], reader_result.ReaderResultBasedN[_ValueType, _ErrorType, _EnvType]):
    """
    The ``RequiresContextResult`` combinator.

    See :class:`returns.context.requires_context.RequiresContext` for more docs.

    This is just a handy wrapper around ``RequiresContext[Result[a, b], env]``
    which represents a context-dependent pure operation
    that might fail and return :class:`returns.result.Result`.

    It has several important differences from the regular ``Result`` classes.
    It does not have ``Success`` and ``Failure`` subclasses.
    Because, the computation is not yet performed.
    And we cannot know the type in advance.

    So, this is a thin wrapper, without any changes in logic.

    Why do we need this wrapper? That's just for better usability!

    .. code:: python

      >>> from returns.context import RequiresContext
      >>> from returns.result import Success, Result

      >>> def function(arg: int) -> Result[int, str]:
      ...      return Success(arg + 1)

      >>> # Without wrapper:
      >>> assert RequiresContext.from_value(Success(1)).map(
      ...     lambda result: result.bind(function),
      ... )(...) == Success(2)

      >>> # With wrapper:
      >>> assert RequiresContextResult.from_value(1).bind_result(
      ...     function,
      ... )(...) == Success(2)

    This way ``RequiresContextResult`` allows to simply work with:

    - raw values and pure functions
    - ``RequiresContext`` values and pure functions returning it
    - ``Result`` and functions returning it

    .. rubric:: Important implementation details

    Due it is meaning, ``RequiresContextResult``
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
    _inner_value: Callable[[_EnvType], Result[_ValueType, _ErrorType]]
    no_args: ClassVar[NoDeps] = object()

    def __init__(self, inner_value: Callable[[_EnvType], Result[_ValueType, _ErrorType]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Public constructor for this type. Also required for typing.\n\n        Only allows functions of kind ``* -> *``\n        and returning :class:`returns.result.Result` instances.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.result import Success\n          >>> str(RequiresContextResult(lambda deps: Success(deps + 1)))\n          '<RequiresContextResult: <function <lambda> at ...>>'\n\n        "
        super().__init__(inner_value)

    def __call__(self, deps: _EnvType) -> Result[_ValueType, _ErrorType]:
        if False:
            print('Hello World!')
        '\n        Evaluates the wrapped function.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.result import Success\n\n          >>> def first(lg: bool) -> RequiresContextResult[int, str, float]:\n          ...     # `deps` has `float` type here:\n          ...     return RequiresContextResult(\n          ...         lambda deps: Success(deps if lg else -deps),\n          ...     )\n\n          >>> instance = first(False)\n          >>> assert instance(3.5) == Success(-3.5)\n\n        In other things, it is a regular Python magic method.\n\n        '
        return self._inner_value(deps)

    def swap(self) -> RequiresContextResult[_ErrorType, _ValueType, _EnvType]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Swaps value and error types.\n\n        So, values become errors and errors become values.\n        It is useful when you have to work with errors a lot.\n        And since we have a lot of ``.bind_`` related methods\n        and only a single ``.lash`` - it is easier to work with values.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.result import Failure, Success\n\n          >>> success = RequiresContextResult.from_value(1)\n          >>> failure = RequiresContextResult.from_failure(1)\n\n          >>> assert success.swap()(...) == Failure(1)\n          >>> assert failure.swap()(...) == Success(1)\n\n        '
        return RequiresContextResult(lambda deps: self(deps).swap())

    def map(self, function: Callable[[_ValueType], _NewValueType]) -> RequiresContextResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Composes successful container with a pure function.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.result import Success, Failure\n\n          >>> assert RequiresContextResult.from_value(1).map(\n          ...     lambda x: x + 1,\n          ... )(...) == Success(2)\n\n          >>> assert RequiresContextResult.from_failure(1).map(\n          ...     lambda x: x + 1,\n          ... )(...) == Failure(1)\n\n        '
        return RequiresContextResult(lambda deps: self(deps).map(function))

    def apply(self, container: Kind3[RequiresContextResult, Callable[[_ValueType], _NewValueType], _ErrorType, _EnvType]) -> RequiresContextResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            return 10
        "\n        Calls a wrapped function in a container on this container.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.result import Failure, Success\n\n          >>> def transform(arg: str) -> str:\n          ...     return arg + 'b'\n\n          >>> assert RequiresContextResult.from_value('a').apply(\n          ...    RequiresContextResult.from_value(transform),\n          ... )(...) == Success('ab')\n\n          >>> assert RequiresContextResult.from_failure('a').apply(\n          ...    RequiresContextResult.from_value(transform),\n          ... )(...) == Failure('a')\n\n          >>> assert isinstance(RequiresContextResult.from_value('a').apply(\n          ...    RequiresContextResult.from_failure(transform),\n          ... )(...), Failure) is True\n\n        "
        return RequiresContextResult(lambda deps: self(deps).apply(dekind(container)(deps)))

    def bind(self, function: Callable[[_ValueType], Kind3[RequiresContextResult, _NewValueType, _ErrorType, _EnvType]]) -> RequiresContextResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            print('Hello World!')
        "\n        Composes this container with a function returning the same type.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.result import Success, Failure\n\n          >>> def first(lg: bool) -> RequiresContextResult[int, int, float]:\n          ...     # `deps` has `float` type here:\n          ...     return RequiresContextResult(\n          ...         lambda deps: Success(deps) if lg else Failure(-deps),\n          ...     )\n\n          >>> def second(\n          ...     number: int,\n          ... ) -> RequiresContextResult[str, int, float]:\n          ...     # `deps` has `float` type here:\n          ...     return RequiresContextResult(\n          ...         lambda deps: Success('>=' if number >= deps else '<'),\n          ...     )\n\n          >>> assert first(True).bind(second)(1) == Success('>=')\n          >>> assert first(False).bind(second)(2) == Failure(-2)\n\n        "
        return RequiresContextResult(lambda deps: self(deps).bind(lambda inner: function(inner)(deps)))
    bind_context_result = bind

    def bind_result(self, function: Callable[[_ValueType], Result[_NewValueType, _ErrorType]]) -> RequiresContextResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            while True:
                i = 10
        "\n        Binds ``Result`` returning function to current container.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.result import Success, Failure, Result\n\n          >>> def function(num: int) -> Result[str, int]:\n          ...     return Success(num + 1) if num > 0 else Failure('<0')\n\n          >>> assert RequiresContextResult.from_value(1).bind_result(\n          ...     function,\n          ... )(RequiresContextResult.no_args) == Success(2)\n\n          >>> assert RequiresContextResult.from_value(0).bind_result(\n          ...     function,\n          ... )(RequiresContextResult.no_args) == Failure('<0')\n\n          >>> assert RequiresContextResult.from_failure(':(').bind_result(\n          ...     function,\n          ... )(RequiresContextResult.no_args) == Failure(':(')\n\n        "
        return RequiresContextResult(lambda deps: self(deps).bind(function))

    def bind_context(self, function: Callable[[_ValueType], RequiresContext[_NewValueType, _EnvType]]) -> RequiresContextResult[_NewValueType, _ErrorType, _EnvType]:
        if False:
            print('Hello World!')
        "\n        Binds ``RequiresContext`` returning function to current container.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n          >>> from returns.result import Success, Failure\n\n          >>> def function(arg: int) -> RequiresContext[int, str]:\n          ...     return RequiresContext(lambda deps: len(deps) + arg)\n\n          >>> assert function(2)('abc') == 5\n\n          >>> assert RequiresContextResult.from_value(2).bind_context(\n          ...     function,\n          ... )('abc') == Success(5)\n\n          >>> assert RequiresContextResult.from_failure(2).bind_context(\n          ...     function,\n          ... )('abc') == Failure(2)\n\n        "
        return RequiresContextResult(lambda deps: self(deps).map(lambda inner: function(inner)(deps)))

    def alt(self, function: Callable[[_ErrorType], _NewErrorType]) -> RequiresContextResult[_ValueType, _NewErrorType, _EnvType]:
        if False:
            i = 10
            return i + 15
        '\n        Composes failed container with a pure function.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.result import Success, Failure\n\n          >>> assert RequiresContextResult.from_value(1).alt(\n          ...     lambda x: x + 1,\n          ... )(...) == Success(1)\n\n          >>> assert RequiresContextResult.from_failure(1).alt(\n          ...     lambda x: x + 1,\n          ... )(...) == Failure(2)\n\n        '
        return RequiresContextResult(lambda deps: self(deps).alt(function))

    def lash(self, function: Callable[[_ErrorType], Kind3[RequiresContextResult, _ValueType, _NewErrorType, _EnvType]]) -> RequiresContextResult[_ValueType, _NewErrorType, _EnvType]:
        if False:
            print('Hello World!')
        "\n        Composes this container with a function returning the same type.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.result import Success, Failure\n\n          >>> def lashable(arg: str) -> RequiresContextResult[str, str, str]:\n          ...      if len(arg) > 1:\n          ...          return RequiresContextResult(\n          ...              lambda deps: Success(deps + arg),\n          ...          )\n          ...      return RequiresContextResult(\n          ...          lambda deps: Failure(arg + deps),\n          ...      )\n\n          >>> assert RequiresContextResult.from_value('a').lash(\n          ...     lashable,\n          ... )('c') == Success('a')\n          >>> assert RequiresContextResult.from_failure('a').lash(\n          ...     lashable,\n          ... )('c') == Failure('ac')\n          >>> assert RequiresContextResult.from_failure('aa').lash(\n          ...     lashable,\n          ... )('b') == Success('baa')\n\n        "
        return RequiresContextResult(lambda deps: self(deps).lash(lambda inner: function(inner)(deps)))

    def modify_env(self, function: Callable[[_NewEnvType], _EnvType]) -> RequiresContextResult[_ValueType, _ErrorType, _NewEnvType]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Allows to modify the environment type.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextResultE\n          >>> from returns.result import Success, safe\n\n          >>> def div(arg: int) -> RequiresContextResultE[float, int]:\n          ...     return RequiresContextResultE(\n          ...         safe(lambda deps: arg / deps),\n          ...     )\n\n          >>> assert div(3).modify_env(int)('2') == Success(1.5)\n          >>> assert div(3).modify_env(int)('0').failure()\n\n        "
        return RequiresContextResult(lambda deps: self(function(deps)))

    @classmethod
    def ask(cls) -> RequiresContextResult[_EnvType, _ErrorType, _EnvType]:
        if False:
            i = 10
            return i + 15
        "\n        Is used to get the current dependencies inside the call stack.\n\n        Similar to :meth:`returns.context.requires_context.RequiresContext.ask`,\n        but returns ``Result`` instead of a regular value.\n\n        Please, refer to the docs there to learn how to use it.\n\n        One important note that is worth duplicating here:\n        you might need to provide ``_EnvType`` explicitly,\n        so ``mypy`` will know about it statically.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextResultE\n          >>> from returns.result import Success\n          >>> assert RequiresContextResultE[int, int].ask().map(\n          ...    str,\n          ... )(1) == Success('1')\n\n        "
        return RequiresContextResult(Success)

    @classmethod
    def from_result(cls, inner_value: Result[_NewValueType, _NewErrorType]) -> RequiresContextResult[_NewValueType, _NewErrorType, NoDeps]:
        if False:
            return 10
        '\n        Creates new container with ``Result`` as a unit value.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.result import Success, Failure\n          >>> deps = RequiresContextResult.no_args\n\n          >>> assert RequiresContextResult.from_result(\n          ...    Success(1),\n          ... )(deps) == Success(1)\n\n          >>> assert RequiresContextResult.from_result(\n          ...    Failure(1),\n          ... )(deps) == Failure(1)\n\n        '
        return RequiresContextResult(lambda _: inner_value)

    @classmethod
    def from_typecast(cls, inner_value: RequiresContext[Result[_NewValueType, _NewErrorType], _EnvType]) -> RequiresContextResult[_NewValueType, _NewErrorType, _EnvType]:
        if False:
            return 10
        '\n        You might end up with ``RequiresContext[Result[...]]`` as a value.\n\n        This method is designed to turn it into ``RequiresContextResult``.\n        It will save all the typing information.\n\n        It is just more useful!\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n          >>> from returns.result import Success, Failure\n\n          >>> assert RequiresContextResult.from_typecast(\n          ...     RequiresContext.from_value(Success(1)),\n          ... )(RequiresContextResult.no_args) == Success(1)\n\n          >>> assert RequiresContextResult.from_typecast(\n          ...     RequiresContext.from_value(Failure(1)),\n          ... )(RequiresContextResult.no_args) == Failure(1)\n\n        '
        return RequiresContextResult(inner_value)

    @classmethod
    def from_context(cls, inner_value: RequiresContext[_NewValueType, _NewEnvType]) -> RequiresContextResult[_NewValueType, Any, _NewEnvType]:
        if False:
            while True:
                i = 10
        '\n        Creates new container from ``RequiresContext`` as a success unit.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n          >>> from returns.result import Success\n          >>> assert RequiresContextResult.from_context(\n          ...     RequiresContext.from_value(1),\n          ... )(...) == Success(1)\n\n        '
        return RequiresContextResult(lambda deps: Success(inner_value(deps)))

    @classmethod
    def from_failed_context(cls, inner_value: RequiresContext[_NewValueType, _NewEnvType]) -> RequiresContextResult[Any, _NewValueType, _NewEnvType]:
        if False:
            i = 10
            return i + 15
        '\n        Creates new container from ``RequiresContext`` as a failure unit.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContext\n          >>> from returns.result import Failure\n          >>> assert RequiresContextResult.from_failed_context(\n          ...     RequiresContext.from_value(1),\n          ... )(...) == Failure(1)\n\n        '
        return RequiresContextResult(lambda deps: Failure(inner_value(deps)))

    @classmethod
    def from_result_context(cls, inner_value: RequiresContextResult[_NewValueType, _NewErrorType, _NewEnvType]) -> RequiresContextResult[_NewValueType, _NewErrorType, _NewEnvType]:
        if False:
            return 10
        '\n        Creates ``RequiresContextResult`` from another instance of it.\n\n        .. code:: python\n\n          >>> from returns.context import ReaderResult\n          >>> from returns.result import Success, Failure\n\n          >>> assert ReaderResult.from_result_context(\n          ...     ReaderResult.from_value(1),\n          ... )(...) == Success(1)\n\n          >>> assert ReaderResult.from_result_context(\n          ...     ReaderResult.from_failure(1),\n          ... )(...) == Failure(1)\n\n        '
        return inner_value

    @classmethod
    def from_value(cls, inner_value: _FirstType) -> RequiresContextResult[_FirstType, Any, NoDeps]:
        if False:
            while True:
                i = 10
        '\n        Creates new container with ``Success(inner_value)`` as a unit value.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.result import Success\n          >>> assert RequiresContextResult.from_value(1)(...) == Success(1)\n\n        '
        return RequiresContextResult(lambda _: Success(inner_value))

    @classmethod
    def from_failure(cls, inner_value: _FirstType) -> RequiresContextResult[Any, _FirstType, NoDeps]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates new container with ``Failure(inner_value)`` as a unit value.\n\n        .. code:: python\n\n          >>> from returns.context import RequiresContextResult\n          >>> from returns.result import Failure\n          >>> assert RequiresContextResult.from_failure(1)(...) == Failure(1)\n\n        '
        return RequiresContextResult(lambda _: Failure(inner_value))
RequiresContextResultE = RequiresContextResult[_ValueType, Exception, _EnvType]
ReaderResult = RequiresContextResult
ReaderResultE = RequiresContextResult[_ValueType, Exception, _EnvType]