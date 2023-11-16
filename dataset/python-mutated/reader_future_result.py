from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Awaitable, Callable, ClassVar, Sequence, TypeVar, final
from returns.interfaces.specific import future_result, reader, reader_ioresult
from returns.primitives.asserts import assert_equal
from returns.primitives.hkt import KindN
from returns.primitives.laws import Law, Law2, Lawful, LawSpecDef, law_definition
if TYPE_CHECKING:
    from returns.context import ReaderFutureResult
    from returns.future import FutureResult
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_ValueType = TypeVar('_ValueType')
_ErrorType = TypeVar('_ErrorType')
_EnvType = TypeVar('_EnvType')
_ReaderFutureResultLikeType = TypeVar('_ReaderFutureResultLikeType', bound='ReaderFutureResultLikeN')

class ReaderFutureResultLikeN(reader_ioresult.ReaderIOResultLikeN[_FirstType, _SecondType, _ThirdType], future_result.FutureResultLikeN[_FirstType, _SecondType, _ThirdType]):
    """
    Interface for all types that do look like ``ReaderFutureResult`` instance.

    Cannot be called.
    """
    __slots__ = ()

    @abstractmethod
    def bind_context_future_result(self: _ReaderFutureResultLikeType, function: Callable[[_FirstType], ReaderFutureResult[_UpdatedType, _SecondType, _ThirdType]]) -> KindN[_ReaderFutureResultLikeType, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            print('Hello World!')
        'Bind a ``ReaderFutureResult`` returning function over a container.'

    @abstractmethod
    def bind_async_context_future_result(self: _ReaderFutureResultLikeType, function: Callable[[_FirstType], Awaitable[ReaderFutureResult[_UpdatedType, _SecondType, _ThirdType],]]) -> KindN[_ReaderFutureResultLikeType, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            print('Hello World!')
        'Bind async ``ReaderFutureResult`` function.'

    @classmethod
    @abstractmethod
    def from_future_result_context(cls: type[_ReaderFutureResultLikeType], inner_value: ReaderFutureResult[_ValueType, _ErrorType, _EnvType]) -> KindN[_ReaderFutureResultLikeType, _ValueType, _ErrorType, _EnvType]:
        if False:
            while True:
                i = 10
        'Unit method to create new containers from ``ReaderFutureResult``.'
ReaderFutureResultLike3 = ReaderFutureResultLikeN[_FirstType, _SecondType, _ThirdType]

@final
class _LawSpec(LawSpecDef):
    """
    Concrete laws for ``ReaderFutureResultBasedN``.

    See: https://github.com/haskell/mtl/pull/61/files
    """
    __slots__ = ()

    @law_definition
    def asking_law(container: ReaderFutureResultBasedN[_FirstType, _SecondType, _ThirdType], env: _ThirdType) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Asking for an env, always returns the env.'
        assert_equal(container.ask().__call__(env), container.from_value(env).__call__(env))

class ReaderFutureResultBasedN(ReaderFutureResultLikeN[_FirstType, _SecondType, _ThirdType], reader.CallableReader3[_FirstType, _SecondType, _ThirdType, 'FutureResult[_FirstType, _SecondType]', _ThirdType], Lawful['ReaderFutureResultBasedN[_FirstType, _SecondType, _ThirdType]']):
    """
    This interface is very specific to our ``ReaderFutureResult`` type.

    The only thing that differs from ``ReaderFutureResultLikeN``
    is that we know the specific types for its ``__call__`` method.

    In this case the return type of ``__call__`` is ``FutureResult``.
    """
    __slots__ = ()
    _laws: ClassVar[Sequence[Law]] = (Law2(_LawSpec.asking_law),)
ReaderFutureResultBased3 = ReaderFutureResultBasedN[_FirstType, _SecondType, _ThirdType]