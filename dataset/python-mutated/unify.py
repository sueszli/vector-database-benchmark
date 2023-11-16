from typing import Callable, TypeVar, Union
from returns.interfaces.failable import DiverseFailableN
from returns.primitives.hkt import Kinded, KindN, kinded
_FirstType = TypeVar('_FirstType')
_NewFirstType = TypeVar('_NewFirstType')
_SecondType = TypeVar('_SecondType')
_NewSecondType = TypeVar('_NewSecondType')
_ThirdType = TypeVar('_ThirdType')
_NewThirdType = TypeVar('_NewThirdType')
_DiverseFailableKind = TypeVar('_DiverseFailableKind', bound=DiverseFailableN)

def unify(function: Callable[[_FirstType], KindN[_DiverseFailableKind, _NewFirstType, _NewSecondType, _NewThirdType]]) -> Kinded[Callable[[KindN[_DiverseFailableKind, _FirstType, _SecondType, _ThirdType]], KindN[_DiverseFailableKind, _NewFirstType, Union[_SecondType, _NewSecondType], _NewThirdType]]]:
    if False:
        while True:
            i = 10
    '\n    Composes successful container with a function that returns a container.\n\n    Similar to :func:`~returns.pointfree.bind` but has different type.\n    It returns ``Result[ValueType, Union[OldErrorType, NewErrorType]]``\n    instead of ``Result[ValueType, OldErrorType]``.\n\n    So, it can be more useful in some situations.\n    Probably with specific exceptions.\n\n    .. code:: python\n\n      >>> from returns.methods import cond\n      >>> from returns.pointfree import unify\n      >>> from returns.result import Result, Success, Failure\n\n      >>> def bindable(arg: int) -> Result[int, int]:\n      ...     return cond(Result, arg % 2 == 0, arg + 1, arg - 1)\n\n      >>> assert unify(bindable)(Success(2)) == Success(3)\n      >>> assert unify(bindable)(Success(1)) == Failure(0)\n      >>> assert unify(bindable)(Failure(42)) == Failure(42)\n\n    '

    @kinded
    def factory(container: KindN[_DiverseFailableKind, _FirstType, _SecondType, _ThirdType]) -> KindN[_DiverseFailableKind, _NewFirstType, Union[_SecondType, _NewSecondType], _NewThirdType]:
        if False:
            while True:
                i = 10
        return container.bind(function)
    return factory