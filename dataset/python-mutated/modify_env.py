from typing import Callable, TypeVar
from returns.interfaces.specific.reader import ReaderLike2, ReaderLike3
from returns.primitives.hkt import Kind2, Kind3, Kinded, kinded
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_Reader2Kind = TypeVar('_Reader2Kind', bound=ReaderLike2)
_Reader3Kind = TypeVar('_Reader3Kind', bound=ReaderLike3)

def modify_env2(function: Callable[[_UpdatedType], _SecondType]) -> Kinded[Callable[[Kind2[_Reader2Kind, _FirstType, _SecondType]], Kind2[_Reader2Kind, _FirstType, _UpdatedType]]]:
    if False:
        return 10
    "\n    Modifies the second type argument of a ``ReaderLike2``.\n\n    In other words, it modifies the function's\n    signature from:\n    ``a -> b``\n    to:\n    ``Container[x, a] -> Container[x, b]``\n\n    .. code:: python\n\n      >>> from returns.pointfree import modify_env2\n      >>> from returns.context import RequiresContext\n\n      >>> def multiply(arg: int) -> RequiresContext[int, int]:\n      ...     return RequiresContext(lambda deps: arg * deps)\n\n      >>> assert modify_env2(int)(multiply(3))('4') == 12\n\n    Note, that this function works with only ``Kind2`` containers\n    with ``.modify_env`` method.\n    See :class:`returns.primitives.interfaces.specific.reader.ReaderLike2`\n    for more info.\n\n    "

    @kinded
    def factory(container: Kind2[_Reader2Kind, _FirstType, _SecondType]) -> Kind2[_Reader2Kind, _FirstType, _UpdatedType]:
        if False:
            print('Hello World!')
        return container.modify_env(function)
    return factory

def modify_env3(function: Callable[[_UpdatedType], _ThirdType]) -> Kinded[Callable[[Kind3[_Reader3Kind, _FirstType, _SecondType, _ThirdType]], Kind3[_Reader3Kind, _FirstType, _SecondType, _UpdatedType]]]:
    if False:
        i = 10
        return i + 15
    "\n    Modifies the third type argument of a ``ReaderLike3``.\n\n    In other words, it modifies the function's\n    signature from: ``a -> b``\n    to: ``Container[x, a] -> Container[x, b]``\n\n    .. code:: python\n\n      >>> from returns.pointfree import modify_env\n      >>> from returns.context import RequiresContextResultE\n      >>> from returns.result import Success, safe\n\n      >>> def divide(arg: int) -> RequiresContextResultE[float, int]:\n      ...     return RequiresContextResultE(safe(lambda deps: arg / deps))\n\n      >>> assert modify_env(int)(divide(3))('2') == Success(1.5)\n      >>> assert modify_env(int)(divide(3))('0').failure()\n\n    Note, that this function works with only ``Kind3`` containers\n    with ``.modify_env`` method.\n    See :class:`returns.primitives.interfaces.specific.reader.ReaderLike3`\n    for more info.\n\n    "

    @kinded
    def factory(container: Kind3[_Reader3Kind, _FirstType, _SecondType, _ThirdType]) -> Kind3[_Reader3Kind, _FirstType, _SecondType, _UpdatedType]:
        if False:
            i = 10
            return i + 15
        return container.modify_env(function)
    return factory
modify_env = modify_env3