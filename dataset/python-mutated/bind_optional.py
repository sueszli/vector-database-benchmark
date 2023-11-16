from typing import Callable, Optional, TypeVar
from returns.interfaces.specific.maybe import MaybeLikeN
from returns.primitives.hkt import Kinded, KindN, kinded
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_MaybeLikeKind = TypeVar('_MaybeLikeKind', bound=MaybeLikeN)

def bind_optional(function: Callable[[_FirstType], Optional[_UpdatedType]]) -> Kinded[Callable[[KindN[_MaybeLikeKind, _FirstType, _SecondType, _ThirdType]], KindN[_MaybeLikeKind, _UpdatedType, _SecondType, _ThirdType]]]:
    if False:
        return 10
    "\n    Binds a function returning optional value over a container.\n\n    In other words, it modifies the function's\n    signature from:\n    ``a -> Optional[b]``\n    to:\n    ``Container[a] -> Container[b]``\n\n    .. code:: python\n\n      >>> from typing import Optional\n      >>> from returns.pointfree import bind_optional\n      >>> from returns.maybe import Some, Nothing\n\n      >>> def example(argument: int) -> Optional[int]:\n      ...     return argument + 1 if argument > 0 else None\n\n      >>> assert bind_optional(example)(Some(1)) == Some(2)\n      >>> assert bind_optional(example)(Some(0)) == Nothing\n      >>> assert bind_optional(example)(Nothing) == Nothing\n\n    Note, that this function works\n    for all containers with ``.bind_optional`` method.\n    See :class:`returns.primitives.interfaces.specific.maybe._MaybeLikeKind`\n    for more info.\n\n    "

    @kinded
    def factory(container: KindN[_MaybeLikeKind, _FirstType, _SecondType, _ThirdType]) -> KindN[_MaybeLikeKind, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            return 10
        return container.bind_optional(function)
    return factory