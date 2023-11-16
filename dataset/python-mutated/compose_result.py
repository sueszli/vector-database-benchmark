from typing import Callable, TypeVar
from returns.interfaces.specific.ioresult import IOResultLikeN
from returns.primitives.hkt import Kind3, Kinded, kinded
from returns.result import Result
_FirstType = TypeVar('_FirstType')
_NewFirstType = TypeVar('_NewFirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_IOResultLikeKind = TypeVar('_IOResultLikeKind', bound=IOResultLikeN)

def compose_result(function: Callable[[Result[_FirstType, _SecondType]], Kind3[_IOResultLikeKind, _NewFirstType, _SecondType, _ThirdType]]) -> Kinded[Callable[[Kind3[_IOResultLikeKind, _FirstType, _SecondType, _ThirdType]], Kind3[_IOResultLikeKind, _NewFirstType, _SecondType, _ThirdType]]]:
    if False:
        return 10
    "\n    Composes inner ``Result`` with ``IOResultLike`` returning function.\n\n    Can be useful when you need an access to both states of the result.\n\n    .. code:: python\n\n      >>> from returns.io import IOResult, IOSuccess, IOFailure\n      >>> from returns.pointfree import compose_result\n      >>> from returns.result import Result\n\n      >>> def modify_string(container: Result[str, str]) -> IOResult[str, str]:\n      ...     return IOResult.from_result(\n      ...         container.map(str.upper).alt(str.lower),\n      ...     )\n\n      >>> assert compose_result(modify_string)(\n      ...     IOSuccess('success')\n      ... ) == IOSuccess('SUCCESS')\n      >>> assert compose_result(modify_string)(\n      ...     IOFailure('FAILURE')\n      ... ) == IOFailure('failure')\n\n    "

    @kinded
    def factory(container: Kind3[_IOResultLikeKind, _FirstType, _SecondType, _ThirdType]) -> Kind3[_IOResultLikeKind, _NewFirstType, _SecondType, _ThirdType]:
        if False:
            i = 10
            return i + 15
        return container.compose_result(function)
    return factory