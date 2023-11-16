from typing import TypeVar, Union
from returns.interfaces.unwrappable import Unwrappable
from returns.pipeline import is_successful
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')

def unwrap_or_failure(container: Unwrappable[_FirstType, _SecondType]) -> Union[_FirstType, _SecondType]:
    if False:
        return 10
    "\n    Unwraps either successful or failed value.\n\n    .. code:: python\n\n      >>> from returns.io import IO, IOSuccess, IOFailure\n      >>> from returns.methods import unwrap_or_failure\n\n      >>> assert unwrap_or_failure(IOSuccess(1)) == IO(1)\n      >>> assert unwrap_or_failure(IOFailure('a')) == IO('a')\n\n    "
    if is_successful(container):
        return container.unwrap()
    return container.failure()