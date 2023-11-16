from collections.abc import Callable
from typing import TypeVar
T = TypeVar('T')

def docs_argspec(argspec: str) -> Callable[[T], T]:
    if False:
        for i in range(10):
            print('nop')
    'Override the argspec of the function in the documentation.\n\n    This is defined as a no-op here, but is overridden when building the docs.\n\n    It is not easy to satisfy mypy so frequently we have to put something like\n    *args: Any, **kwargs: Any as the type for the method itself. This makes the\n    docs look bad because they ignore the overloads. This allows us to satisfy\n    mypy but also render something better in the docs. See implementation in\n    docs/conf.py.\n    '

    def dec(func):
        if False:
            while True:
                i = 10
        return func
    return dec
import builtins
globals()['docs_argspec'] = getattr(builtins, '--docs_argspec--', docs_argspec)
__all__ = ['docs_argspec']