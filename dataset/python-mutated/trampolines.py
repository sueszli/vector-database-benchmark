from functools import wraps
from typing import Callable, Generic, TypeVar, Union, final
from typing_extensions import ParamSpec
_ReturnType = TypeVar('_ReturnType')
_FuncParams = ParamSpec('_FuncParams')

@final
class Trampoline(Generic[_ReturnType]):
    """
    Represents a wrapped function call.

    Primitive to convert recursion into an actual object.
    """
    __slots__ = ('func', 'args', 'kwargs')

    def __init__(self, func: Callable[_FuncParams, _ReturnType], /, *args: _FuncParams.args, **kwargs: _FuncParams.kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Save function and given arguments.'
        self.func = getattr(func, '_orig_func', func)
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> _ReturnType:
        if False:
            while True:
                i = 10
        'Call wrapped function with given arguments.'
        return self.func(*self.args, **self.kwargs)

def trampoline(func: Callable[_FuncParams, Union[_ReturnType, Trampoline[_ReturnType]]]) -> Callable[_FuncParams, _ReturnType]:
    if False:
        i = 10
        return i + 15
    '\n    Convert functions using recursion to regular functions.\n\n    Trampolines allow to unwrap recursion into a regular ``while`` loop,\n    which does not raise any ``RecursionError`` ever.\n\n    Since python does not have TCO (tail call optimization),\n    we have to provide this helper.\n\n    This is done by wrapping real function calls into\n    :class:`returns.trampolines.Trampoline` objects:\n\n    .. code:: python\n\n        >>> from typing import Union\n        >>> from returns.trampolines import Trampoline, trampoline\n\n        >>> @trampoline\n        ... def get_factorial(\n        ...     for_number: int,\n        ...     current_number: int = 0,\n        ...     acc: int = 1,\n        ... ) -> Union[int, Trampoline[int]]:\n        ...     assert for_number >= 0\n        ...     if for_number <= current_number:\n        ...         return acc\n        ...     return Trampoline(\n        ...         get_factorial,\n        ...         for_number,\n        ...         current_number=current_number + 1,\n        ...         acc=acc * (current_number + 1),\n        ...     )\n\n        >>> assert get_factorial(0) == 1\n        >>> assert get_factorial(3) == 6\n        >>> assert get_factorial(4) == 24\n\n    See also:\n        - eli.thegreenplace.net/2017/on-recursion-continuations-and-trampolines\n        - https://en.wikipedia.org/wiki/Tail_call\n\n    '

    @wraps(func)
    def decorator(*args: _FuncParams.args, **kwargs: _FuncParams.kwargs) -> _ReturnType:
        if False:
            while True:
                i = 10
        trampoline_result = func(*args, **kwargs)
        while isinstance(trampoline_result, Trampoline):
            trampoline_result = trampoline_result()
        return trampoline_result
    decorator._orig_func = func
    return decorator