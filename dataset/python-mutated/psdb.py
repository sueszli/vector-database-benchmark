from __future__ import annotations
import builtins
import types
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from typing import TypeVar
    from typing_extensions import ParamSpec
    T = TypeVar('T')
    P = ParamSpec('P')
NO_BREAKGRAPH_CODES: set[types.CodeType] = set()
NO_FALLBACK_CODES: set[types.CodeType] = set()

def assert_true(input: bool):
    if False:
        print('Hello World!')
    assert input

def print(*args, **kwargs):
    if False:
        print('Hello World!')
    builtins.print('[Dygraph]', *args, **kwargs)

def breakpoint():
    if False:
        for i in range(10):
            print('nop')
    import paddle
    old = paddle.framework.core.set_eval_frame(None)
    builtins.breakpoint()
    paddle.framework.core.set_eval_frame(old)

def check_no_breakgraph(fn: Callable[P, T]) -> Callable[P, T]:
    if False:
        print('Hello World!')
    NO_BREAKGRAPH_CODES.add(fn.__code__)
    return fn

def breakgraph():
    if False:
        while True:
            i = 10
    pass

def check_no_fallback(fn: Callable[P, T]) -> Callable[P, T]:
    if False:
        print('Hello World!')
    NO_FALLBACK_CODES.add(fn.__code__)
    return fn

def fallback():
    if False:
        i = 10
        return i + 15
    pass

def in_sot():
    if False:
        return 10
    return False