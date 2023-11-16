from __future__ import annotations
from functools import partial, wraps
from typing import TYPE_CHECKING, TypeVar
from .. import _core
from ..abc import Clock, Instrument
if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing_extensions import ParamSpec
    ArgsT = ParamSpec('ArgsT')
RetT = TypeVar('RetT')

def trio_test(fn: Callable[ArgsT, Awaitable[RetT]]) -> Callable[ArgsT, RetT]:
    if False:
        print('Hello World!')
    'Converts an async test function to be synchronous, running via Trio.\n\n    Usage::\n\n        @trio_test\n        async def test_whatever():\n            await ...\n\n    If a pytest fixture is passed in that subclasses the :class:`~trio.abc.Clock` or\n    :class:`~trio.abc.Instrument` ABCs, then those are passed to :meth:`trio.run()`.\n    '

    @wraps(fn)
    def wrapper(*args: ArgsT.args, **kwargs: ArgsT.kwargs) -> RetT:
        if False:
            for i in range(10):
                print('nop')
        __tracebackhide__ = True
        clocks = [c for c in kwargs.values() if isinstance(c, Clock)]
        if not clocks:
            clock = None
        elif len(clocks) == 1:
            clock = clocks[0]
        else:
            raise ValueError('too many clocks spoil the broth!')
        instruments = [i for i in kwargs.values() if isinstance(i, Instrument)]
        return _core.run(partial(fn, *args, **kwargs), clock=clock, instruments=instruments)
    return wrapper