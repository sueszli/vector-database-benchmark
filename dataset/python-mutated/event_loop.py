"""
Thread-safe utilities for working with asynchronous event loops.
"""
import asyncio
import concurrent.futures
import functools
from typing import Awaitable, Callable, Coroutine, Optional, TypeVar
from typing_extensions import ParamSpec
P = ParamSpec('P')
T = TypeVar('T')

def get_running_loop() -> Optional[asyncio.BaseEventLoop]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the current running loop.\n\n    Returns `None` if there is no running loop.\n    '
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None

def call_in_loop(__loop: asyncio.AbstractEventLoop, __fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    if False:
        print('Hello World!')
    "\n    Run a synchronous call in event loop's thread from another thread.\n\n    This function is blocking and not safe to call from an asynchronous context.\n\n    Returns the result of the call.\n    "
    if __loop is get_running_loop():
        return __fn(*args, **kwargs)
    else:
        future = call_soon_in_loop(__loop, __fn, *args, **kwargs)
        return future.result()

def call_soon_in_loop(__loop: asyncio.AbstractEventLoop, __fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> concurrent.futures.Future:
    if False:
        i = 10
        return i + 15
    "\n    Run a synchronous call in an event loop's thread from another thread.\n\n    This function is non-blocking and safe to call from an asynchronous context.\n\n    Returns a future that can be used to retrieve the result of the call.\n    "
    future = concurrent.futures.Future()

    @functools.wraps(__fn)
    def wrapper() -> None:
        if False:
            return 10
        try:
            result = __fn(*args, **kwargs)
        except BaseException as exc:
            future.set_exception(exc)
            if not isinstance(exc, Exception):
                raise
        else:
            future.set_result(result)
    if __loop is get_running_loop():
        __loop.call_soon(wrapper)
    else:
        __loop.call_soon_threadsafe(wrapper)
    return future

async def run_coroutine_in_loop_from_async(__loop: asyncio.AbstractEventLoop, __coro: Coroutine) -> Awaitable:
    """
    Run an asynchronous call in an event loop from an asynchronous context.

    Returns an awaitable that returns the result of the coroutine.
    """
    if __loop is get_running_loop():
        return await __coro
    else:
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(__coro, __loop))