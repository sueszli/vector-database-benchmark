"""
Utilities for interoperability with async functions and workers from various contexts.
"""
import asyncio
import ctypes
import inspect
import threading
import warnings
from contextlib import asynccontextmanager
from functools import partial, wraps
from threading import Thread
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional, Type, TypeVar, Union
from uuid import UUID, uuid4
import anyio
import anyio.abc
import sniffio
from typing_extensions import Literal, ParamSpec, TypeGuard
T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')
Async = Literal[True]
Sync = Literal[False]
A = TypeVar('A', Async, Sync, covariant=True)
EVENT_LOOP_GC_REFS = {}
PREFECT_THREAD_LIMITER: Optional[anyio.CapacityLimiter] = None

def get_thread_limiter():
    if False:
        while True:
            i = 10
    global PREFECT_THREAD_LIMITER
    if PREFECT_THREAD_LIMITER is None:
        PREFECT_THREAD_LIMITER = anyio.CapacityLimiter(250)
    return PREFECT_THREAD_LIMITER

def is_async_fn(func: Union[Callable[P, R], Callable[P, Awaitable[R]]]) -> TypeGuard[Callable[P, Awaitable[R]]]:
    if False:
        return 10
    '\n    Returns `True` if a function returns a coroutine.\n\n    See https://github.com/microsoft/pyright/issues/2142 for an example use\n    '
    while hasattr(func, '__wrapped__'):
        func = func.__wrapped__
    return inspect.iscoroutinefunction(func)

def is_async_gen_fn(func):
    if False:
        i = 10
        return i + 15
    '\n    Returns `True` if a function is an async generator.\n    '
    while hasattr(func, '__wrapped__'):
        func = func.__wrapped__
    return inspect.isasyncgenfunction(func)

async def run_sync_in_worker_thread(__fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Runs a sync function in a new worker thread so that the main thread's event loop
    is not blocked

    Unlike the anyio function, this defaults to a cancellable thread and does not allow
    passing arguments to the anyio function so users can pass kwargs to their function.

    Note that cancellation of threads will not result in interrupted computation, the
    thread may continue running — the outcome will just be ignored.
    """
    call = partial(__fn, *args, **kwargs)
    return await anyio.to_thread.run_sync(call, cancellable=True, limiter=get_thread_limiter())

def raise_async_exception_in_thread(thread: Thread, exc_type: Type[BaseException]):
    if False:
        i = 10
        return i + 15
    '\n    Raise an exception in a thread asynchronously.\n\n    This will not interrupt long-running system calls like `sleep` or `wait`.\n    '
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), ctypes.py_object(exc_type))
    if ret == 0:
        raise ValueError('Thread not found.')

async def run_sync_in_interruptible_worker_thread(__fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Runs a sync function in a new interruptible worker thread so that the main
    thread's event loop is not blocked

    Unlike the anyio function, this performs best-effort cancellation of the
    thread using the C API. Cancellation will not interrupt system calls like
    `sleep`.
    """

    class NotSet:
        pass
    thread: Thread = None
    result = NotSet
    event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def capture_worker_thread_and_result():
        if False:
            for i in range(10):
                print('nop')
        nonlocal thread, result
        try:
            thread = threading.current_thread()
            result = __fn(*args, **kwargs)
        except BaseException as exc:
            result = exc
            raise
        finally:
            loop.call_soon_threadsafe(event.set)

    async def send_interrupt_to_thread():
        try:
            await event.wait()
        except anyio.get_cancelled_exc_class():
            raise_async_exception_in_thread(thread, anyio.get_cancelled_exc_class())
            raise
    async with anyio.create_task_group() as tg:
        tg.start_soon(send_interrupt_to_thread)
        tg.start_soon(partial(anyio.to_thread.run_sync, capture_worker_thread_and_result, cancellable=True, limiter=get_thread_limiter()))
    assert result is not NotSet
    return result

def run_async_from_worker_thread(__fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
    if False:
        for i in range(10):
            print('nop')
    "\n    Runs an async function in the main thread's event loop, blocking the worker\n    thread until completion\n    "
    call = partial(__fn, *args, **kwargs)
    return anyio.from_thread.run(call)

def run_async_in_new_loop(__fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any):
    if False:
        while True:
            i = 10
    return anyio.run(partial(__fn, *args, **kwargs))

def in_async_worker_thread() -> bool:
    if False:
        print('Hello World!')
    try:
        anyio.from_thread.threadlocals.current_async_module
    except AttributeError:
        return False
    else:
        return True

def in_async_main_thread() -> bool:
    if False:
        i = 10
        return i + 15
    try:
        sniffio.current_async_library()
    except sniffio.AsyncLibraryNotFoundError:
        return False
    else:
        return not in_async_worker_thread()

def sync_compatible(async_fn: T) -> T:
    if False:
        print('Hello World!')
    '\n    Converts an async function into a dual async and sync function.\n\n    When the returned function is called, we will attempt to determine the best way\n    to enter the async function.\n\n    - If in a thread with a running event loop, we will return the coroutine for the\n        caller to await. This is normal async behavior.\n    - If in a blocking worker thread with access to an event loop in another thread, we\n        will submit the async method to the event loop.\n    - If we cannot find an event loop, we will create a new one and run the async method\n        then tear down the loop.\n    '

    @wraps(async_fn)
    def coroutine_wrapper(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        from prefect._internal.concurrency.api import create_call, from_sync
        from prefect._internal.concurrency.calls import get_current_call, logger
        from prefect._internal.concurrency.event_loop import get_running_loop
        from prefect._internal.concurrency.threads import get_global_loop
        global_thread_portal = get_global_loop()
        current_thread = threading.current_thread()
        current_call = get_current_call()
        current_loop = get_running_loop()
        if current_thread.ident == global_thread_portal.thread.ident:
            logger.debug(f'{async_fn} --> return coroutine for internal await')
            return async_fn(*args, **kwargs)
        elif in_async_main_thread() and (not current_call or is_async_fn(current_call.fn)):
            logger.debug(f'{async_fn} --> return coroutine for user await')
            return async_fn(*args, **kwargs)
        elif in_async_worker_thread():
            return run_async_from_worker_thread(async_fn, *args, **kwargs)
        elif current_loop is not None:
            logger.debug(f'{async_fn} --> run async in global loop portal')
            return from_sync.call_soon_in_loop_thread(create_call(async_fn, *args, **kwargs)).result()
        else:
            logger.debug(f'{async_fn} --> run async in new loop')
            call = create_call(async_fn, *args, **kwargs)
            return call()
    if is_async_fn(async_fn):
        wrapper = coroutine_wrapper
    elif is_async_gen_fn(async_fn):
        raise ValueError('Async generators cannot yet be marked as `sync_compatible`')
    else:
        raise TypeError('The decorated function must be async.')
    wrapper.aio = async_fn
    return wrapper

@asynccontextmanager
async def asyncnullcontext():
    yield

def sync(__async_fn: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
    if False:
        for i in range(10):
            print('nop')
    '\n    Call an async function from a synchronous context. Block until completion.\n\n    If in an asynchronous context, we will run the code in a separate loop instead of\n    failing but a warning will be displayed since this is not recommended.\n    '
    if in_async_main_thread():
        warnings.warn('`sync` called from an asynchronous context; you should `await` the async function directly instead.')
        with anyio.start_blocking_portal() as portal:
            return portal.call(partial(__async_fn, *args, **kwargs))
    elif in_async_worker_thread():
        return run_async_from_worker_thread(__async_fn, *args, **kwargs)
    else:
        return run_async_in_new_loop(__async_fn, *args, **kwargs)

async def add_event_loop_shutdown_callback(coroutine_fn: Callable[[], Awaitable]):
    """
    Adds a callback to the given callable on event loop closure. The callable must be
    a coroutine function. It will be awaited when the current event loop is shutting
    down.

    Requires use of `asyncio.run()` which waits for async generator shutdown by
    default or explicit call of `asyncio.shutdown_asyncgens()`. If the application
    is entered with `asyncio.run_until_complete()` and the user calls
    `asyncio.close()` without the generator shutdown call, this will not trigger
    callbacks.

    asyncio does not provided _any_ other way to clean up a resource when the event
    loop is about to close.
    """

    async def on_shutdown(key):
        _ = EVENT_LOOP_GC_REFS
        try:
            yield
        except GeneratorExit:
            await coroutine_fn()
            EVENT_LOOP_GC_REFS.pop(key)
    key = id(on_shutdown)
    EVENT_LOOP_GC_REFS[key] = on_shutdown(key)
    await EVENT_LOOP_GC_REFS[key].__anext__()

class GatherIncomplete(RuntimeError):
    """Used to indicate retrieving gather results before completion"""

class GatherTaskGroup(anyio.abc.TaskGroup):
    """
    A task group that gathers results.

    AnyIO does not include support `gather`. This class extends the `TaskGroup`
    interface to allow simple gathering.

    See https://github.com/agronholm/anyio/issues/100

    This class should be instantiated with `create_gather_task_group`.
    """

    def __init__(self, task_group: anyio.abc.TaskGroup):
        if False:
            for i in range(10):
                print('nop')
        self._results: Dict[UUID, Any] = {}
        self._task_group: anyio.abc.TaskGroup = task_group

    async def _run_and_store(self, key, fn, args):
        self._results[key] = await fn(*args)

    def start_soon(self, fn, *args) -> UUID:
        if False:
            print('Hello World!')
        key = uuid4()
        self._results[key] = GatherIncomplete
        self._task_group.start_soon(self._run_and_store, key, fn, args)
        return key

    async def start(self, fn, *args):
        """
        Since `start` returns the result of `task_status.started()` but here we must
        return the key instead, we just won't support this method for now.
        """
        raise RuntimeError('`GatherTaskGroup` does not support `start`.')

    def get_result(self, key: UUID) -> Any:
        if False:
            i = 10
            return i + 15
        result = self._results[key]
        if result is GatherIncomplete:
            raise GatherIncomplete('Task is not complete. Results should not be retrieved until the task group exits.')
        return result

    async def __aenter__(self):
        await self._task_group.__aenter__()
        return self

    async def __aexit__(self, *tb):
        try:
            retval = await self._task_group.__aexit__(*tb)
            return retval
        finally:
            del self._task_group

def create_gather_task_group() -> GatherTaskGroup:
    if False:
        while True:
            i = 10
    'Create a new task group that gathers results'
    return GatherTaskGroup(anyio.create_task_group())

async def gather(*calls: Callable[[], Coroutine[Any, Any, T]]) -> List[T]:
    """
    Run calls concurrently and gather their results.

    Unlike `asyncio.gather` this expects to receive _callables_ not _coroutines_.
    This matches `anyio` semantics.
    """
    keys = []
    async with create_gather_task_group() as tg:
        for call in calls:
            keys.append(tg.start_soon(call))
    return [tg.get_result(key) for key in keys]

class LazySemaphore:

    def __init__(self, initial_value_func):
        if False:
            print('Hello World!')
        self._semaphore = None
        self._initial_value_func = initial_value_func

    async def __aenter__(self):
        self._initialize_semaphore()
        await self._semaphore.__aenter__()
        return self._semaphore

    async def __aexit__(self, exc_type, exc, tb):
        await self._semaphore.__aexit__(exc_type, exc, tb)

    def _initialize_semaphore(self):
        if False:
            return 10
        if self._semaphore is None:
            initial_value = self._initial_value_func()
            self._semaphore = asyncio.Semaphore(initial_value)