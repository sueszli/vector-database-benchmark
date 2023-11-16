import asyncio
import asyncio.tasks
import functools
import logging
import sys
import warnings
from asyncio import ensure_future, events, exceptions
logger = logging.getLogger(__name__)

def base_lost_result_handler(task_result):
    if False:
        while True:
            i = 10
    if task_result is not None:
        logger.error(f'The result of the task was lost: {task_result.__class__.__name__}: {task_result!r}')

async def _cancel_and_wait(fut, loop):
    """Cancel the *fut* future or task and wait until it completes."""
    waiter = loop.create_future()
    cb = functools.partial(_release_waiter, waiter)
    fut.add_done_callback(cb)
    try:
        fut.cancel()
        await waiter
    finally:
        fut.remove_done_callback(cb)

def _release_waiter(waiter, *args):
    if False:
        for i in range(10):
            print('nop')
    if not waiter.done():
        waiter.set_result(None)

async def wait_for(fut, timeout, *, loop=None, lost_result_handler=None):
    """Wait for the single Future or coroutine to complete, with timeout.

    Coroutine will be wrapped in Task.

    Returns result of the Future or coroutine.  When a timeout occurs,
    it cancels the task and raises TimeoutError.  To avoid the task
    cancellation, wrap it in shield().

    If the wait is cancelled, the task is also cancelled.

    This function is a coroutine.
    """
    if loop is None:
        loop = events.get_running_loop()
    else:
        warnings.warn('The loop argument is deprecated since Python 3.8, and scheduled for removal in Python 3.10.', DeprecationWarning, stacklevel=2)
    if timeout is None:
        return await fut
    fut = ensure_future(fut, loop=loop)
    if timeout <= 0:
        if fut.done():
            return fut.result()
        await _cancel_and_wait(fut, loop=loop)
        try:
            return fut.result()
        except exceptions.CancelledError as exc:
            raise exceptions.TimeoutError() from exc
    waiter = loop.create_future()
    timeout_handle = loop.call_later(timeout, _release_waiter, waiter)
    cb = functools.partial(_release_waiter, waiter)
    fut.add_done_callback(cb)
    try:
        try:
            await waiter
        except exceptions.CancelledError:
            if fut.done():
                if fut.exception() is None:
                    handler = lost_result_handler or base_lost_result_handler
                    handler(fut.result())
                raise
            fut.remove_done_callback(cb)
            await _cancel_and_wait(fut, loop=loop)
            raise
        if fut.done():
            return fut.result()
        else:
            fut.remove_done_callback(cb)
            await _cancel_and_wait(fut, loop=loop)
            try:
                return fut.result()
            except exceptions.CancelledError as exc:
                raise exceptions.TimeoutError() from exc
    finally:
        timeout_handle.cancel()
wait_for.patched = True

def patch_wait_for():
    if False:
        return 10
    if sys.version_info >= (3, 12):
        return
    if getattr(asyncio.wait_for, 'patched', False):
        return
    asyncio.wait_for = asyncio.tasks.wait_for = wait_for