import asyncio
import sys
from typing import Awaitable, TypeVar
if sys.version_info[0] == 3 and sys.version_info[1] < 7:
    asyncio.Task = asyncio.tasks._PyTask
    asyncio.tasks.Task = asyncio.tasks._PyTask

    def enter_task(loop, task):
        if False:
            print('Hello World!')
        task.__class__._current_tasks[loop] = task

    def leave_task(loop, task):
        if False:
            i = 10
            return i + 15
        task.__class__._current_tasks.pop(loop)
    _enter_task = enter_task
    _leave_task = leave_task
    _all_tasks = asyncio.Task.all_tasks
    _get_current_task = asyncio.Task.current_task
else:
    _enter_task = asyncio.tasks._enter_task
    _leave_task = asyncio.tasks._leave_task
    _all_tasks = asyncio.all_tasks
    _get_current_task = asyncio.current_task
T = TypeVar('T')

def _sync_await(awaitable: Awaitable[T]) -> T:
    if False:
        print('Hello World!')
    '\n    _sync_await waits for the given future to complete by effectively yielding the current task and pumping the event\n    loop.\n    '
    loop = _ensure_event_loop()
    fut = asyncio.ensure_future(awaitable)
    if not loop.is_running():
        return loop.run_until_complete(fut)
    task = _get_current_task(loop)
    if task is not None:
        _leave_task(loop, task)
    ntodo = len(loop._ready)
    while not fut.done() and (not fut.cancelled()):
        loop._run_once()
        if loop._stopping:
            break
    while len(loop._ready) < ntodo:
        handle = asyncio.Handle(lambda : None, [], loop)
        handle._cancelled = True
        loop._ready.append(handle)
    if task is not None:
        _enter_task(loop, task)
    return fut.result()

def _ensure_event_loop():
    if False:
        print('Hello World!')
    'Ensures an asyncio event loop exists for the current thread.'
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop