import asyncio
import time
from collections.abc import Coroutine
from mitmproxy.utils import human

def create_task(coro: Coroutine, *, name: str, client: tuple | None=None) -> asyncio.Task:
    if False:
        while True:
            i = 10
    '\n    Like asyncio.create_task, but also store some debug info on the task object.\n    '
    t = asyncio.create_task(coro)
    set_task_debug_info(t, name=name, client=client)
    return t

def set_task_debug_info(task: asyncio.Task, *, name: str, client: tuple | None=None) -> None:
    if False:
        while True:
            i = 10
    'Set debug info for an externally-spawned task.'
    task.created = time.time()
    task.set_name(name)
    if client:
        task.client = client

def set_current_task_debug_info(*, name: str, client: tuple | None=None) -> None:
    if False:
        i = 10
        return i + 15
    'Set debug info for the current task.'
    task = asyncio.current_task()
    assert task
    set_task_debug_info(task, name=name, client=client)

def task_repr(task: asyncio.Task) -> str:
    if False:
        while True:
            i = 10
    'Get a task representation with debug info.'
    name = task.get_name()
    a: float = getattr(task, 'created', 0)
    if a:
        age = f' (age: {time.time() - a:.0f}s)'
    else:
        age = ''
    client = getattr(task, 'client', '')
    if client:
        client = f'{human.format_address(client)}: '
    return f'{client}{name}{age}'