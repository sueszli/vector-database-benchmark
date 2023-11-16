from __future__ import annotations
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING
from .. import _core
if TYPE_CHECKING:
    from collections.abc import Generator

@contextmanager
def _assert_yields_or_not(expected: bool) -> Generator[None, None, None]:
    if False:
        print('Hello World!')
    'Check if checkpoints are executed in a block of code.'
    __tracebackhide__ = True
    task = _core.current_task()
    orig_cancel = task._cancel_points
    orig_schedule = task._schedule_points
    try:
        yield
        if expected and (task._cancel_points == orig_cancel or task._schedule_points == orig_schedule):
            raise AssertionError('assert_checkpoints block did not yield!')
    finally:
        if not expected and (task._cancel_points != orig_cancel or task._schedule_points != orig_schedule):
            raise AssertionError('assert_no_checkpoints block yielded!')

def assert_checkpoints() -> AbstractContextManager[None]:
    if False:
        while True:
            i = 10
    "Use as a context manager to check that the code inside the ``with``\n    block either exits with an exception or executes at least one\n    :ref:`checkpoint <checkpoints>`.\n\n    Raises:\n      AssertionError: if no checkpoint was executed.\n\n    Example:\n      Check that :func:`trio.sleep` is a checkpoint, even if it doesn't\n      block::\n\n         with trio.testing.assert_checkpoints():\n             await trio.sleep(0)\n\n    "
    __tracebackhide__ = True
    return _assert_yields_or_not(True)

def assert_no_checkpoints() -> AbstractContextManager[None]:
    if False:
        i = 10
        return i + 15
    'Use as a context manager to check that the code inside the ``with``\n    block does not execute any :ref:`checkpoints <checkpoints>`.\n\n    Raises:\n      AssertionError: if a checkpoint was executed.\n\n    Example:\n      Synchronous code never contains any checkpoints, but we can double-check\n      that::\n\n         send_channel, receive_channel = trio.open_memory_channel(10)\n         with trio.testing.assert_no_checkpoints():\n             send_channel.send_nowait(None)\n\n    '
    __tracebackhide__ = True
    return _assert_yields_or_not(False)