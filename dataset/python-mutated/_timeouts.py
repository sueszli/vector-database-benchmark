from __future__ import annotations
import math
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING
import trio

def move_on_at(deadline: float) -> trio.CancelScope:
    if False:
        print('Hello World!')
    'Use as a context manager to create a cancel scope with the given\n    absolute deadline.\n\n    Args:\n      deadline (float): The deadline.\n\n    Raises:\n      ValueError: if deadline is NaN.\n\n    '
    if math.isnan(deadline):
        raise ValueError('deadline must not be NaN')
    return trio.CancelScope(deadline=deadline)

def move_on_after(seconds: float) -> trio.CancelScope:
    if False:
        i = 10
        return i + 15
    'Use as a context manager to create a cancel scope whose deadline is\n    set to now + *seconds*.\n\n    Args:\n      seconds (float): The timeout.\n\n    Raises:\n      ValueError: if timeout is less than zero or NaN.\n\n    '
    if seconds < 0:
        raise ValueError('timeout must be non-negative')
    return move_on_at(trio.current_time() + seconds)

async def sleep_forever() -> None:
    """Pause execution of the current task forever (or until cancelled).

    Equivalent to calling ``await sleep(math.inf)``.

    """
    await trio.lowlevel.wait_task_rescheduled(lambda _: trio.lowlevel.Abort.SUCCEEDED)

async def sleep_until(deadline: float) -> None:
    """Pause execution of the current task until the given time.

    The difference between :func:`sleep` and :func:`sleep_until` is that the
    former takes a relative time and the latter takes an absolute time
    according to Trio's internal clock (as returned by :func:`current_time`).

    Args:
        deadline (float): The time at which we should wake up again. May be in
            the past, in which case this function executes a checkpoint but
            does not block.

    Raises:
      ValueError: if deadline is NaN.

    """
    with move_on_at(deadline):
        await sleep_forever()

async def sleep(seconds: float) -> None:
    """Pause execution of the current task for the given number of seconds.

    Args:
        seconds (float): The number of seconds to sleep. May be zero to
            insert a checkpoint without actually blocking.

    Raises:
        ValueError: if *seconds* is negative or NaN.

    """
    if seconds < 0:
        raise ValueError('duration must be non-negative')
    if seconds == 0:
        await trio.lowlevel.checkpoint()
    else:
        await sleep_until(trio.current_time() + seconds)

class TooSlowError(Exception):
    """Raised by :func:`fail_after` and :func:`fail_at` if the timeout
    expires.

    """

def fail_at(deadline: float) -> AbstractContextManager[trio.CancelScope]:
    if False:
        i = 10
        return i + 15
    "Creates a cancel scope with the given deadline, and raises an error if it\n    is actually cancelled.\n\n    This function and :func:`move_on_at` are similar in that both create a\n    cancel scope with a given absolute deadline, and if the deadline expires\n    then both will cause :exc:`Cancelled` to be raised within the scope. The\n    difference is that when the :exc:`Cancelled` exception reaches\n    :func:`move_on_at`, it's caught and discarded. When it reaches\n    :func:`fail_at`, then it's caught and :exc:`TooSlowError` is raised in its\n    place.\n\n    Args:\n      deadline (float): The deadline.\n\n    Raises:\n      TooSlowError: if a :exc:`Cancelled` exception is raised in this scope\n        and caught by the context manager.\n      ValueError: if deadline is NaN.\n\n    "
    with move_on_at(deadline) as scope:
        yield scope
    if scope.cancelled_caught:
        raise TooSlowError
if not TYPE_CHECKING:
    fail_at = contextmanager(fail_at)

def fail_after(seconds: float) -> AbstractContextManager[trio.CancelScope]:
    if False:
        for i in range(10):
            print('nop')
    "Creates a cancel scope with the given timeout, and raises an error if\n    it is actually cancelled.\n\n    This function and :func:`move_on_after` are similar in that both create a\n    cancel scope with a given timeout, and if the timeout expires then both\n    will cause :exc:`Cancelled` to be raised within the scope. The difference\n    is that when the :exc:`Cancelled` exception reaches :func:`move_on_after`,\n    it's caught and discarded. When it reaches :func:`fail_after`, then it's\n    caught and :exc:`TooSlowError` is raised in its place.\n\n    Args:\n      seconds (float): The timeout.\n\n    Raises:\n      TooSlowError: if a :exc:`Cancelled` exception is raised in this scope\n        and caught by the context manager.\n      ValueError: if *seconds* is less than zero or NaN.\n\n    "
    if seconds < 0:
        raise ValueError('timeout must be non-negative')
    return fail_at(trio.current_time() + seconds)