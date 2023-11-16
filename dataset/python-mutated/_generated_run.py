from __future__ import annotations
from typing import TYPE_CHECKING, Any
from ._ki import LOCALS_KEY_KI_PROTECTION_ENABLED
from ._run import _NO_SEND, GLOBAL_RUN_CONTEXT, RunStatistics, Task
if TYPE_CHECKING:
    import contextvars
    from collections.abc import Awaitable, Callable
    from outcome import Outcome
    from .._abc import Clock
    from ._entry_queue import TrioToken

def current_statistics() -> RunStatistics:
    if False:
        i = 10
        return i + 15
    "Returns ``RunStatistics``, which contains run-loop-level debugging information.\n\n    Currently, the following fields are defined:\n\n    * ``tasks_living`` (int): The number of tasks that have been spawned\n      and not yet exited.\n    * ``tasks_runnable`` (int): The number of tasks that are currently\n      queued on the run queue (as opposed to blocked waiting for something\n      to happen).\n    * ``seconds_to_next_deadline`` (float): The time until the next\n      pending cancel scope deadline. May be negative if the deadline has\n      expired but we haven't yet processed cancellations. May be\n      :data:`~math.inf` if there are no pending deadlines.\n    * ``run_sync_soon_queue_size`` (int): The number of\n      unprocessed callbacks queued via\n      :meth:`trio.lowlevel.TrioToken.run_sync_soon`.\n    * ``io_statistics`` (object): Some statistics from Trio's I/O\n      backend. This always has an attribute ``backend`` which is a string\n      naming which operating-system-specific I/O backend is in use; the\n      other attributes vary between backends.\n\n    "
    locals()[LOCALS_KEY_KI_PROTECTION_ENABLED] = True
    try:
        return GLOBAL_RUN_CONTEXT.runner.current_statistics()
    except AttributeError:
        raise RuntimeError('must be called from async context') from None

def current_time() -> float:
    if False:
        while True:
            i = 10
    "Returns the current time according to Trio's internal clock.\n\n    Returns:\n        float: The current time.\n\n    Raises:\n        RuntimeError: if not inside a call to :func:`trio.run`.\n\n    "
    locals()[LOCALS_KEY_KI_PROTECTION_ENABLED] = True
    try:
        return GLOBAL_RUN_CONTEXT.runner.current_time()
    except AttributeError:
        raise RuntimeError('must be called from async context') from None

def current_clock() -> Clock:
    if False:
        print('Hello World!')
    'Returns the current :class:`~trio.abc.Clock`.'
    locals()[LOCALS_KEY_KI_PROTECTION_ENABLED] = True
    try:
        return GLOBAL_RUN_CONTEXT.runner.current_clock()
    except AttributeError:
        raise RuntimeError('must be called from async context') from None

def current_root_task() -> Task | None:
    if False:
        for i in range(10):
            print('nop')
    'Returns the current root :class:`Task`.\n\n    This is the task that is the ultimate parent of all other tasks.\n\n    '
    locals()[LOCALS_KEY_KI_PROTECTION_ENABLED] = True
    try:
        return GLOBAL_RUN_CONTEXT.runner.current_root_task()
    except AttributeError:
        raise RuntimeError('must be called from async context') from None

def reschedule(task: Task, next_send: Outcome[Any]=_NO_SEND) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Reschedule the given task with the given\n    :class:`outcome.Outcome`.\n\n    See :func:`wait_task_rescheduled` for the gory details.\n\n    There must be exactly one call to :func:`reschedule` for every call to\n    :func:`wait_task_rescheduled`. (And when counting, keep in mind that\n    returning :data:`Abort.SUCCEEDED` from an abort callback is equivalent\n    to calling :func:`reschedule` once.)\n\n    Args:\n      task (trio.lowlevel.Task): the task to be rescheduled. Must be blocked\n          in a call to :func:`wait_task_rescheduled`.\n      next_send (outcome.Outcome): the value (or error) to return (or\n          raise) from :func:`wait_task_rescheduled`.\n\n    '
    locals()[LOCALS_KEY_KI_PROTECTION_ENABLED] = True
    try:
        return GLOBAL_RUN_CONTEXT.runner.reschedule(task, next_send)
    except AttributeError:
        raise RuntimeError('must be called from async context') from None

def spawn_system_task(async_fn: Callable[..., Awaitable[object]], *args: object, name: object=None, context: contextvars.Context | None=None) -> Task:
    if False:
        return 10
    'Spawn a "system" task.\n\n    System tasks have a few differences from regular tasks:\n\n    * They don\'t need an explicit nursery; instead they go into the\n      internal "system nursery".\n\n    * If a system task raises an exception, then it\'s converted into a\n      :exc:`~trio.TrioInternalError` and *all* tasks are cancelled. If you\n      write a system task, you should be careful to make sure it doesn\'t\n      crash.\n\n    * System tasks are automatically cancelled when the main task exits.\n\n    * By default, system tasks have :exc:`KeyboardInterrupt` protection\n      *enabled*. If you want your task to be interruptible by control-C,\n      then you need to use :func:`disable_ki_protection` explicitly (and\n      come up with some plan for what to do with a\n      :exc:`KeyboardInterrupt`, given that system tasks aren\'t allowed to\n      raise exceptions).\n\n    * System tasks do not inherit context variables from their creator.\n\n    Towards the end of a call to :meth:`trio.run`, after the main\n    task and all system tasks have exited, the system nursery\n    becomes closed. At this point, new calls to\n    :func:`spawn_system_task` will raise ``RuntimeError("Nursery\n    is closed to new arrivals")`` instead of creating a system\n    task. It\'s possible to encounter this state either in\n    a ``finally`` block in an async generator, or in a callback\n    passed to :meth:`TrioToken.run_sync_soon` at the right moment.\n\n    Args:\n      async_fn: An async callable.\n      args: Positional arguments for ``async_fn``. If you want to pass\n          keyword arguments, use :func:`functools.partial`.\n      name: The name for this task. Only used for debugging/introspection\n          (e.g. ``repr(task_obj)``). If this isn\'t a string,\n          :func:`spawn_system_task` will try to make it one. A common use\n          case is if you\'re wrapping a function before spawning a new\n          task, you might pass the original function as the ``name=`` to\n          make debugging easier.\n      context: An optional ``contextvars.Context`` object with context variables\n          to use for this task. You would normally get a copy of the current\n          context with ``context = contextvars.copy_context()`` and then you would\n          pass that ``context`` object here.\n\n    Returns:\n      Task: the newly spawned task\n\n    '
    locals()[LOCALS_KEY_KI_PROTECTION_ENABLED] = True
    try:
        return GLOBAL_RUN_CONTEXT.runner.spawn_system_task(async_fn, *args, name=name, context=context)
    except AttributeError:
        raise RuntimeError('must be called from async context') from None

def current_trio_token() -> TrioToken:
    if False:
        for i in range(10):
            print('nop')
    'Retrieve the :class:`TrioToken` for the current call to\n    :func:`trio.run`.\n\n    '
    locals()[LOCALS_KEY_KI_PROTECTION_ENABLED] = True
    try:
        return GLOBAL_RUN_CONTEXT.runner.current_trio_token()
    except AttributeError:
        raise RuntimeError('must be called from async context') from None

async def wait_all_tasks_blocked(cushion: float=0.0) -> None:
    """Block until there are no runnable tasks.

    This is useful in testing code when you want to give other tasks a
    chance to "settle down". The calling task is blocked, and doesn't wake
    up until all other tasks are also blocked for at least ``cushion``
    seconds. (Setting a non-zero ``cushion`` is intended to handle cases
    like two tasks talking to each other over a local socket, where we
    want to ignore the potential brief moment between a send and receive
    when all tasks are blocked.)

    Note that ``cushion`` is measured in *real* time, not the Trio clock
    time.

    If there are multiple tasks blocked in :func:`wait_all_tasks_blocked`,
    then the one with the shortest ``cushion`` is the one woken (and
    this task becoming unblocked resets the timers for the remaining
    tasks). If there are multiple tasks that have exactly the same
    ``cushion``, then all are woken.

    You should also consider :class:`trio.testing.Sequencer`, which
    provides a more explicit way to control execution ordering within a
    test, and will often produce more readable tests.

    Example:
      Here's an example of one way to test that Trio's locks are fair: we
      take the lock in the parent, start a child, wait for the child to be
      blocked waiting for the lock (!), and then check that we can't
      release and immediately re-acquire the lock::

         async def lock_taker(lock):
             await lock.acquire()
             lock.release()

         async def test_lock_fairness():
             lock = trio.Lock()
             await lock.acquire()
             async with trio.open_nursery() as nursery:
                 nursery.start_soon(lock_taker, lock)
                 # child hasn't run yet, we have the lock
                 assert lock.locked()
                 assert lock._owner is trio.lowlevel.current_task()
                 await trio.testing.wait_all_tasks_blocked()
                 # now the child has run and is blocked on lock.acquire(), we
                 # still have the lock
                 assert lock.locked()
                 assert lock._owner is trio.lowlevel.current_task()
                 lock.release()
                 try:
                     # The child has a prior claim, so we can't have it
                     lock.acquire_nowait()
                 except trio.WouldBlock:
                     assert lock._owner is not trio.lowlevel.current_task()
                     print("PASS")
                 else:
                     print("FAIL")

    """
    locals()[LOCALS_KEY_KI_PROTECTION_ENABLED] = True
    try:
        return await GLOBAL_RUN_CONTEXT.runner.wait_all_tasks_blocked(cushion)
    except AttributeError:
        raise RuntimeError('must be called from async context') from None