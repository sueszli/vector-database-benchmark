from __future__ import annotations
import contextlib
import contextvars
import functools
import inspect
import queue as stdlib_queue
import threading
from itertools import count
from typing import TYPE_CHECKING, Generic, TypeVar, overload
import attr
import outcome
from sniffio import current_async_library_cvar
import trio
from ._core import RunVar, TrioToken, disable_ki_protection, enable_ki_protection, start_thread_soon
from ._deprecate import warn_deprecated
from ._sync import CapacityLimiter
from ._util import coroutine_or_error
if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from trio._core._traps import RaiseCancelT
RetT = TypeVar('RetT')

class _ParentTaskData(threading.local):
    """Global due to Threading API, thread local storage for data related to the
    parent task of native Trio threads."""
    token: TrioToken
    abandon_on_cancel: bool
    cancel_register: list[RaiseCancelT | None]
    task_register: list[trio.lowlevel.Task | None]
PARENT_TASK_DATA = _ParentTaskData()
_limiter_local: RunVar[CapacityLimiter] = RunVar('limiter')
DEFAULT_LIMIT = 40
_thread_counter = count()

def current_default_thread_limiter() -> CapacityLimiter:
    if False:
        while True:
            i = 10
    'Get the default `~trio.CapacityLimiter` used by\n    `trio.to_thread.run_sync`.\n\n    The most common reason to call this would be if you want to modify its\n    :attr:`~trio.CapacityLimiter.total_tokens` attribute.\n\n    '
    try:
        limiter = _limiter_local.get()
    except LookupError:
        limiter = CapacityLimiter(DEFAULT_LIMIT)
        _limiter_local.set(limiter)
    return limiter

@attr.s(frozen=True, eq=False, hash=False)
class ThreadPlaceholder:
    name: str = attr.ib()

@attr.s(frozen=True, eq=False)
class Run(Generic[RetT]):
    afn: Callable[..., Awaitable[RetT]] = attr.ib()
    args: tuple[object, ...] = attr.ib()
    context: contextvars.Context = attr.ib()
    queue: stdlib_queue.SimpleQueue[outcome.Outcome[RetT]] = attr.ib(init=False, factory=stdlib_queue.SimpleQueue)

    @disable_ki_protection
    async def unprotected_afn(self) -> RetT:
        coro = coroutine_or_error(self.afn, *self.args)
        return await coro

    async def run(self) -> None:
        task = trio.lowlevel.current_task()
        old_context = task.context
        task.context = self.context.copy()
        await trio.lowlevel.cancel_shielded_checkpoint()
        result = await outcome.acapture(self.unprotected_afn)
        task.context = old_context
        await trio.lowlevel.cancel_shielded_checkpoint()
        self.queue.put_nowait(result)

    async def run_system(self) -> None:
        result = await outcome.acapture(self.unprotected_afn)
        self.queue.put_nowait(result)

    def run_in_host_task(self, token: TrioToken) -> None:
        if False:
            while True:
                i = 10
        task_register = PARENT_TASK_DATA.task_register

        def in_trio_thread() -> None:
            if False:
                i = 10
                return i + 15
            task = task_register[0]
            assert task is not None, 'guaranteed by abandon_on_cancel semantics'
            trio.lowlevel.reschedule(task, outcome.Value(self))
        token.run_sync_soon(in_trio_thread)

    def run_in_system_nursery(self, token: TrioToken) -> None:
        if False:
            for i in range(10):
                print('nop')

        def in_trio_thread() -> None:
            if False:
                return 10
            try:
                trio.lowlevel.spawn_system_task(self.run_system, name=self.afn, context=self.context)
            except RuntimeError:
                self.queue.put_nowait(outcome.Error(trio.RunFinishedError('system nursery is closed')))
        token.run_sync_soon(in_trio_thread)

@attr.s(frozen=True, eq=False)
class RunSync(Generic[RetT]):
    fn: Callable[..., RetT] = attr.ib()
    args: tuple[object, ...] = attr.ib()
    context: contextvars.Context = attr.ib()
    queue: stdlib_queue.SimpleQueue[outcome.Outcome[RetT]] = attr.ib(init=False, factory=stdlib_queue.SimpleQueue)

    @disable_ki_protection
    def unprotected_fn(self) -> RetT:
        if False:
            print('Hello World!')
        ret = self.fn(*self.args)
        if inspect.iscoroutine(ret):
            ret.close()
            raise TypeError('Trio expected a synchronous function, but {!r} appears to be asynchronous'.format(getattr(self.fn, '__qualname__', self.fn)))
        return ret

    def run_sync(self) -> None:
        if False:
            i = 10
            return i + 15
        runner: Callable[[Callable[[], RetT]], RetT] = self.context.run
        result = outcome.capture(runner, self.unprotected_fn)
        self.queue.put_nowait(result)

    def run_in_host_task(self, token: TrioToken) -> None:
        if False:
            while True:
                i = 10
        task_register = PARENT_TASK_DATA.task_register

        def in_trio_thread() -> None:
            if False:
                return 10
            task = task_register[0]
            assert task is not None, 'guaranteed by abandon_on_cancel semantics'
            trio.lowlevel.reschedule(task, outcome.Value(self))
        token.run_sync_soon(in_trio_thread)

    def run_in_system_nursery(self, token: TrioToken) -> None:
        if False:
            print('Hello World!')
        token.run_sync_soon(self.run_sync)

@overload
async def to_thread_run_sync(sync_fn: Callable[..., RetT], *args: object, thread_name: str | None=None, abandon_on_cancel: bool=False, limiter: CapacityLimiter | None=None) -> RetT:
    ...

@overload
async def to_thread_run_sync(sync_fn: Callable[..., RetT], *args: object, thread_name: str | None=None, cancellable: bool=False, limiter: CapacityLimiter | None=None) -> RetT:
    ...

@enable_ki_protection
async def to_thread_run_sync(sync_fn: Callable[..., RetT], *args: object, thread_name: str | None=None, abandon_on_cancel: bool | None=None, cancellable: bool | None=None, limiter: CapacityLimiter | None=None) -> RetT:
    """Convert a blocking operation into an async operation using a thread.

    These two lines are equivalent::

        sync_fn(*args)
        await trio.to_thread.run_sync(sync_fn, *args)

    except that if ``sync_fn`` takes a long time, then the first line will
    block the Trio loop while it runs, while the second line allows other Trio
    tasks to continue working while ``sync_fn`` runs. This is accomplished by
    pushing the call to ``sync_fn(*args)`` off into a worker thread.

    From inside the worker thread, you can get back into Trio using the
    functions in `trio.from_thread`.

    Args:
      sync_fn: An arbitrary synchronous callable.
      *args: Positional arguments to pass to sync_fn. If you need keyword
          arguments, use :func:`functools.partial`.
      abandon_on_cancel (bool): Whether to abandon this thread upon
          cancellation of this operation. See discussion below.
      thread_name (str): Optional string to set the name of the thread.
          Will always set `threading.Thread.name`, but only set the os name
          if pthread.h is available (i.e. most POSIX installations).
          pthread names are limited to 15 characters, and can be read from
          ``/proc/<PID>/task/<SPID>/comm`` or with ``ps -eT``, among others.
          Defaults to ``{sync_fn.__name__|None} from {trio.lowlevel.current_task().name}``.
      limiter (None, or CapacityLimiter-like object):
          An object used to limit the number of simultaneous threads. Most
          commonly this will be a `~trio.CapacityLimiter`, but it could be
          anything providing compatible
          :meth:`~trio.CapacityLimiter.acquire_on_behalf_of` and
          :meth:`~trio.CapacityLimiter.release_on_behalf_of` methods. This
          function will call ``acquire_on_behalf_of`` before starting the
          thread, and ``release_on_behalf_of`` after the thread has finished.

          If None (the default), uses the default `~trio.CapacityLimiter`, as
          returned by :func:`current_default_thread_limiter`.

    **Cancellation handling**: Cancellation is a tricky issue here, because
    neither Python nor the operating systems it runs on provide any general
    mechanism for cancelling an arbitrary synchronous function running in a
    thread. This function will always check for cancellation on entry, before
    starting the thread. But once the thread is running, there are two ways it
    can handle being cancelled:

    * If ``abandon_on_cancel=False``, the function ignores the cancellation and
      keeps going, just like if we had called ``sync_fn`` synchronously. This
      is the default behavior.

    * If ``abandon_on_cancel=True``, then this function immediately raises
      `~trio.Cancelled`. In this case **the thread keeps running in
      background** – we just abandon it to do whatever it's going to do, and
      silently discard any return value or errors that it raises. Only use
      this if you know that the operation is safe and side-effect free. (For
      example: :func:`trio.socket.getaddrinfo` uses a thread with
      ``abandon_on_cancel=True``, because it doesn't really affect anything if a
      stray hostname lookup keeps running in the background.)

      The ``limiter`` is only released after the thread has *actually*
      finished – which in the case of cancellation may be some time after this
      function has returned. If :func:`trio.run` finishes before the thread
      does, then the limiter release method will never be called at all.

    .. warning::

       You should not use this function to call long-running CPU-bound
       functions! In addition to the usual GIL-related reasons why using
       threads for CPU-bound work is not very effective in Python, there is an
       additional problem: on CPython, `CPU-bound threads tend to "starve out"
       IO-bound threads <https://bugs.python.org/issue7946>`__, so using
       threads for CPU-bound work is likely to adversely affect the main
       thread running Trio. If you need to do this, you're better off using a
       worker process, or perhaps PyPy (which still has a GIL, but may do a
       better job of fairly allocating CPU time between threads).

    Returns:
      Whatever ``sync_fn(*args)`` returns.

    Raises:
      Exception: Whatever ``sync_fn(*args)`` raises.

    """
    await trio.lowlevel.checkpoint_if_cancelled()
    if cancellable is not None:
        if abandon_on_cancel is not None:
            raise ValueError('Cannot set `cancellable` and `abandon_on_cancel` simultaneously.')
        warn_deprecated('The `cancellable=` keyword argument to `trio.to_thread.run_sync`', '0.23.0', issue=2841, instead='`abandon_on_cancel=`')
        abandon_on_cancel = cancellable
    abandon_on_cancel = bool(abandon_on_cancel)
    if limiter is None:
        limiter = current_default_thread_limiter()
    task_register: list[trio.lowlevel.Task | None] = [trio.lowlevel.current_task()]
    cancel_register: list[RaiseCancelT | None] = [None]
    name = f'trio.to_thread.run_sync-{next(_thread_counter)}'
    placeholder = ThreadPlaceholder(name)

    def report_back_in_trio_thread_fn(result: outcome.Outcome[RetT]) -> None:
        if False:
            return 10

        def do_release_then_return_result() -> RetT:
            if False:
                for i in range(10):
                    print('nop')
            try:
                return result.unwrap()
            finally:
                limiter.release_on_behalf_of(placeholder)
        result = outcome.capture(do_release_then_return_result)
        if task_register[0] is not None:
            trio.lowlevel.reschedule(task_register[0], outcome.Value(result))
    current_trio_token = trio.lowlevel.current_trio_token()
    if thread_name is None:
        thread_name = f"{getattr(sync_fn, '__name__', None)} from {trio.lowlevel.current_task().name}"

    def worker_fn() -> RetT:
        if False:
            for i in range(10):
                print('nop')
        current_async_library_cvar.set(None)
        PARENT_TASK_DATA.token = current_trio_token
        PARENT_TASK_DATA.abandon_on_cancel = abandon_on_cancel
        PARENT_TASK_DATA.cancel_register = cancel_register
        PARENT_TASK_DATA.task_register = task_register
        try:
            ret = sync_fn(*args)
            if inspect.iscoroutine(ret):
                ret.close()
                raise TypeError('Trio expected a sync function, but {!r} appears to be asynchronous'.format(getattr(sync_fn, '__qualname__', sync_fn)))
            return ret
        finally:
            del PARENT_TASK_DATA.token
            del PARENT_TASK_DATA.abandon_on_cancel
            del PARENT_TASK_DATA.cancel_register
            del PARENT_TASK_DATA.task_register
    context = contextvars.copy_context()
    contextvars_aware_worker_fn: Callable[[], RetT] = functools.partial(context.run, worker_fn)

    def deliver_worker_fn_result(result: outcome.Outcome[RetT]) -> None:
        if False:
            i = 10
            return i + 15
        with contextlib.suppress(trio.RunFinishedError):
            current_trio_token.run_sync_soon(report_back_in_trio_thread_fn, result)
    await limiter.acquire_on_behalf_of(placeholder)
    try:
        start_thread_soon(contextvars_aware_worker_fn, deliver_worker_fn_result, thread_name)
    except:
        limiter.release_on_behalf_of(placeholder)
        raise

    def abort(raise_cancel: RaiseCancelT) -> trio.lowlevel.Abort:
        if False:
            return 10
        cancel_register[0] = raise_cancel
        if abandon_on_cancel:
            task_register[0] = None
            return trio.lowlevel.Abort.SUCCEEDED
        else:
            return trio.lowlevel.Abort.FAILED
    while True:
        msg_from_thread: outcome.Outcome[RetT] | Run[object] | RunSync[object] = await trio.lowlevel.wait_task_rescheduled(abort)
        if isinstance(msg_from_thread, outcome.Outcome):
            return msg_from_thread.unwrap()
        elif isinstance(msg_from_thread, Run):
            await msg_from_thread.run()
        elif isinstance(msg_from_thread, RunSync):
            msg_from_thread.run_sync()
        else:
            raise TypeError('trio.to_thread.run_sync received unrecognized thread message {!r}.'.format(msg_from_thread))
        del msg_from_thread

def from_thread_check_cancelled() -> None:
    if False:
        i = 10
        return i + 15
    'Raise `trio.Cancelled` if the associated Trio task entered a cancelled status.\n\n     Only applicable to threads spawned by `trio.to_thread.run_sync`. Poll to allow\n     ``abandon_on_cancel=False`` threads to raise :exc:`~trio.Cancelled` at a suitable\n     place, or to end abandoned ``abandon_on_cancel=True`` threads sooner than they may\n     otherwise.\n\n    Raises:\n        Cancelled: If the corresponding call to `trio.to_thread.run_sync` has had a\n            delivery of cancellation attempted against it, regardless of the value of\n            ``abandon_on_cancel`` supplied as an argument to it.\n        RuntimeError: If this thread is not spawned from `trio.to_thread.run_sync`.\n\n    .. note::\n\n       To be precise, :func:`~trio.from_thread.check_cancelled` checks whether the task\n       running :func:`trio.to_thread.run_sync` has ever been cancelled since the last\n       time it was running a :func:`trio.from_thread.run` or :func:`trio.from_thread.run_sync`\n       function. It may raise `trio.Cancelled` even if a cancellation occurred that was\n       later hidden by a modification to `trio.CancelScope.shield` between the cancelled\n       `~trio.CancelScope` and :func:`trio.to_thread.run_sync`. This differs from the\n       behavior of normal Trio checkpoints, which raise `~trio.Cancelled` only if the\n       cancellation is still active when the checkpoint executes. The distinction here is\n       *exceedingly* unlikely to be relevant to your application, but we mention it\n       for completeness.\n    '
    try:
        raise_cancel = PARENT_TASK_DATA.cancel_register[0]
    except AttributeError as exc:
        raise RuntimeError("this thread wasn't created by Trio, can't check for cancellation") from exc
    if raise_cancel is not None:
        raise_cancel()

def _check_token(trio_token: TrioToken | None) -> TrioToken:
    if False:
        while True:
            i = 10
    "Raise a RuntimeError if this function is called within a trio run.\n\n    Avoids deadlock by making sure we're not called from inside a context\n    that we might be waiting for and blocking it.\n    "
    if trio_token is not None and (not isinstance(trio_token, TrioToken)):
        raise RuntimeError('Passed kwarg trio_token is not of type TrioToken')
    if trio_token is None:
        try:
            trio_token = PARENT_TASK_DATA.token
        except AttributeError:
            raise RuntimeError("this thread wasn't created by Trio, pass kwarg trio_token=...") from None
    try:
        trio.lowlevel.current_task()
    except RuntimeError:
        pass
    else:
        raise RuntimeError('this is a blocking function; call it from a thread')
    return trio_token

def from_thread_run(afn: Callable[..., Awaitable[RetT]], *args: object, trio_token: TrioToken | None=None) -> RetT:
    if False:
        print('Hello World!')
    'Run the given async function in the parent Trio thread, blocking until it\n    is complete.\n\n    Returns:\n      Whatever ``afn(*args)`` returns.\n\n    Returns or raises whatever the given function returns or raises. It\n    can also raise exceptions of its own:\n\n    Raises:\n        RunFinishedError: if the corresponding call to :func:`trio.run` has\n            already completed, or if the run has started its final cleanup phase\n            and can no longer spawn new system tasks.\n        Cancelled: If the original call to :func:`trio.to_thread.run_sync` is cancelled\n            (if *trio_token* is None) or the call to :func:`trio.run` completes\n            (if *trio_token* is not None) while ``afn(*args)`` is running,\n            then *afn* is likely to raise :exc:`trio.Cancelled`.\n        RuntimeError: if you try calling this from inside the Trio thread,\n            which would otherwise cause a deadlock, or if no ``trio_token`` was\n            provided, and we can\'t infer one from context.\n        TypeError: if ``afn`` is not an asynchronous function.\n\n    **Locating a TrioToken**: There are two ways to specify which\n    `trio.run` loop to reenter:\n\n        - Spawn this thread from `trio.to_thread.run_sync`. Trio will\n          automatically capture the relevant Trio token and use it\n          to re-enter the same Trio task.\n        - Pass a keyword argument, ``trio_token`` specifying a specific\n          `trio.run` loop to re-enter. This is useful in case you have a\n          "foreign" thread, spawned using some other framework, and still want\n          to enter Trio, or if you want to use a new system task to call ``afn``,\n          maybe to avoid the cancellation context of a corresponding\n          `trio.to_thread.run_sync` task.\n    '
    token_provided = trio_token is not None
    trio_token = _check_token(trio_token)
    message_to_trio = Run(afn, args, contextvars.copy_context())
    if token_provided or PARENT_TASK_DATA.abandon_on_cancel:
        message_to_trio.run_in_system_nursery(trio_token)
    else:
        message_to_trio.run_in_host_task(trio_token)
    return message_to_trio.queue.get().unwrap()

def from_thread_run_sync(fn: Callable[..., RetT], *args: object, trio_token: TrioToken | None=None) -> RetT:
    if False:
        print('Hello World!')
    'Run the given sync function in the parent Trio thread, blocking until it\n    is complete.\n\n    Returns:\n      Whatever ``fn(*args)`` returns.\n\n    Returns or raises whatever the given function returns or raises. It\n    can also raise exceptions of its own:\n\n    Raises:\n        RunFinishedError: if the corresponding call to `trio.run` has\n            already completed.\n        RuntimeError: if you try calling this from inside the Trio thread,\n            which would otherwise cause a deadlock or if no ``trio_token`` was\n            provided, and we can\'t infer one from context.\n        TypeError: if ``fn`` is an async function.\n\n    **Locating a TrioToken**: There are two ways to specify which\n    `trio.run` loop to reenter:\n\n        - Spawn this thread from `trio.to_thread.run_sync`. Trio will\n          automatically capture the relevant Trio token and use it when you\n          want to re-enter Trio.\n        - Pass a keyword argument, ``trio_token`` specifying a specific\n          `trio.run` loop to re-enter. This is useful in case you have a\n          "foreign" thread, spawned using some other framework, and still want\n          to enter Trio, or if you want to use a new system task to call ``fn``,\n          maybe to avoid the cancellation context of a corresponding\n          `trio.to_thread.run_sync` task.\n    '
    token_provided = trio_token is not None
    trio_token = _check_token(trio_token)
    message_to_trio = RunSync(fn, args, contextvars.copy_context())
    if token_provided or PARENT_TASK_DATA.abandon_on_cancel:
        message_to_trio.run_in_system_nursery(trio_token)
    else:
        message_to_trio.run_in_host_task(trio_token)
    return message_to_trio.queue.get().unwrap()