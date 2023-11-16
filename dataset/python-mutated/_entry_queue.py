from __future__ import annotations
import threading
from collections import deque
from typing import Callable, Iterable, NoReturn, Tuple
import attr
from .. import _core
from .._util import NoPublicConstructor, final
from ._wakeup_socketpair import WakeupSocketpair
Function = Callable[..., object]
Job = Tuple[Function, Iterable[object]]

@attr.s(slots=True)
class EntryQueue:
    queue: deque[Job] = attr.ib(factory=deque)
    idempotent_queue: dict[Job, None] = attr.ib(factory=dict)
    wakeup: WakeupSocketpair = attr.ib(factory=WakeupSocketpair)
    done: bool = attr.ib(default=False)
    lock: threading.RLock = attr.ib(factory=threading.RLock)

    async def task(self) -> None:
        assert _core.currently_ki_protected()
        assert self.lock.__class__.__module__ == '_thread'

        def run_cb(job: Job) -> None:
            if False:
                while True:
                    i = 10
            (sync_fn, args) = job
            try:
                sync_fn(*args)
            except BaseException as exc:

                async def kill_everything(exc: BaseException) -> NoReturn:
                    raise exc
                try:
                    _core.spawn_system_task(kill_everything, exc)
                except RuntimeError:
                    parent_nursery = _core.current_task().parent_nursery
                    if parent_nursery is None:
                        raise AssertionError('Internal error: `parent_nursery` should never be `None`') from exc
                    parent_nursery.start_soon(kill_everything, exc)

        def run_all_bounded() -> None:
            if False:
                return 10
            for _ in range(len(self.queue)):
                run_cb(self.queue.popleft())
            for job in list(self.idempotent_queue):
                del self.idempotent_queue[job]
                run_cb(job)
        try:
            while True:
                run_all_bounded()
                if not self.queue and (not self.idempotent_queue):
                    await self.wakeup.wait_woken()
                else:
                    await _core.checkpoint()
        except _core.Cancelled:
            with self.lock:
                self.done = True
            run_all_bounded()
            assert not self.queue
            assert not self.idempotent_queue

    def close(self) -> None:
        if False:
            while True:
                i = 10
        self.wakeup.close()

    def size(self) -> int:
        if False:
            while True:
                i = 10
        return len(self.queue) + len(self.idempotent_queue)

    def run_sync_soon(self, sync_fn: Function, *args: object, idempotent: bool=False) -> None:
        if False:
            print('Hello World!')
        with self.lock:
            if self.done:
                raise _core.RunFinishedError('run() has exited')
            if idempotent:
                self.idempotent_queue[sync_fn, args] = None
            else:
                self.queue.append((sync_fn, args))
            self.wakeup.wakeup_thread_and_signal_safe()

@final
@attr.s(eq=False, hash=False, slots=True)
class TrioToken(metaclass=NoPublicConstructor):
    """An opaque object representing a single call to :func:`trio.run`.

    It has no public constructor; instead, see :func:`current_trio_token`.

    This object has two uses:

    1. It lets you re-enter the Trio run loop from external threads or signal
       handlers. This is the low-level primitive that :func:`trio.to_thread`
       and `trio.from_thread` use to communicate with worker threads, that
       `trio.open_signal_receiver` uses to receive notifications about
       signals, and so forth.

    2. Each call to :func:`trio.run` has exactly one associated
       :class:`TrioToken` object, so you can use it to identify a particular
       call.

    """
    _reentry_queue: EntryQueue = attr.ib()

    def run_sync_soon(self, sync_fn: Function, *args: object, idempotent: bool=False) -> None:
        if False:
            print('Hello World!')
        'Schedule a call to ``sync_fn(*args)`` to occur in the context of a\n        Trio task.\n\n        This is safe to call from the main thread, from other threads, and\n        from signal handlers. This is the fundamental primitive used to\n        re-enter the Trio run loop from outside of it.\n\n        The call will happen "soon", but there\'s no guarantee about exactly\n        when, and no mechanism provided for finding out when it\'s happened.\n        If you need this, you\'ll have to build your own.\n\n        The call is effectively run as part of a system task (see\n        :func:`~trio.lowlevel.spawn_system_task`). In particular this means\n        that:\n\n        * :exc:`KeyboardInterrupt` protection is *enabled* by default; if\n          you want ``sync_fn`` to be interruptible by control-C, then you\n          need to use :func:`~trio.lowlevel.disable_ki_protection`\n          explicitly.\n\n        * If ``sync_fn`` raises an exception, then it\'s converted into a\n          :exc:`~trio.TrioInternalError` and *all* tasks are cancelled. You\n          should be careful that ``sync_fn`` doesn\'t crash.\n\n        All calls with ``idempotent=False`` are processed in strict\n        first-in first-out order.\n\n        If ``idempotent=True``, then ``sync_fn`` and ``args`` must be\n        hashable, and Trio will make a best-effort attempt to discard any\n        call submission which is equal to an already-pending call. Trio\n        will process these in first-in first-out order.\n\n        Any ordering guarantees apply separately to ``idempotent=False``\n        and ``idempotent=True`` calls; there\'s no rule for how calls in the\n        different categories are ordered with respect to each other.\n\n        :raises trio.RunFinishedError:\n              if the associated call to :func:`trio.run`\n              has already exited. (Any call that *doesn\'t* raise this error\n              is guaranteed to be fully processed before :func:`trio.run`\n              exits.)\n\n        '
        self._reentry_queue.run_sync_soon(sync_fn, *args, idempotent=idempotent)