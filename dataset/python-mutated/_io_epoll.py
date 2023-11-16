from __future__ import annotations
import contextlib
import select
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Literal
import attr
from .. import _core
from ._io_common import wake_all
from ._run import Task, _public
from ._wakeup_socketpair import WakeupSocketpair
if TYPE_CHECKING:
    from typing_extensions import TypeAlias
    from .._core import Abort, RaiseCancelT
    from .._file_io import _HasFileNo

@attr.s(slots=True, eq=False)
class EpollWaiters:
    read_task: Task | None = attr.ib(default=None)
    write_task: Task | None = attr.ib(default=None)
    current_flags: int = attr.ib(default=0)
assert not TYPE_CHECKING or sys.platform == 'linux'
EventResult: TypeAlias = 'list[tuple[int, int]]'

@attr.s(slots=True, eq=False, frozen=True)
class _EpollStatistics:
    tasks_waiting_read: int = attr.ib()
    tasks_waiting_write: int = attr.ib()
    backend: Literal['epoll'] = attr.ib(init=False, default='epoll')

@attr.s(slots=True, eq=False, hash=False)
class EpollIOManager:
    _epoll: select.epoll = attr.ib(factory=select.epoll)
    _registered: defaultdict[int, EpollWaiters] = attr.ib(factory=lambda : defaultdict(EpollWaiters))
    _force_wakeup: WakeupSocketpair = attr.ib(factory=WakeupSocketpair)
    _force_wakeup_fd: int | None = attr.ib(default=None)

    def __attrs_post_init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self._epoll.register(self._force_wakeup.wakeup_sock, select.EPOLLIN)
        self._force_wakeup_fd = self._force_wakeup.wakeup_sock.fileno()

    def statistics(self) -> _EpollStatistics:
        if False:
            print('Hello World!')
        tasks_waiting_read = 0
        tasks_waiting_write = 0
        for waiter in self._registered.values():
            if waiter.read_task is not None:
                tasks_waiting_read += 1
            if waiter.write_task is not None:
                tasks_waiting_write += 1
        return _EpollStatistics(tasks_waiting_read=tasks_waiting_read, tasks_waiting_write=tasks_waiting_write)

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        self._epoll.close()
        self._force_wakeup.close()

    def force_wakeup(self) -> None:
        if False:
            i = 10
            return i + 15
        self._force_wakeup.wakeup_thread_and_signal_safe()

    def get_events(self, timeout: float) -> EventResult:
        if False:
            while True:
                i = 10
        max_events = max(1, len(self._registered))
        return self._epoll.poll(timeout, max_events)

    def process_events(self, events: EventResult) -> None:
        if False:
            for i in range(10):
                print('nop')
        for (fd, flags) in events:
            if fd == self._force_wakeup_fd:
                self._force_wakeup.drain()
                continue
            waiters = self._registered[fd]
            waiters.current_flags = 0
            if flags & ~select.EPOLLIN and waiters.write_task is not None:
                _core.reschedule(waiters.write_task)
                waiters.write_task = None
            if flags & ~select.EPOLLOUT and waiters.read_task is not None:
                _core.reschedule(waiters.read_task)
                waiters.read_task = None
            self._update_registrations(fd)

    def _update_registrations(self, fd: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        waiters = self._registered[fd]
        wanted_flags = 0
        if waiters.read_task is not None:
            wanted_flags |= select.EPOLLIN
        if waiters.write_task is not None:
            wanted_flags |= select.EPOLLOUT
        if wanted_flags != waiters.current_flags:
            try:
                try:
                    self._epoll.modify(fd, wanted_flags | select.EPOLLONESHOT)
                except OSError:
                    self._epoll.register(fd, wanted_flags | select.EPOLLONESHOT)
                waiters.current_flags = wanted_flags
            except OSError as exc:
                del self._registered[fd]
                wake_all(waiters, exc)
                return
        if not wanted_flags:
            del self._registered[fd]

    async def _epoll_wait(self, fd: int | _HasFileNo, attr_name: str) -> None:
        if not isinstance(fd, int):
            fd = fd.fileno()
        waiters = self._registered[fd]
        if getattr(waiters, attr_name) is not None:
            raise _core.BusyResourceError('another task is already reading / writing this fd')
        setattr(waiters, attr_name, _core.current_task())
        self._update_registrations(fd)

        def abort(_: RaiseCancelT) -> Abort:
            if False:
                print('Hello World!')
            setattr(waiters, attr_name, None)
            self._update_registrations(fd)
            return _core.Abort.SUCCEEDED
        await _core.wait_task_rescheduled(abort)

    @_public
    async def wait_readable(self, fd: int | _HasFileNo) -> None:
        """Block until the kernel reports that the given object is readable.

        On Unix systems, ``fd`` must either be an integer file descriptor,
        or else an object with a ``.fileno()`` method which returns an
        integer file descriptor. Any kind of file descriptor can be passed,
        though the exact semantics will depend on your kernel. For example,
        this probably won't do anything useful for on-disk files.

        On Windows systems, ``fd`` must either be an integer ``SOCKET``
        handle, or else an object with a ``.fileno()`` method which returns
        an integer ``SOCKET`` handle. File descriptors aren't supported,
        and neither are handles that refer to anything besides a
        ``SOCKET``.

        :raises trio.BusyResourceError:
            if another task is already waiting for the given socket to
            become readable.
        :raises trio.ClosedResourceError:
            if another task calls :func:`notify_closing` while this
            function is still working.
        """
        await self._epoll_wait(fd, 'read_task')

    @_public
    async def wait_writable(self, fd: int | _HasFileNo) -> None:
        """Block until the kernel reports that the given object is writable.

        See `wait_readable` for the definition of ``fd``.

        :raises trio.BusyResourceError:
            if another task is already waiting for the given socket to
            become writable.
        :raises trio.ClosedResourceError:
            if another task calls :func:`notify_closing` while this
            function is still working.
        """
        await self._epoll_wait(fd, 'write_task')

    @_public
    def notify_closing(self, fd: int | _HasFileNo) -> None:
        if False:
            print('Hello World!')
        "Notify waiters of the given object that it will be closed.\n\n        Call this before closing a file descriptor (on Unix) or socket (on\n        Windows). This will cause any `wait_readable` or `wait_writable`\n        calls on the given object to immediately wake up and raise\n        `~trio.ClosedResourceError`.\n\n        This doesn't actually close the object – you still have to do that\n        yourself afterwards. Also, you want to be careful to make sure no\n        new tasks start waiting on the object in between when you call this\n        and when it's actually closed. So to close something properly, you\n        usually want to do these steps in order:\n\n        1. Explicitly mark the object as closed, so that any new attempts\n           to use it will abort before they start.\n        2. Call `notify_closing` to wake up any already-existing users.\n        3. Actually close the object.\n\n        It's also possible to do them in a different order if that's more\n        convenient, *but only if* you make sure not to have any checkpoints in\n        between the steps. This way they all happen in a single atomic\n        step, so other tasks won't be able to tell what order they happened\n        in anyway.\n        "
        if not isinstance(fd, int):
            fd = fd.fileno()
        wake_all(self._registered[fd], _core.ClosedResourceError('another task closed this fd'))
        del self._registered[fd]
        with contextlib.suppress(OSError, ValueError):
            self._epoll.unregister(fd)