from __future__ import annotations
import enum
import itertools
import socket
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, TypeVar, cast
import attr
from outcome import Value
from .. import _core
from ._io_common import wake_all
from ._run import _public
from ._windows_cffi import INVALID_HANDLE_VALUE, AFDPollFlags, CData, CompletionModes, ErrorCodes, FileFlags, Handle, IoControlCodes, WSAIoctls, _handle, _Overlapped, ffi, kernel32, ntdll, raise_winerror, ws2_32
if TYPE_CHECKING:
    from typing_extensions import Buffer, TypeAlias
    from .._file_io import _HasFileNo
    from ._traps import Abort, RaiseCancelT
    from ._unbounded_queue import UnboundedQueue
EventResult: TypeAlias = int
T = TypeVar('T')

class CKeys(enum.IntEnum):
    AFD_POLL = 0
    WAIT_OVERLAPPED = 1
    LATE_CANCEL = 2
    FORCE_WAKEUP = 3
    USER_DEFINED = 4
READABLE_FLAGS = AFDPollFlags.AFD_POLL_RECEIVE | AFDPollFlags.AFD_POLL_ACCEPT | AFDPollFlags.AFD_POLL_DISCONNECT | AFDPollFlags.AFD_POLL_ABORT | AFDPollFlags.AFD_POLL_LOCAL_CLOSE
WRITABLE_FLAGS = AFDPollFlags.AFD_POLL_SEND | AFDPollFlags.AFD_POLL_CONNECT_FAIL | AFDPollFlags.AFD_POLL_ABORT | AFDPollFlags.AFD_POLL_LOCAL_CLOSE

@attr.s(slots=True, eq=False)
class AFDWaiters:
    read_task: _core.Task | None = attr.ib(default=None)
    write_task: _core.Task | None = attr.ib(default=None)
    current_op: AFDPollOp | None = attr.ib(default=None)

@attr.s(slots=True, eq=False, frozen=True)
class AFDPollOp:
    lpOverlapped: CData = attr.ib()
    poll_info: Any = attr.ib()
    waiters: AFDWaiters = attr.ib()
    afd_group: AFDGroup = attr.ib()
MAX_AFD_GROUP_SIZE = 500

@attr.s(slots=True, eq=False)
class AFDGroup:
    size: int = attr.ib()
    handle: Handle = attr.ib()
assert not TYPE_CHECKING or sys.platform == 'win32'

@attr.s(slots=True, eq=False, frozen=True)
class _WindowsStatistics:
    tasks_waiting_read: int = attr.ib()
    tasks_waiting_write: int = attr.ib()
    tasks_waiting_overlapped: int = attr.ib()
    completion_key_monitors: int = attr.ib()
    backend: Literal['windows'] = attr.ib(init=False, default='windows')
MAX_EVENTS = 1000

def _check(success: T) -> T:
    if False:
        i = 10
        return i + 15
    if not success:
        raise_winerror()
    return success

def _get_underlying_socket(sock: _HasFileNo | int | Handle, *, which: WSAIoctls=WSAIoctls.SIO_BASE_HANDLE) -> Handle:
    if False:
        i = 10
        return i + 15
    if hasattr(sock, 'fileno'):
        sock = sock.fileno()
    base_ptr = ffi.new('HANDLE *')
    out_size = ffi.new('DWORD *')
    failed = ws2_32.WSAIoctl(ffi.cast('SOCKET', sock), which, ffi.NULL, 0, base_ptr, ffi.sizeof('HANDLE'), out_size, ffi.NULL, ffi.NULL)
    if failed:
        code = ws2_32.WSAGetLastError()
        raise_winerror(code)
    return Handle(base_ptr[0])

def _get_base_socket(sock: _HasFileNo | int | Handle) -> Handle:
    if False:
        for i in range(10):
            print('nop')
    while True:
        try:
            return _get_underlying_socket(sock)
        except OSError as ex:
            if ex.winerror == ErrorCodes.ERROR_NOT_SOCKET:
                raise
            if hasattr(sock, 'fileno'):
                sock = sock.fileno()
            sock = _handle(sock)
            next_sock = _get_underlying_socket(sock, which=WSAIoctls.SIO_BSP_HANDLE_POLL)
            if next_sock == sock:
                raise RuntimeError("Unexpected network configuration detected: SIO_BASE_HANDLE failed and SIO_BSP_HANDLE_POLL didn't return a different socket. Please file a bug at https://github.com/python-trio/trio/issues/new, and include the output of running: netsh winsock show catalog") from ex
            sock = next_sock

def _afd_helper_handle() -> Handle:
    if False:
        i = 10
        return i + 15
    rawname = '\\\\.\\GLOBALROOT\\Device\\Afd\\Trio'.encode('utf-16le') + b'\x00\x00'
    rawname_buf = ffi.from_buffer(rawname)
    handle = kernel32.CreateFileW(ffi.cast('LPCWSTR', rawname_buf), FileFlags.SYNCHRONIZE, FileFlags.FILE_SHARE_READ | FileFlags.FILE_SHARE_WRITE, ffi.NULL, FileFlags.OPEN_EXISTING, FileFlags.FILE_FLAG_OVERLAPPED, ffi.NULL)
    if handle == INVALID_HANDLE_VALUE:
        raise_winerror()
    return handle

@attr.s(frozen=True)
class CompletionKeyEventInfo:
    lpOverlapped: CData = attr.ib()
    dwNumberOfBytesTransferred: int = attr.ib()

class WindowsIOManager:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self._iocp = None
        self._all_afd_handles: list[Handle] = []
        self._iocp = _check(kernel32.CreateIoCompletionPort(INVALID_HANDLE_VALUE, ffi.NULL, 0, 0))
        self._events = ffi.new('OVERLAPPED_ENTRY[]', MAX_EVENTS)
        self._vacant_afd_groups: set[AFDGroup] = set()
        self._afd_ops: dict[CData, AFDPollOp] = {}
        self._afd_waiters: dict[Handle, AFDWaiters] = {}
        self._overlapped_waiters: dict[CData, _core.Task] = {}
        self._posted_too_late_to_cancel: set[CData] = set()
        self._completion_key_queues: dict[int, UnboundedQueue[object]] = {}
        self._completion_key_counter = itertools.count(CKeys.USER_DEFINED)
        with socket.socket() as s:
            select_handle = _get_underlying_socket(s, which=WSAIoctls.SIO_BSP_HANDLE_SELECT)
            try:
                base_handle = _get_underlying_socket(s, which=WSAIoctls.SIO_BASE_HANDLE)
            except OSError:
                _get_base_socket(s)
            else:
                if base_handle != select_handle:
                    raise RuntimeError('Unexpected network configuration detected: SIO_BASE_HANDLE and SIO_BSP_HANDLE_SELECT differ. Please file a bug at https://github.com/python-trio/trio/issues/new, and include the output of running: netsh winsock show catalog')

    def close(self) -> None:
        if False:
            return 10
        try:
            if self._iocp is not None:
                iocp = self._iocp
                self._iocp = None
                _check(kernel32.CloseHandle(iocp))
        finally:
            while self._all_afd_handles:
                afd_handle = self._all_afd_handles.pop()
                _check(kernel32.CloseHandle(afd_handle))

    def __del__(self) -> None:
        if False:
            while True:
                i = 10
        self.close()

    def statistics(self) -> _WindowsStatistics:
        if False:
            for i in range(10):
                print('nop')
        tasks_waiting_read = 0
        tasks_waiting_write = 0
        for waiter in self._afd_waiters.values():
            if waiter.read_task is not None:
                tasks_waiting_read += 1
            if waiter.write_task is not None:
                tasks_waiting_write += 1
        return _WindowsStatistics(tasks_waiting_read=tasks_waiting_read, tasks_waiting_write=tasks_waiting_write, tasks_waiting_overlapped=len(self._overlapped_waiters), completion_key_monitors=len(self._completion_key_queues))

    def force_wakeup(self) -> None:
        if False:
            while True:
                i = 10
        assert self._iocp is not None
        _check(kernel32.PostQueuedCompletionStatus(self._iocp, 0, CKeys.FORCE_WAKEUP, ffi.NULL))

    def get_events(self, timeout: float) -> EventResult:
        if False:
            print('Hello World!')
        received = ffi.new('PULONG')
        milliseconds = round(1000 * timeout)
        if timeout > 0 and milliseconds == 0:
            milliseconds = 1
        try:
            assert self._iocp is not None
            _check(kernel32.GetQueuedCompletionStatusEx(self._iocp, self._events, MAX_EVENTS, received, milliseconds, 0))
        except OSError as exc:
            if exc.winerror != ErrorCodes.WAIT_TIMEOUT:
                raise
            return 0
        result = received[0]
        assert isinstance(result, int)
        return result

    def process_events(self, received: EventResult) -> None:
        if False:
            while True:
                i = 10
        for i in range(received):
            entry = self._events[i]
            if entry.lpCompletionKey == CKeys.AFD_POLL:
                lpo = entry.lpOverlapped
                op = self._afd_ops.pop(lpo)
                waiters = op.waiters
                if waiters.current_op is not op:
                    pass
                else:
                    waiters.current_op = None
                    if lpo.Internal != 0:
                        code = ntdll.RtlNtStatusToDosError(lpo.Internal)
                        raise_winerror(code)
                    flags = op.poll_info.Handles[0].Events
                    if waiters.read_task and flags & READABLE_FLAGS:
                        _core.reschedule(waiters.read_task)
                        waiters.read_task = None
                    if waiters.write_task and flags & WRITABLE_FLAGS:
                        _core.reschedule(waiters.write_task)
                        waiters.write_task = None
                    self._refresh_afd(op.poll_info.Handles[0].Handle)
            elif entry.lpCompletionKey == CKeys.WAIT_OVERLAPPED:
                waiter = self._overlapped_waiters.pop(entry.lpOverlapped)
                overlapped = entry.lpOverlapped
                transferred = entry.dwNumberOfBytesTransferred
                info = CompletionKeyEventInfo(lpOverlapped=overlapped, dwNumberOfBytesTransferred=transferred)
                _core.reschedule(waiter, Value(info))
            elif entry.lpCompletionKey == CKeys.LATE_CANCEL:
                self._posted_too_late_to_cancel.remove(entry.lpOverlapped)
                try:
                    waiter = self._overlapped_waiters.pop(entry.lpOverlapped)
                except KeyError:
                    pass
                else:
                    exc = _core.TrioInternalError(f"Failed to cancel overlapped I/O in {waiter.name} and didn't receive the completion either. Did you forget to call register_with_iocp()?")
                    raise exc
            elif entry.lpCompletionKey == CKeys.FORCE_WAKEUP:
                pass
            else:
                queue = self._completion_key_queues[entry.lpCompletionKey]
                overlapped = int(ffi.cast('uintptr_t', entry.lpOverlapped))
                transferred = entry.dwNumberOfBytesTransferred
                info = CompletionKeyEventInfo(lpOverlapped=overlapped, dwNumberOfBytesTransferred=transferred)
                queue.put_nowait(info)

    def _register_with_iocp(self, handle_: int | CData, completion_key: int) -> None:
        if False:
            while True:
                i = 10
        handle = _handle(handle_)
        assert self._iocp is not None
        _check(kernel32.CreateIoCompletionPort(handle, self._iocp, completion_key, 0))
        _check(kernel32.SetFileCompletionNotificationModes(handle, CompletionModes.FILE_SKIP_SET_EVENT_ON_HANDLE))

    def _refresh_afd(self, base_handle: Handle) -> None:
        if False:
            return 10
        waiters = self._afd_waiters[base_handle]
        if waiters.current_op is not None:
            afd_group = waiters.current_op.afd_group
            try:
                _check(kernel32.CancelIoEx(afd_group.handle, waiters.current_op.lpOverlapped))
            except OSError as exc:
                if exc.winerror != ErrorCodes.ERROR_NOT_FOUND:
                    raise
            waiters.current_op = None
            afd_group.size -= 1
            self._vacant_afd_groups.add(afd_group)
        flags = 0
        if waiters.read_task is not None:
            flags |= READABLE_FLAGS
        if waiters.write_task is not None:
            flags |= WRITABLE_FLAGS
        if not flags:
            del self._afd_waiters[base_handle]
        else:
            try:
                afd_group = self._vacant_afd_groups.pop()
            except KeyError:
                afd_group = AFDGroup(0, _afd_helper_handle())
                self._register_with_iocp(afd_group.handle, CKeys.AFD_POLL)
                self._all_afd_handles.append(afd_group.handle)
            self._vacant_afd_groups.add(afd_group)
            lpOverlapped = ffi.new('LPOVERLAPPED')
            poll_info: Any = ffi.new('AFD_POLL_INFO *')
            poll_info.Timeout = 2 ** 63 - 1
            poll_info.NumberOfHandles = 1
            poll_info.Exclusive = 0
            poll_info.Handles[0].Handle = base_handle
            poll_info.Handles[0].Status = 0
            poll_info.Handles[0].Events = flags
            try:
                _check(kernel32.DeviceIoControl(afd_group.handle, IoControlCodes.IOCTL_AFD_POLL, poll_info, ffi.sizeof('AFD_POLL_INFO'), poll_info, ffi.sizeof('AFD_POLL_INFO'), ffi.NULL, lpOverlapped))
            except OSError as exc:
                if exc.winerror != ErrorCodes.ERROR_IO_PENDING:
                    del self._afd_waiters[base_handle]
                    wake_all(waiters, exc)
                    return
            op = AFDPollOp(lpOverlapped, poll_info, waiters, afd_group)
            waiters.current_op = op
            self._afd_ops[lpOverlapped] = op
            afd_group.size += 1
            if afd_group.size >= MAX_AFD_GROUP_SIZE:
                self._vacant_afd_groups.remove(afd_group)

    async def _afd_poll(self, sock: _HasFileNo | int, mode: str) -> None:
        base_handle = _get_base_socket(sock)
        waiters = self._afd_waiters.get(base_handle)
        if waiters is None:
            waiters = AFDWaiters()
            self._afd_waiters[base_handle] = waiters
        if getattr(waiters, mode) is not None:
            raise _core.BusyResourceError
        setattr(waiters, mode, _core.current_task())
        self._refresh_afd(base_handle)

        def abort_fn(_: RaiseCancelT) -> Abort:
            if False:
                for i in range(10):
                    print('nop')
            setattr(waiters, mode, None)
            self._refresh_afd(base_handle)
            return _core.Abort.SUCCEEDED
        await _core.wait_task_rescheduled(abort_fn)

    @_public
    async def wait_readable(self, sock: _HasFileNo | int) -> None:
        """Block until the kernel reports that the given object is readable.

        On Unix systems, ``sock`` must either be an integer file descriptor,
        or else an object with a ``.fileno()`` method which returns an
        integer file descriptor. Any kind of file descriptor can be passed,
        though the exact semantics will depend on your kernel. For example,
        this probably won't do anything useful for on-disk files.

        On Windows systems, ``sock`` must either be an integer ``SOCKET``
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
        await self._afd_poll(sock, 'read_task')

    @_public
    async def wait_writable(self, sock: _HasFileNo | int) -> None:
        """Block until the kernel reports that the given object is writable.

        See `wait_readable` for the definition of ``sock``.

        :raises trio.BusyResourceError:
            if another task is already waiting for the given socket to
            become writable.
        :raises trio.ClosedResourceError:
            if another task calls :func:`notify_closing` while this
            function is still working.
        """
        await self._afd_poll(sock, 'write_task')

    @_public
    def notify_closing(self, handle: Handle | int | _HasFileNo) -> None:
        if False:
            return 10
        "Notify waiters of the given object that it will be closed.\n\n        Call this before closing a file descriptor (on Unix) or socket (on\n        Windows). This will cause any `wait_readable` or `wait_writable`\n        calls on the given object to immediately wake up and raise\n        `~trio.ClosedResourceError`.\n\n        This doesn't actually close the object – you still have to do that\n        yourself afterwards. Also, you want to be careful to make sure no\n        new tasks start waiting on the object in between when you call this\n        and when it's actually closed. So to close something properly, you\n        usually want to do these steps in order:\n\n        1. Explicitly mark the object as closed, so that any new attempts\n           to use it will abort before they start.\n        2. Call `notify_closing` to wake up any already-existing users.\n        3. Actually close the object.\n\n        It's also possible to do them in a different order if that's more\n        convenient, *but only if* you make sure not to have any checkpoints in\n        between the steps. This way they all happen in a single atomic\n        step, so other tasks won't be able to tell what order they happened\n        in anyway.\n        "
        handle = _get_base_socket(handle)
        waiters = self._afd_waiters.get(handle)
        if waiters is not None:
            wake_all(waiters, _core.ClosedResourceError())
            self._refresh_afd(handle)

    @_public
    def register_with_iocp(self, handle: int | CData) -> None:
        if False:
            i = 10
            return i + 15
        'TODO: these are implemented, but are currently more of a sketch than\n        anything real. See `#26\n        <https://github.com/python-trio/trio/issues/26>`__ and `#52\n        <https://github.com/python-trio/trio/issues/52>`__.\n        '
        self._register_with_iocp(handle, CKeys.WAIT_OVERLAPPED)

    @_public
    async def wait_overlapped(self, handle_: int | CData, lpOverlapped: CData | int) -> object:
        """TODO: these are implemented, but are currently more of a sketch than
        anything real. See `#26
        <https://github.com/python-trio/trio/issues/26>`__ and `#52
        <https://github.com/python-trio/trio/issues/52>`__.
        """
        handle = _handle(handle_)
        if isinstance(lpOverlapped, int):
            lpOverlapped = ffi.cast('LPOVERLAPPED', lpOverlapped)
        if lpOverlapped in self._overlapped_waiters:
            raise _core.BusyResourceError('another task is already waiting on that lpOverlapped')
        task = _core.current_task()
        self._overlapped_waiters[lpOverlapped] = task
        raise_cancel = None

        def abort(raise_cancel_: RaiseCancelT) -> Abort:
            if False:
                for i in range(10):
                    print('nop')
            nonlocal raise_cancel
            raise_cancel = raise_cancel_
            try:
                _check(kernel32.CancelIoEx(handle, lpOverlapped))
            except OSError as exc:
                if exc.winerror == ErrorCodes.ERROR_NOT_FOUND:
                    assert self._iocp is not None
                    _check(kernel32.PostQueuedCompletionStatus(self._iocp, 0, CKeys.LATE_CANCEL, lpOverlapped))
                    self._posted_too_late_to_cancel.add(lpOverlapped)
                else:
                    raise _core.TrioInternalError('CancelIoEx failed with unexpected error') from exc
            return _core.Abort.FAILED
        info = await _core.wait_task_rescheduled(abort)
        lpOverlappedTyped = cast('_Overlapped', lpOverlapped)
        if lpOverlappedTyped.Internal != 0:
            code = ntdll.RtlNtStatusToDosError(lpOverlappedTyped.Internal)
            if code == ErrorCodes.ERROR_OPERATION_ABORTED:
                if raise_cancel is not None:
                    raise_cancel()
                else:
                    raise _core.ClosedResourceError('another task closed this resource')
            else:
                raise_winerror(code)
        return info

    async def _perform_overlapped(self, handle: int | CData, submit_fn: Callable[[_Overlapped], None]) -> _Overlapped:
        await _core.checkpoint_if_cancelled()
        lpOverlapped = cast(_Overlapped, ffi.new('LPOVERLAPPED'))
        try:
            submit_fn(lpOverlapped)
        except OSError as exc:
            if exc.winerror != ErrorCodes.ERROR_IO_PENDING:
                raise
        await self.wait_overlapped(handle, cast(CData, lpOverlapped))
        return lpOverlapped

    @_public
    async def write_overlapped(self, handle: int | CData, data: Buffer, file_offset: int=0) -> int:
        """TODO: these are implemented, but are currently more of a sketch than
        anything real. See `#26
        <https://github.com/python-trio/trio/issues/26>`__ and `#52
        <https://github.com/python-trio/trio/issues/52>`__.
        """
        with ffi.from_buffer(data) as cbuf:

            def submit_write(lpOverlapped: _Overlapped) -> None:
                if False:
                    while True:
                        i = 10
                offset_fields = lpOverlapped.DUMMYUNIONNAME.DUMMYSTRUCTNAME
                offset_fields.Offset = file_offset & 4294967295
                offset_fields.OffsetHigh = file_offset >> 32
                _check(kernel32.WriteFile(_handle(handle), ffi.cast('LPCVOID', cbuf), len(cbuf), ffi.NULL, lpOverlapped))
            lpOverlapped = await self._perform_overlapped(handle, submit_write)
            return lpOverlapped.InternalHigh

    @_public
    async def readinto_overlapped(self, handle: int | CData, buffer: Buffer, file_offset: int=0) -> int:
        """TODO: these are implemented, but are currently more of a sketch than
        anything real. See `#26
        <https://github.com/python-trio/trio/issues/26>`__ and `#52
        <https://github.com/python-trio/trio/issues/52>`__.
        """
        with ffi.from_buffer(buffer, require_writable=True) as cbuf:

            def submit_read(lpOverlapped: _Overlapped) -> None:
                if False:
                    print('Hello World!')
                offset_fields = lpOverlapped.DUMMYUNIONNAME.DUMMYSTRUCTNAME
                offset_fields.Offset = file_offset & 4294967295
                offset_fields.OffsetHigh = file_offset >> 32
                _check(kernel32.ReadFile(_handle(handle), ffi.cast('LPVOID', cbuf), len(cbuf), ffi.NULL, lpOverlapped))
            lpOverlapped = await self._perform_overlapped(handle, submit_read)
            return lpOverlapped.InternalHigh

    @_public
    def current_iocp(self) -> int:
        if False:
            i = 10
            return i + 15
        'TODO: these are implemented, but are currently more of a sketch than\n        anything real. See `#26\n        <https://github.com/python-trio/trio/issues/26>`__ and `#52\n        <https://github.com/python-trio/trio/issues/52>`__.\n        '
        assert self._iocp is not None
        return int(ffi.cast('uintptr_t', self._iocp))

    @contextmanager
    @_public
    def monitor_completion_key(self) -> Iterator[tuple[int, UnboundedQueue[object]]]:
        if False:
            for i in range(10):
                print('nop')
        'TODO: these are implemented, but are currently more of a sketch than\n        anything real. See `#26\n        <https://github.com/python-trio/trio/issues/26>`__ and `#52\n        <https://github.com/python-trio/trio/issues/52>`__.\n        '
        key = next(self._completion_key_counter)
        queue = _core.UnboundedQueue[object]()
        self._completion_key_queues[key] = queue
        try:
            yield (key, queue)
        finally:
            del self._completion_key_queues[key]