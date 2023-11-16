from __future__ import annotations
import errno
import os
import sys
from typing import TYPE_CHECKING
import trio
from ._abc import Stream
from ._util import ConflictDetector, final
if TYPE_CHECKING:
    from typing import Final as FinalType
assert not TYPE_CHECKING or sys.platform != 'win32'
if os.name != 'posix':
    raise ImportError
DEFAULT_RECEIVE_SIZE: FinalType = 65536

class _FdHolder:
    fd: int

    def __init__(self, fd: int) -> None:
        if False:
            return 10
        self.fd = -1
        if not isinstance(fd, int):
            raise TypeError('file descriptor must be an int')
        self.fd = fd
        self._original_is_blocking = os.get_blocking(fd)
        os.set_blocking(fd, False)

    @property
    def closed(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.fd == -1

    def _raw_close(self) -> None:
        if False:
            print('Hello World!')
        if self.closed:
            return
        fd = self.fd
        self.fd = -1
        os.set_blocking(fd, self._original_is_blocking)
        os.close(fd)

    def __del__(self) -> None:
        if False:
            print('Hello World!')
        self._raw_close()

    def close(self) -> None:
        if False:
            while True:
                i = 10
        if not self.closed:
            trio.lowlevel.notify_closing(self.fd)
            self._raw_close()

@final
class FdStream(Stream):
    """
    Represents a stream given the file descriptor to a pipe, TTY, etc.

    *fd* must refer to a file that is open for reading and/or writing and
    supports non-blocking I/O (pipes and TTYs will work, on-disk files probably
    not).  The returned stream takes ownership of the fd, so closing the stream
    will close the fd too.  As with `os.fdopen`, you should not directly use
    an fd after you have wrapped it in a stream using this function.

    To be used as a Trio stream, an open file must be placed in non-blocking
    mode.  Unfortunately, this impacts all I/O that goes through the
    underlying open file, including I/O that uses a different
    file descriptor than the one that was passed to Trio. If other threads
    or processes are using file descriptors that are related through `os.dup`
    or inheritance across `os.fork` to the one that Trio is using, they are
    unlikely to be prepared to have non-blocking I/O semantics suddenly
    thrust upon them.  For example, you can use
    ``FdStream(os.dup(sys.stdin.fileno()))`` to obtain a stream for reading
    from standard input, but it is only safe to do so with heavy caveats: your
    stdin must not be shared by any other processes, and you must not make any
    calls to synchronous methods of `sys.stdin` until the stream returned by
    `FdStream` is closed. See `issue #174
    <https://github.com/python-trio/trio/issues/174>`__ for a discussion of the
    challenges involved in relaxing this restriction.

    Args:
      fd (int): The fd to be wrapped.

    Returns:
      A new `FdStream` object.
    """

    def __init__(self, fd: int) -> None:
        if False:
            return 10
        self._fd_holder = _FdHolder(fd)
        self._send_conflict_detector = ConflictDetector('another task is using this stream for send')
        self._receive_conflict_detector = ConflictDetector('another task is using this stream for receive')

    async def send_all(self, data: bytes) -> None:
        with self._send_conflict_detector:
            if self._fd_holder.closed:
                raise trio.ClosedResourceError('file was already closed')
            await trio.lowlevel.checkpoint()
            length = len(data)
            with memoryview(data) as view:
                sent = 0
                while sent < length:
                    with view[sent:] as remaining:
                        try:
                            sent += os.write(self._fd_holder.fd, remaining)
                        except BlockingIOError:
                            await trio.lowlevel.wait_writable(self._fd_holder.fd)
                        except OSError as e:
                            if e.errno == errno.EBADF:
                                raise trio.ClosedResourceError('file was already closed') from None
                            else:
                                raise trio.BrokenResourceError from e

    async def wait_send_all_might_not_block(self) -> None:
        with self._send_conflict_detector:
            if self._fd_holder.closed:
                raise trio.ClosedResourceError('file was already closed')
            try:
                await trio.lowlevel.wait_writable(self._fd_holder.fd)
            except BrokenPipeError as e:
                raise trio.BrokenResourceError from e

    async def receive_some(self, max_bytes: int | None=None) -> bytes:
        with self._receive_conflict_detector:
            if max_bytes is None:
                max_bytes = DEFAULT_RECEIVE_SIZE
            else:
                if not isinstance(max_bytes, int):
                    raise TypeError('max_bytes must be integer >= 1')
                if max_bytes < 1:
                    raise ValueError('max_bytes must be integer >= 1')
            await trio.lowlevel.checkpoint()
            while True:
                try:
                    data = os.read(self._fd_holder.fd, max_bytes)
                except BlockingIOError:
                    await trio.lowlevel.wait_readable(self._fd_holder.fd)
                except OSError as e:
                    if e.errno == errno.EBADF:
                        raise trio.ClosedResourceError('file was already closed') from None
                    else:
                        raise trio.BrokenResourceError from e
                else:
                    break
            return data

    def close(self) -> None:
        if False:
            return 10
        self._fd_holder.close()

    async def aclose(self) -> None:
        self.close()
        await trio.lowlevel.checkpoint()

    def fileno(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._fd_holder.fd