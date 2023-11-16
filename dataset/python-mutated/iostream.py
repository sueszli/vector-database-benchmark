"""Utility classes to write to and read from non-blocking files and sockets.

Contents:

* `BaseIOStream`: Generic interface for reading and writing.
* `IOStream`: Implementation of BaseIOStream using non-blocking sockets.
* `SSLIOStream`: SSL-aware version of IOStream.
* `PipeIOStream`: Pipe-based IOStream implementation.
"""
import asyncio
import collections
import errno
import io
import numbers
import os
import socket
import ssl
import sys
import re
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import ioloop
from tornado.log import gen_log
from tornado.netutil import ssl_wrap_socket, _client_ssl_defaults, _server_ssl_defaults
from tornado.util import errno_from_exception
import typing
from typing import Union, Optional, Awaitable, Callable, Pattern, Any, Dict, TypeVar, Tuple
from types import TracebackType
if typing.TYPE_CHECKING:
    from typing import Deque, List, Type
_IOStreamType = TypeVar('_IOStreamType', bound='IOStream')
_ERRNO_CONNRESET = (errno.ECONNRESET, errno.ECONNABORTED, errno.EPIPE, errno.ETIMEDOUT)
if hasattr(errno, 'WSAECONNRESET'):
    _ERRNO_CONNRESET += (errno.WSAECONNRESET, errno.WSAECONNABORTED, errno.WSAETIMEDOUT)
if sys.platform == 'darwin':
    _ERRNO_CONNRESET += (errno.EPROTOTYPE,)
_WINDOWS = sys.platform.startswith('win')

class StreamClosedError(IOError):
    """Exception raised by `IOStream` methods when the stream is closed.

    Note that the close callback is scheduled to run *after* other
    callbacks on the stream (to allow for buffered data to be processed),
    so you may see this error before you see the close callback.

    The ``real_error`` attribute contains the underlying error that caused
    the stream to close (if any).

    .. versionchanged:: 4.3
       Added the ``real_error`` attribute.
    """

    def __init__(self, real_error: Optional[BaseException]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__('Stream is closed')
        self.real_error = real_error

class UnsatisfiableReadError(Exception):
    """Exception raised when a read cannot be satisfied.

    Raised by ``read_until`` and ``read_until_regex`` with a ``max_bytes``
    argument.
    """
    pass

class StreamBufferFullError(Exception):
    """Exception raised by `IOStream` methods when the buffer is full."""

class _StreamBuffer(object):
    """
    A specialized buffer that tries to avoid copies when large pieces
    of data are encountered.
    """

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._buffers = collections.deque()
        self._first_pos = 0
        self._size = 0

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return self._size
    _large_buf_threshold = 2048

    def append(self, data: Union[bytes, bytearray, memoryview]) -> None:
        if False:
            print('Hello World!')
        '\n        Append the given piece of data (should be a buffer-compatible object).\n        '
        size = len(data)
        if size > self._large_buf_threshold:
            if not isinstance(data, memoryview):
                data = memoryview(data)
            self._buffers.append((True, data))
        elif size > 0:
            if self._buffers:
                (is_memview, b) = self._buffers[-1]
                new_buf = is_memview or len(b) >= self._large_buf_threshold
            else:
                new_buf = True
            if new_buf:
                self._buffers.append((False, bytearray(data)))
            else:
                b += data
        self._size += size

    def peek(self, size: int) -> memoryview:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a view over at most ``size`` bytes (possibly fewer) at the\n        current buffer position.\n        '
        assert size > 0
        try:
            (is_memview, b) = self._buffers[0]
        except IndexError:
            return memoryview(b'')
        pos = self._first_pos
        if is_memview:
            return typing.cast(memoryview, b[pos:pos + size])
        else:
            return memoryview(b)[pos:pos + size]

    def advance(self, size: int) -> None:
        if False:
            print('Hello World!')
        '\n        Advance the current buffer position by ``size`` bytes.\n        '
        assert 0 < size <= self._size
        self._size -= size
        pos = self._first_pos
        buffers = self._buffers
        while buffers and size > 0:
            (is_large, b) = buffers[0]
            b_remain = len(b) - size - pos
            if b_remain <= 0:
                buffers.popleft()
                size -= len(b) - pos
                pos = 0
            elif is_large:
                pos += size
                size = 0
            else:
                pos += size
                del typing.cast(bytearray, b)[:pos]
                pos = 0
                size = 0
        assert size == 0
        self._first_pos = pos

class BaseIOStream(object):
    """A utility class to write to and read from a non-blocking file or socket.

    We support a non-blocking ``write()`` and a family of ``read_*()``
    methods. When the operation completes, the ``Awaitable`` will resolve
    with the data read (or ``None`` for ``write()``). All outstanding
    ``Awaitables`` will resolve with a `StreamClosedError` when the
    stream is closed; `.BaseIOStream.set_close_callback` can also be used
    to be notified of a closed stream.

    When a stream is closed due to an error, the IOStream's ``error``
    attribute contains the exception object.

    Subclasses must implement `fileno`, `close_fd`, `write_to_fd`,
    `read_from_fd`, and optionally `get_fd_error`.

    """

    def __init__(self, max_buffer_size: Optional[int]=None, read_chunk_size: Optional[int]=None, max_write_buffer_size: Optional[int]=None) -> None:
        if False:
            return 10
        '`BaseIOStream` constructor.\n\n        :arg max_buffer_size: Maximum amount of incoming data to buffer;\n            defaults to 100MB.\n        :arg read_chunk_size: Amount of data to read at one time from the\n            underlying transport; defaults to 64KB.\n        :arg max_write_buffer_size: Amount of outgoing data to buffer;\n            defaults to unlimited.\n\n        .. versionchanged:: 4.0\n           Add the ``max_write_buffer_size`` parameter.  Changed default\n           ``read_chunk_size`` to 64KB.\n        .. versionchanged:: 5.0\n           The ``io_loop`` argument (deprecated since version 4.1) has been\n           removed.\n        '
        self.io_loop = ioloop.IOLoop.current()
        self.max_buffer_size = max_buffer_size or 104857600
        self.read_chunk_size = min(read_chunk_size or 65536, self.max_buffer_size // 2)
        self.max_write_buffer_size = max_write_buffer_size
        self.error = None
        self._read_buffer = bytearray()
        self._read_buffer_size = 0
        self._user_read_buffer = False
        self._after_user_read_buffer = None
        self._write_buffer = _StreamBuffer()
        self._total_write_index = 0
        self._total_write_done_index = 0
        self._read_delimiter = None
        self._read_regex = None
        self._read_max_bytes = None
        self._read_bytes = None
        self._read_partial = False
        self._read_until_close = False
        self._read_future = None
        self._write_futures = collections.deque()
        self._close_callback = None
        self._connect_future = None
        self._ssl_connect_future = None
        self._connecting = False
        self._state = None
        self._closed = False

    def fileno(self) -> Union[int, ioloop._Selectable]:
        if False:
            print('Hello World!')
        'Returns the file descriptor for this stream.'
        raise NotImplementedError()

    def close_fd(self) -> None:
        if False:
            while True:
                i = 10
        'Closes the file underlying this stream.\n\n        ``close_fd`` is called by `BaseIOStream` and should not be called\n        elsewhere; other users should call `close` instead.\n        '
        raise NotImplementedError()

    def write_to_fd(self, data: memoryview) -> int:
        if False:
            print('Hello World!')
        'Attempts to write ``data`` to the underlying file.\n\n        Returns the number of bytes written.\n        '
        raise NotImplementedError()

    def read_from_fd(self, buf: Union[bytearray, memoryview]) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        'Attempts to read from the underlying file.\n\n        Reads up to ``len(buf)`` bytes, storing them in the buffer.\n        Returns the number of bytes read. Returns None if there was\n        nothing to read (the socket returned `~errno.EWOULDBLOCK` or\n        equivalent), and zero on EOF.\n\n        .. versionchanged:: 5.0\n\n           Interface redesigned to take a buffer and return a number\n           of bytes instead of a freshly-allocated object.\n        '
        raise NotImplementedError()

    def get_fd_error(self) -> Optional[Exception]:
        if False:
            for i in range(10):
                print('nop')
        'Returns information about any error on the underlying file.\n\n        This method is called after the `.IOLoop` has signaled an error on the\n        file descriptor, and should return an Exception (such as `socket.error`\n        with additional information, or None if no such information is\n        available.\n        '
        return None

    def read_until_regex(self, regex: bytes, max_bytes: Optional[int]=None) -> Awaitable[bytes]:
        if False:
            i = 10
            return i + 15
        'Asynchronously read until we have matched the given regex.\n\n        The result includes the data that matches the regex and anything\n        that came before it.\n\n        If ``max_bytes`` is not None, the connection will be closed\n        if more than ``max_bytes`` bytes have been read and the regex is\n        not satisfied.\n\n        .. versionchanged:: 4.0\n            Added the ``max_bytes`` argument.  The ``callback`` argument is\n            now optional and a `.Future` will be returned if it is omitted.\n\n        .. versionchanged:: 6.0\n\n           The ``callback`` argument was removed. Use the returned\n           `.Future` instead.\n\n        '
        future = self._start_read()
        self._read_regex = re.compile(regex)
        self._read_max_bytes = max_bytes
        try:
            self._try_inline_read()
        except UnsatisfiableReadError as e:
            gen_log.info('Unsatisfiable read, closing connection: %s' % e)
            self.close(exc_info=e)
            return future
        except:
            future.add_done_callback(lambda f: f.exception())
            raise
        return future

    def read_until(self, delimiter: bytes, max_bytes: Optional[int]=None) -> Awaitable[bytes]:
        if False:
            while True:
                i = 10
        'Asynchronously read until we have found the given delimiter.\n\n        The result includes all the data read including the delimiter.\n\n        If ``max_bytes`` is not None, the connection will be closed\n        if more than ``max_bytes`` bytes have been read and the delimiter\n        is not found.\n\n        .. versionchanged:: 4.0\n            Added the ``max_bytes`` argument.  The ``callback`` argument is\n            now optional and a `.Future` will be returned if it is omitted.\n\n        .. versionchanged:: 6.0\n\n           The ``callback`` argument was removed. Use the returned\n           `.Future` instead.\n        '
        future = self._start_read()
        self._read_delimiter = delimiter
        self._read_max_bytes = max_bytes
        try:
            self._try_inline_read()
        except UnsatisfiableReadError as e:
            gen_log.info('Unsatisfiable read, closing connection: %s' % e)
            self.close(exc_info=e)
            return future
        except:
            future.add_done_callback(lambda f: f.exception())
            raise
        return future

    def read_bytes(self, num_bytes: int, partial: bool=False) -> Awaitable[bytes]:
        if False:
            for i in range(10):
                print('nop')
        'Asynchronously read a number of bytes.\n\n        If ``partial`` is true, data is returned as soon as we have\n        any bytes to return (but never more than ``num_bytes``)\n\n        .. versionchanged:: 4.0\n            Added the ``partial`` argument.  The callback argument is now\n            optional and a `.Future` will be returned if it is omitted.\n\n        .. versionchanged:: 6.0\n\n           The ``callback`` and ``streaming_callback`` arguments have\n           been removed. Use the returned `.Future` (and\n           ``partial=True`` for ``streaming_callback``) instead.\n\n        '
        future = self._start_read()
        assert isinstance(num_bytes, numbers.Integral)
        self._read_bytes = num_bytes
        self._read_partial = partial
        try:
            self._try_inline_read()
        except:
            future.add_done_callback(lambda f: f.exception())
            raise
        return future

    def read_into(self, buf: bytearray, partial: bool=False) -> Awaitable[int]:
        if False:
            print('Hello World!')
        'Asynchronously read a number of bytes.\n\n        ``buf`` must be a writable buffer into which data will be read.\n\n        If ``partial`` is true, the callback is run as soon as any bytes\n        have been read.  Otherwise, it is run when the ``buf`` has been\n        entirely filled with read data.\n\n        .. versionadded:: 5.0\n\n        .. versionchanged:: 6.0\n\n           The ``callback`` argument was removed. Use the returned\n           `.Future` instead.\n\n        '
        future = self._start_read()
        available_bytes = self._read_buffer_size
        n = len(buf)
        if available_bytes >= n:
            buf[:] = memoryview(self._read_buffer)[:n]
            del self._read_buffer[:n]
            self._after_user_read_buffer = self._read_buffer
        elif available_bytes > 0:
            buf[:available_bytes] = memoryview(self._read_buffer)[:]
        self._user_read_buffer = True
        self._read_buffer = buf
        self._read_buffer_size = available_bytes
        self._read_bytes = n
        self._read_partial = partial
        try:
            self._try_inline_read()
        except:
            future.add_done_callback(lambda f: f.exception())
            raise
        return future

    def read_until_close(self) -> Awaitable[bytes]:
        if False:
            print('Hello World!')
        'Asynchronously reads all data from the socket until it is closed.\n\n        This will buffer all available data until ``max_buffer_size``\n        is reached. If flow control or cancellation are desired, use a\n        loop with `read_bytes(partial=True) <.read_bytes>` instead.\n\n        .. versionchanged:: 4.0\n            The callback argument is now optional and a `.Future` will\n            be returned if it is omitted.\n\n        .. versionchanged:: 6.0\n\n           The ``callback`` and ``streaming_callback`` arguments have\n           been removed. Use the returned `.Future` (and `read_bytes`\n           with ``partial=True`` for ``streaming_callback``) instead.\n\n        '
        future = self._start_read()
        if self.closed():
            self._finish_read(self._read_buffer_size)
            return future
        self._read_until_close = True
        try:
            self._try_inline_read()
        except:
            future.add_done_callback(lambda f: f.exception())
            raise
        return future

    def write(self, data: Union[bytes, memoryview]) -> 'Future[None]':
        if False:
            return 10
        'Asynchronously write the given data to this stream.\n\n        This method returns a `.Future` that resolves (with a result\n        of ``None``) when the write has been completed.\n\n        The ``data`` argument may be of type `bytes` or `memoryview`.\n\n        .. versionchanged:: 4.0\n            Now returns a `.Future` if no callback is given.\n\n        .. versionchanged:: 4.5\n            Added support for `memoryview` arguments.\n\n        .. versionchanged:: 6.0\n\n           The ``callback`` argument was removed. Use the returned\n           `.Future` instead.\n\n        '
        self._check_closed()
        if data:
            if isinstance(data, memoryview):
                data = memoryview(data).cast('B')
            if self.max_write_buffer_size is not None and len(self._write_buffer) + len(data) > self.max_write_buffer_size:
                raise StreamBufferFullError('Reached maximum write buffer size')
            self._write_buffer.append(data)
            self._total_write_index += len(data)
        future = Future()
        future.add_done_callback(lambda f: f.exception())
        self._write_futures.append((self._total_write_index, future))
        if not self._connecting:
            self._handle_write()
            if self._write_buffer:
                self._add_io_state(self.io_loop.WRITE)
            self._maybe_add_error_listener()
        return future

    def set_close_callback(self, callback: Optional[Callable[[], None]]) -> None:
        if False:
            return 10
        'Call the given callback when the stream is closed.\n\n        This mostly is not necessary for applications that use the\n        `.Future` interface; all outstanding ``Futures`` will resolve\n        with a `StreamClosedError` when the stream is closed. However,\n        it is still useful as a way to signal that the stream has been\n        closed while no other read or write is in progress.\n\n        Unlike other callback-based interfaces, ``set_close_callback``\n        was not removed in Tornado 6.0.\n        '
        self._close_callback = callback
        self._maybe_add_error_listener()

    def close(self, exc_info: Union[None, bool, BaseException, Tuple['Optional[Type[BaseException]]', Optional[BaseException], Optional[TracebackType]]]=False) -> None:
        if False:
            i = 10
            return i + 15
        'Close this stream.\n\n        If ``exc_info`` is true, set the ``error`` attribute to the current\n        exception from `sys.exc_info` (or if ``exc_info`` is a tuple,\n        use that instead of `sys.exc_info`).\n        '
        if not self.closed():
            if exc_info:
                if isinstance(exc_info, tuple):
                    self.error = exc_info[1]
                elif isinstance(exc_info, BaseException):
                    self.error = exc_info
                else:
                    exc_info = sys.exc_info()
                    if any(exc_info):
                        self.error = exc_info[1]
            if self._read_until_close:
                self._read_until_close = False
                self._finish_read(self._read_buffer_size)
            elif self._read_future is not None:
                try:
                    pos = self._find_read_pos()
                except UnsatisfiableReadError:
                    pass
                else:
                    if pos is not None:
                        self._read_from_buffer(pos)
            if self._state is not None:
                self.io_loop.remove_handler(self.fileno())
                self._state = None
            self.close_fd()
            self._closed = True
        self._signal_closed()

    def _signal_closed(self) -> None:
        if False:
            print('Hello World!')
        futures = []
        if self._read_future is not None:
            futures.append(self._read_future)
            self._read_future = None
        futures += [future for (_, future) in self._write_futures]
        self._write_futures.clear()
        if self._connect_future is not None:
            futures.append(self._connect_future)
            self._connect_future = None
        for future in futures:
            if not future.done():
                future.set_exception(StreamClosedError(real_error=self.error))
            try:
                future.exception()
            except asyncio.CancelledError:
                pass
        if self._ssl_connect_future is not None:
            if not self._ssl_connect_future.done():
                if self.error is not None:
                    self._ssl_connect_future.set_exception(self.error)
                else:
                    self._ssl_connect_future.set_exception(StreamClosedError())
            self._ssl_connect_future.exception()
            self._ssl_connect_future = None
        if self._close_callback is not None:
            cb = self._close_callback
            self._close_callback = None
            self.io_loop.add_callback(cb)
        self._write_buffer = None

    def reading(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Returns ``True`` if we are currently reading from the stream.'
        return self._read_future is not None

    def writing(self) -> bool:
        if False:
            while True:
                i = 10
        'Returns ``True`` if we are currently writing to the stream.'
        return bool(self._write_buffer)

    def closed(self) -> bool:
        if False:
            print('Hello World!')
        'Returns ``True`` if the stream has been closed.'
        return self._closed

    def set_nodelay(self, value: bool) -> None:
        if False:
            i = 10
            return i + 15
        "Sets the no-delay flag for this stream.\n\n        By default, data written to TCP streams may be held for a time\n        to make the most efficient use of bandwidth (according to\n        Nagle's algorithm).  The no-delay flag requests that data be\n        written as soon as possible, even if doing so would consume\n        additional bandwidth.\n\n        This flag is currently defined only for TCP-based ``IOStreams``.\n\n        .. versionadded:: 3.1\n        "
        pass

    def _handle_connect(self) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def _handle_events(self, fd: Union[int, ioloop._Selectable], events: int) -> None:
        if False:
            while True:
                i = 10
        if self.closed():
            gen_log.warning('Got events for closed stream %s', fd)
            return
        try:
            if self._connecting:
                self._handle_connect()
            if self.closed():
                return
            if events & self.io_loop.READ:
                self._handle_read()
            if self.closed():
                return
            if events & self.io_loop.WRITE:
                self._handle_write()
            if self.closed():
                return
            if events & self.io_loop.ERROR:
                self.error = self.get_fd_error()
                self.io_loop.add_callback(self.close)
                return
            state = self.io_loop.ERROR
            if self.reading():
                state |= self.io_loop.READ
            if self.writing():
                state |= self.io_loop.WRITE
            if state == self.io_loop.ERROR and self._read_buffer_size == 0:
                state |= self.io_loop.READ
            if state != self._state:
                assert self._state is not None, "shouldn't happen: _handle_events without self._state"
                self._state = state
                self.io_loop.update_handler(self.fileno(), self._state)
        except UnsatisfiableReadError as e:
            gen_log.info('Unsatisfiable read, closing connection: %s' % e)
            self.close(exc_info=e)
        except Exception as e:
            gen_log.error('Uncaught exception, closing connection.', exc_info=True)
            self.close(exc_info=e)
            raise

    def _read_to_buffer_loop(self) -> Optional[int]:
        if False:
            while True:
                i = 10
        if self._read_bytes is not None:
            target_bytes = self._read_bytes
        elif self._read_max_bytes is not None:
            target_bytes = self._read_max_bytes
        elif self.reading():
            target_bytes = None
        else:
            target_bytes = 0
        next_find_pos = 0
        while not self.closed():
            if self._read_to_buffer() == 0:
                break
            if target_bytes is not None and self._read_buffer_size >= target_bytes:
                break
            if self._read_buffer_size >= next_find_pos:
                pos = self._find_read_pos()
                if pos is not None:
                    return pos
                next_find_pos = self._read_buffer_size * 2
        return self._find_read_pos()

    def _handle_read(self) -> None:
        if False:
            print('Hello World!')
        try:
            pos = self._read_to_buffer_loop()
        except UnsatisfiableReadError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            gen_log.warning('error on read: %s' % e)
            self.close(exc_info=e)
            return
        if pos is not None:
            self._read_from_buffer(pos)

    def _start_read(self) -> Future:
        if False:
            while True:
                i = 10
        if self._read_future is not None:
            self._check_closed()
            assert self._read_future is None, 'Already reading'
        self._read_future = Future()
        return self._read_future

    def _finish_read(self, size: int) -> None:
        if False:
            while True:
                i = 10
        if self._user_read_buffer:
            self._read_buffer = self._after_user_read_buffer or bytearray()
            self._after_user_read_buffer = None
            self._read_buffer_size = len(self._read_buffer)
            self._user_read_buffer = False
            result = size
        else:
            result = self._consume(size)
        if self._read_future is not None:
            future = self._read_future
            self._read_future = None
            future_set_result_unless_cancelled(future, result)
        self._maybe_add_error_listener()

    def _try_inline_read(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Attempt to complete the current read operation from buffered data.\n\n        If the read can be completed without blocking, schedules the\n        read callback on the next IOLoop iteration; otherwise starts\n        listening for reads on the socket.\n        '
        pos = self._find_read_pos()
        if pos is not None:
            self._read_from_buffer(pos)
            return
        self._check_closed()
        pos = self._read_to_buffer_loop()
        if pos is not None:
            self._read_from_buffer(pos)
            return
        if not self.closed():
            self._add_io_state(ioloop.IOLoop.READ)

    def _read_to_buffer(self) -> Optional[int]:
        if False:
            print('Hello World!')
        'Reads from the socket and appends the result to the read buffer.\n\n        Returns the number of bytes read.  Returns 0 if there is nothing\n        to read (i.e. the read returns EWOULDBLOCK or equivalent).  On\n        error closes the socket and raises an exception.\n        '
        try:
            while True:
                try:
                    if self._user_read_buffer:
                        buf = memoryview(self._read_buffer)[self._read_buffer_size:]
                    else:
                        buf = bytearray(self.read_chunk_size)
                    bytes_read = self.read_from_fd(buf)
                except (socket.error, IOError, OSError) as e:
                    if self._is_connreset(e):
                        self.close(exc_info=e)
                        return None
                    self.close(exc_info=e)
                    raise
                break
            if bytes_read is None:
                return 0
            elif bytes_read == 0:
                self.close()
                return 0
            if not self._user_read_buffer:
                self._read_buffer += memoryview(buf)[:bytes_read]
            self._read_buffer_size += bytes_read
        finally:
            del buf
        if self._read_buffer_size > self.max_buffer_size:
            gen_log.error('Reached maximum read buffer size')
            self.close()
            raise StreamBufferFullError('Reached maximum read buffer size')
        return bytes_read

    def _read_from_buffer(self, pos: int) -> None:
        if False:
            print('Hello World!')
        'Attempts to complete the currently-pending read from the buffer.\n\n        The argument is either a position in the read buffer or None,\n        as returned by _find_read_pos.\n        '
        self._read_bytes = self._read_delimiter = self._read_regex = None
        self._read_partial = False
        self._finish_read(pos)

    def _find_read_pos(self) -> Optional[int]:
        if False:
            print('Hello World!')
        'Attempts to find a position in the read buffer that satisfies\n        the currently-pending read.\n\n        Returns a position in the buffer if the current read can be satisfied,\n        or None if it cannot.\n        '
        if self._read_bytes is not None and (self._read_buffer_size >= self._read_bytes or (self._read_partial and self._read_buffer_size > 0)):
            num_bytes = min(self._read_bytes, self._read_buffer_size)
            return num_bytes
        elif self._read_delimiter is not None:
            if self._read_buffer:
                loc = self._read_buffer.find(self._read_delimiter)
                if loc != -1:
                    delimiter_len = len(self._read_delimiter)
                    self._check_max_bytes(self._read_delimiter, loc + delimiter_len)
                    return loc + delimiter_len
                self._check_max_bytes(self._read_delimiter, self._read_buffer_size)
        elif self._read_regex is not None:
            if self._read_buffer:
                m = self._read_regex.search(self._read_buffer)
                if m is not None:
                    loc = m.end()
                    self._check_max_bytes(self._read_regex, loc)
                    return loc
                self._check_max_bytes(self._read_regex, self._read_buffer_size)
        return None

    def _check_max_bytes(self, delimiter: Union[bytes, Pattern], size: int) -> None:
        if False:
            i = 10
            return i + 15
        if self._read_max_bytes is not None and size > self._read_max_bytes:
            raise UnsatisfiableReadError('delimiter %r not found within %d bytes' % (delimiter, self._read_max_bytes))

    def _handle_write(self) -> None:
        if False:
            return 10
        while True:
            size = len(self._write_buffer)
            if not size:
                break
            assert size > 0
            try:
                if _WINDOWS:
                    size = 128 * 1024
                num_bytes = self.write_to_fd(self._write_buffer.peek(size))
                if num_bytes == 0:
                    break
                self._write_buffer.advance(num_bytes)
                self._total_write_done_index += num_bytes
            except BlockingIOError:
                break
            except (socket.error, IOError, OSError) as e:
                if not self._is_connreset(e):
                    gen_log.warning('Write error on %s: %s', self.fileno(), e)
                self.close(exc_info=e)
                return
        while self._write_futures:
            (index, future) = self._write_futures[0]
            if index > self._total_write_done_index:
                break
            self._write_futures.popleft()
            future_set_result_unless_cancelled(future, None)

    def _consume(self, loc: int) -> bytes:
        if False:
            print('Hello World!')
        if loc == 0:
            return b''
        assert loc <= self._read_buffer_size
        b = memoryview(self._read_buffer)[:loc].tobytes()
        self._read_buffer_size -= loc
        del self._read_buffer[:loc]
        return b

    def _check_closed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.closed():
            raise StreamClosedError(real_error=self.error)

    def _maybe_add_error_listener(self) -> None:
        if False:
            return 10
        if self._state is None or self._state == ioloop.IOLoop.ERROR:
            if not self.closed() and self._read_buffer_size == 0 and (self._close_callback is not None):
                self._add_io_state(ioloop.IOLoop.READ)

    def _add_io_state(self, state: int) -> None:
        if False:
            i = 10
            return i + 15
        "Adds `state` (IOLoop.{READ,WRITE} flags) to our event handler.\n\n        Implementation notes: Reads and writes have a fast path and a\n        slow path.  The fast path reads synchronously from socket\n        buffers, while the slow path uses `_add_io_state` to schedule\n        an IOLoop callback.\n\n        To detect closed connections, we must have called\n        `_add_io_state` at some point, but we want to delay this as\n        much as possible so we don't have to set an `IOLoop.ERROR`\n        listener that will be overwritten by the next slow-path\n        operation. If a sequence of fast-path ops do not end in a\n        slow-path op, (e.g. for an @asynchronous long-poll request),\n        we must add the error handler.\n\n        TODO: reevaluate this now that callbacks are gone.\n\n        "
        if self.closed():
            return
        if self._state is None:
            self._state = ioloop.IOLoop.ERROR | state
            self.io_loop.add_handler(self.fileno(), self._handle_events, self._state)
        elif not self._state & state:
            self._state = self._state | state
            self.io_loop.update_handler(self.fileno(), self._state)

    def _is_connreset(self, exc: BaseException) -> bool:
        if False:
            i = 10
            return i + 15
        'Return ``True`` if exc is ECONNRESET or equivalent.\n\n        May be overridden in subclasses.\n        '
        return isinstance(exc, (socket.error, IOError)) and errno_from_exception(exc) in _ERRNO_CONNRESET

class IOStream(BaseIOStream):
    """Socket-based `IOStream` implementation.

    This class supports the read and write methods from `BaseIOStream`
    plus a `connect` method.

    The ``socket`` parameter may either be connected or unconnected.
    For server operations the socket is the result of calling
    `socket.accept <socket.socket.accept>`.  For client operations the
    socket is created with `socket.socket`, and may either be
    connected before passing it to the `IOStream` or connected with
    `IOStream.connect`.

    A very simple (and broken) HTTP client using this class:

    .. testcode::

        import socket
        import tornado

        async def main():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
            stream = tornado.iostream.IOStream(s)
            await stream.connect(("friendfeed.com", 80))
            await stream.write(b"GET / HTTP/1.0\\r\\nHost: friendfeed.com\\r\\n\\r\\n")
            header_data = await stream.read_until(b"\\r\\n\\r\\n")
            headers = {}
            for line in header_data.split(b"\\r\\n"):
                parts = line.split(b":")
                if len(parts) == 2:
                    headers[parts[0].strip()] = parts[1].strip()
            body_data = await stream.read_bytes(int(headers[b"Content-Length"]))
            print(body_data)
            stream.close()

        if __name__ == '__main__':
            asyncio.run(main())

    .. testoutput::
       :hide:

    """

    def __init__(self, socket: socket.socket, *args: Any, **kwargs: Any) -> None:
        if False:
            return 10
        self.socket = socket
        self.socket.setblocking(False)
        super().__init__(*args, **kwargs)

    def fileno(self) -> Union[int, ioloop._Selectable]:
        if False:
            i = 10
            return i + 15
        return self.socket

    def close_fd(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.socket.close()
        self.socket = None

    def get_fd_error(self) -> Optional[Exception]:
        if False:
            print('Hello World!')
        errno = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        return socket.error(errno, os.strerror(errno))

    def read_from_fd(self, buf: Union[bytearray, memoryview]) -> Optional[int]:
        if False:
            return 10
        try:
            return self.socket.recv_into(buf, len(buf))
        except BlockingIOError:
            return None
        finally:
            del buf

    def write_to_fd(self, data: memoryview) -> int:
        if False:
            return 10
        try:
            return self.socket.send(data)
        finally:
            del data

    def connect(self: _IOStreamType, address: Any, server_hostname: Optional[str]=None) -> 'Future[_IOStreamType]':
        if False:
            return 10
        'Connects the socket to a remote address without blocking.\n\n        May only be called if the socket passed to the constructor was\n        not previously connected.  The address parameter is in the\n        same format as for `socket.connect <socket.socket.connect>` for\n        the type of socket passed to the IOStream constructor,\n        e.g. an ``(ip, port)`` tuple.  Hostnames are accepted here,\n        but will be resolved synchronously and block the IOLoop.\n        If you have a hostname instead of an IP address, the `.TCPClient`\n        class is recommended instead of calling this method directly.\n        `.TCPClient` will do asynchronous DNS resolution and handle\n        both IPv4 and IPv6.\n\n        If ``callback`` is specified, it will be called with no\n        arguments when the connection is completed; if not this method\n        returns a `.Future` (whose result after a successful\n        connection will be the stream itself).\n\n        In SSL mode, the ``server_hostname`` parameter will be used\n        for certificate validation (unless disabled in the\n        ``ssl_options``) and SNI (if supported; requires Python\n        2.7.9+).\n\n        Note that it is safe to call `IOStream.write\n        <BaseIOStream.write>` while the connection is pending, in\n        which case the data will be written as soon as the connection\n        is ready.  Calling `IOStream` read methods before the socket is\n        connected works on some platforms but is non-portable.\n\n        .. versionchanged:: 4.0\n            If no callback is given, returns a `.Future`.\n\n        .. versionchanged:: 4.2\n           SSL certificates are validated by default; pass\n           ``ssl_options=dict(cert_reqs=ssl.CERT_NONE)`` or a\n           suitably-configured `ssl.SSLContext` to the\n           `SSLIOStream` constructor to disable.\n\n        .. versionchanged:: 6.0\n\n           The ``callback`` argument was removed. Use the returned\n           `.Future` instead.\n\n        '
        self._connecting = True
        future = Future()
        self._connect_future = typing.cast('Future[IOStream]', future)
        try:
            self.socket.connect(address)
        except BlockingIOError:
            pass
        except socket.error as e:
            if future is None:
                gen_log.warning('Connect error on fd %s: %s', self.socket.fileno(), e)
            self.close(exc_info=e)
            return future
        self._add_io_state(self.io_loop.WRITE)
        return future

    def start_tls(self, server_side: bool, ssl_options: Optional[Union[Dict[str, Any], ssl.SSLContext]]=None, server_hostname: Optional[str]=None) -> Awaitable['SSLIOStream']:
        if False:
            i = 10
            return i + 15
        "Convert this `IOStream` to an `SSLIOStream`.\n\n        This enables protocols that begin in clear-text mode and\n        switch to SSL after some initial negotiation (such as the\n        ``STARTTLS`` extension to SMTP and IMAP).\n\n        This method cannot be used if there are outstanding reads\n        or writes on the stream, or if there is any data in the\n        IOStream's buffer (data in the operating system's socket\n        buffer is allowed).  This means it must generally be used\n        immediately after reading or writing the last clear-text\n        data.  It can also be used immediately after connecting,\n        before any reads or writes.\n\n        The ``ssl_options`` argument may be either an `ssl.SSLContext`\n        object or a dictionary of keyword arguments for the\n        `ssl.SSLContext.wrap_socket` function.  The ``server_hostname`` argument\n        will be used for certificate validation unless disabled\n        in the ``ssl_options``.\n\n        This method returns a `.Future` whose result is the new\n        `SSLIOStream`.  After this method has been called,\n        any other operation on the original stream is undefined.\n\n        If a close callback is defined on this stream, it will be\n        transferred to the new stream.\n\n        .. versionadded:: 4.0\n\n        .. versionchanged:: 4.2\n           SSL certificates are validated by default; pass\n           ``ssl_options=dict(cert_reqs=ssl.CERT_NONE)`` or a\n           suitably-configured `ssl.SSLContext` to disable.\n        "
        if self._read_future or self._write_futures or self._connect_future or self._closed or self._read_buffer or self._write_buffer:
            raise ValueError('IOStream is not idle; cannot convert to SSL')
        if ssl_options is None:
            if server_side:
                ssl_options = _server_ssl_defaults
            else:
                ssl_options = _client_ssl_defaults
        socket = self.socket
        self.io_loop.remove_handler(socket)
        self.socket = None
        socket = ssl_wrap_socket(socket, ssl_options, server_hostname=server_hostname, server_side=server_side, do_handshake_on_connect=False)
        orig_close_callback = self._close_callback
        self._close_callback = None
        future = Future()
        ssl_stream = SSLIOStream(socket, ssl_options=ssl_options)
        ssl_stream.set_close_callback(orig_close_callback)
        ssl_stream._ssl_connect_future = future
        ssl_stream.max_buffer_size = self.max_buffer_size
        ssl_stream.read_chunk_size = self.read_chunk_size
        return future

    def _handle_connect(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        try:
            err = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        except socket.error as e:
            if errno_from_exception(e) == errno.ENOPROTOOPT:
                err = 0
        if err != 0:
            self.error = socket.error(err, os.strerror(err))
            if self._connect_future is None:
                gen_log.warning('Connect error on fd %s: %s', self.socket.fileno(), errno.errorcode[err])
            self.close()
            return
        if self._connect_future is not None:
            future = self._connect_future
            self._connect_future = None
            future_set_result_unless_cancelled(future, self)
        self._connecting = False

    def set_nodelay(self, value: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.socket is not None and self.socket.family in (socket.AF_INET, socket.AF_INET6):
            try:
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1 if value else 0)
            except socket.error as e:
                if e.errno != errno.EINVAL and (not self._is_connreset(e)):
                    raise

class SSLIOStream(IOStream):
    """A utility class to write to and read from a non-blocking SSL socket.

    If the socket passed to the constructor is already connected,
    it should be wrapped with::

        ssl.SSLContext(...).wrap_socket(sock, do_handshake_on_connect=False, **kwargs)

    before constructing the `SSLIOStream`.  Unconnected sockets will be
    wrapped when `IOStream.connect` is finished.
    """
    socket = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'The ``ssl_options`` keyword argument may either be an\n        `ssl.SSLContext` object or a dictionary of keywords arguments\n        for `ssl.SSLContext.wrap_socket`\n        '
        self._ssl_options = kwargs.pop('ssl_options', _client_ssl_defaults)
        super().__init__(*args, **kwargs)
        self._ssl_accepting = True
        self._handshake_reading = False
        self._handshake_writing = False
        self._server_hostname = None
        try:
            self.socket.getpeername()
        except socket.error:
            pass
        else:
            self._add_io_state(self.io_loop.WRITE)

    def reading(self) -> bool:
        if False:
            print('Hello World!')
        return self._handshake_reading or super().reading()

    def writing(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._handshake_writing or super().writing()

    def _do_ssl_handshake(self) -> None:
        if False:
            print('Hello World!')
        try:
            self._handshake_reading = False
            self._handshake_writing = False
            self.socket.do_handshake()
        except ssl.SSLError as err:
            if err.args[0] == ssl.SSL_ERROR_WANT_READ:
                self._handshake_reading = True
                return
            elif err.args[0] == ssl.SSL_ERROR_WANT_WRITE:
                self._handshake_writing = True
                return
            elif err.args[0] in (ssl.SSL_ERROR_EOF, ssl.SSL_ERROR_ZERO_RETURN):
                return self.close(exc_info=err)
            elif err.args[0] == ssl.SSL_ERROR_SSL:
                try:
                    peer = self.socket.getpeername()
                except Exception:
                    peer = '(not connected)'
                gen_log.warning('SSL Error on %s %s: %s', self.socket.fileno(), peer, err)
                return self.close(exc_info=err)
            raise
        except ssl.CertificateError as err:
            return self.close(exc_info=err)
        except socket.error as err:
            if self._is_connreset(err) or err.args[0] in (0, errno.EBADF, errno.ENOTCONN):
                return self.close(exc_info=err)
            raise
        except AttributeError as err:
            return self.close(exc_info=err)
        else:
            self._ssl_accepting = False
            assert ssl.HAS_SNI
            self._finish_ssl_connect()

    def _finish_ssl_connect(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._ssl_connect_future is not None:
            future = self._ssl_connect_future
            self._ssl_connect_future = None
            future_set_result_unless_cancelled(future, self)

    def _handle_read(self) -> None:
        if False:
            while True:
                i = 10
        if self._ssl_accepting:
            self._do_ssl_handshake()
            return
        super()._handle_read()

    def _handle_write(self) -> None:
        if False:
            return 10
        if self._ssl_accepting:
            self._do_ssl_handshake()
            return
        super()._handle_write()

    def connect(self, address: Tuple, server_hostname: Optional[str]=None) -> 'Future[SSLIOStream]':
        if False:
            return 10
        self._server_hostname = server_hostname
        fut = super().connect(address)
        fut.add_done_callback(lambda f: f.exception())
        return self.wait_for_handshake()

    def _handle_connect(self) -> None:
        if False:
            print('Hello World!')
        super()._handle_connect()
        if self.closed():
            return
        self.io_loop.remove_handler(self.socket)
        old_state = self._state
        assert old_state is not None
        self._state = None
        self.socket = ssl_wrap_socket(self.socket, self._ssl_options, server_hostname=self._server_hostname, do_handshake_on_connect=False, server_side=False)
        self._add_io_state(old_state)

    def wait_for_handshake(self) -> 'Future[SSLIOStream]':
        if False:
            while True:
                i = 10
        "Wait for the initial SSL handshake to complete.\n\n        If a ``callback`` is given, it will be called with no\n        arguments once the handshake is complete; otherwise this\n        method returns a `.Future` which will resolve to the\n        stream itself after the handshake is complete.\n\n        Once the handshake is complete, information such as\n        the peer's certificate and NPN/ALPN selections may be\n        accessed on ``self.socket``.\n\n        This method is intended for use on server-side streams\n        or after using `IOStream.start_tls`; it should not be used\n        with `IOStream.connect` (which already waits for the\n        handshake to complete). It may only be called once per stream.\n\n        .. versionadded:: 4.2\n\n        .. versionchanged:: 6.0\n\n           The ``callback`` argument was removed. Use the returned\n           `.Future` instead.\n\n        "
        if self._ssl_connect_future is not None:
            raise RuntimeError('Already waiting')
        future = self._ssl_connect_future = Future()
        if not self._ssl_accepting:
            self._finish_ssl_connect()
        return future

    def write_to_fd(self, data: memoryview) -> int:
        if False:
            return 10
        if len(data) >> 30:
            data = memoryview(data)[:1 << 30]
        try:
            return self.socket.send(data)
        except ssl.SSLError as e:
            if e.args[0] == ssl.SSL_ERROR_WANT_WRITE:
                return 0
            raise
        finally:
            del data

    def read_from_fd(self, buf: Union[bytearray, memoryview]) -> Optional[int]:
        if False:
            while True:
                i = 10
        try:
            if self._ssl_accepting:
                return None
            if len(buf) >> 30:
                buf = memoryview(buf)[:1 << 30]
            try:
                return self.socket.recv_into(buf, len(buf))
            except ssl.SSLError as e:
                if e.args[0] == ssl.SSL_ERROR_WANT_READ:
                    return None
                else:
                    raise
            except BlockingIOError:
                return None
        finally:
            del buf

    def _is_connreset(self, e: BaseException) -> bool:
        if False:
            return 10
        if isinstance(e, ssl.SSLError) and e.args[0] == ssl.SSL_ERROR_EOF:
            return True
        return super()._is_connreset(e)

class PipeIOStream(BaseIOStream):
    """Pipe-based `IOStream` implementation.

    The constructor takes an integer file descriptor (such as one returned
    by `os.pipe`) rather than an open file object.  Pipes are generally
    one-way, so a `PipeIOStream` can be used for reading or writing but not
    both.

    ``PipeIOStream`` is only available on Unix-based platforms.
    """

    def __init__(self, fd: int, *args: Any, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        self.fd = fd
        self._fio = io.FileIO(self.fd, 'r+')
        if sys.platform == 'win32':
            raise AssertionError('PipeIOStream is not supported on Windows')
        os.set_blocking(fd, False)
        super().__init__(*args, **kwargs)

    def fileno(self) -> int:
        if False:
            print('Hello World!')
        return self.fd

    def close_fd(self) -> None:
        if False:
            i = 10
            return i + 15
        self._fio.close()

    def write_to_fd(self, data: memoryview) -> int:
        if False:
            print('Hello World!')
        try:
            return os.write(self.fd, data)
        finally:
            del data

    def read_from_fd(self, buf: Union[bytearray, memoryview]) -> Optional[int]:
        if False:
            print('Hello World!')
        try:
            return self._fio.readinto(buf)
        except (IOError, OSError) as e:
            if errno_from_exception(e) == errno.EBADF:
                self.close(exc_info=e)
                return None
            else:
                raise
        finally:
            del buf

def doctests() -> Any:
    if False:
        print('Hello World!')
    import doctest
    return doctest.DocTestSuite()