"""Utilities for implementing `nbio_interface.AbstractIOServices` for
pika connection adapters.

"""
import collections
import errno
import functools
import logging
import numbers
import os
import socket
import ssl
import sys
import traceback
from pika.adapters.utils.nbio_interface import AbstractIOReference, AbstractStreamTransport
import pika.compat
import pika.diagnostic_utils
_TRY_IO_AGAIN_SOCK_ERROR_CODES = (errno.EAGAIN, errno.EWOULDBLOCK)
_CONNECTION_IN_PROGRESS_SOCK_ERROR_CODES = (errno.EINPROGRESS, errno.EWOULDBLOCK)
_LOGGER = logging.getLogger(__name__)
_log_exceptions = pika.diagnostic_utils.create_log_exception_decorator(_LOGGER)

def check_callback_arg(callback, name):
    if False:
        while True:
            i = 10
    'Raise TypeError if callback is not callable\n\n    :param callback: callback to check\n    :param name: Name to include in exception text\n    :raises TypeError:\n\n    '
    if not callable(callback):
        raise TypeError('{} must be callable, but got {!r}'.format(name, callback))

def check_fd_arg(fd):
    if False:
        return 10
    'Raise TypeError if file descriptor is not an integer\n\n    :param fd: file descriptor\n    :raises TypeError:\n\n    '
    if not isinstance(fd, numbers.Integral):
        raise TypeError(f'Paramter must be a file descriptor, but got {fd!r}')

def _retry_on_sigint(func):
    if False:
        while True:
            i = 10
    'Function decorator for retrying on SIGINT.\n\n    '

    @functools.wraps(func)
    def retry_sigint_wrap(*args, **kwargs):
        if False:
            print('Hello World!')
        'Wrapper for decorated function'
        while True:
            try:
                return func(*args, **kwargs)
            except pika.compat.SOCKET_ERROR as error:
                if error.errno == errno.EINTR:
                    continue
                else:
                    raise
    return retry_sigint_wrap

class SocketConnectionMixin:
    """Implements
    `pika.adapters.utils.nbio_interface.AbstractIOServices.connect_socket()`
    on top of
    `pika.adapters.utils.nbio_interface.AbstractFileDescriptorServices` and
    basic `pika.adapters.utils.nbio_interface.AbstractIOServices`.

    """

    def connect_socket(self, sock, resolved_addr, on_done):
        if False:
            print('Hello World!')
        'Implement\n        :py:meth:`.nbio_interface.AbstractIOServices.connect_socket()`.\n\n        '
        return _AsyncSocketConnector(nbio=self, sock=sock, resolved_addr=resolved_addr, on_done=on_done).start()

class StreamingConnectionMixin:
    """Implements
    `.nbio_interface.AbstractIOServices.create_streaming_connection()` on
    top of `.nbio_interface.AbstractFileDescriptorServices` and basic
    `nbio_interface.AbstractIOServices` services.

    """

    def create_streaming_connection(self, protocol_factory, sock, on_done, ssl_context=None, server_hostname=None):
        if False:
            print('Hello World!')
        'Implement\n        :py:meth:`.nbio_interface.AbstractIOServices.create_streaming_connection()`.\n\n        '
        try:
            return _AsyncStreamConnector(nbio=self, protocol_factory=protocol_factory, sock=sock, ssl_context=ssl_context, server_hostname=server_hostname, on_done=on_done).start()
        except Exception as error:
            _LOGGER.error('create_streaming_connection(%s) failed: %r', sock, error)
            try:
                sock.close()
            except Exception as error:
                _LOGGER.error('%s.close() failed: %r', sock, error)
            raise

class _AsyncServiceAsyncHandle(AbstractIOReference):
    """This module's adaptation of `.nbio_interface.AbstractIOReference`

    """

    def __init__(self, subject):
        if False:
            return 10
        '\n        :param subject: subject of the reference containing a `cancel()` method\n\n        '
        self._cancel = subject.cancel

    def cancel(self):
        if False:
            i = 10
            return i + 15
        'Cancel pending operation\n\n        :returns: False if was already done or cancelled; True otherwise\n        :rtype: bool\n\n        '
        return self._cancel()

class _AsyncSocketConnector:
    """Connects the given non-blocking socket asynchronously using
    `.nbio_interface.AbstractFileDescriptorServices` and basic
    `.nbio_interface.AbstractIOServices`. Used for implementing
    `.nbio_interface.AbstractIOServices.connect_socket()`.
    """
    _STATE_NOT_STARTED = 0
    _STATE_ACTIVE = 1
    _STATE_CANCELED = 2
    _STATE_COMPLETED = 3

    def __init__(self, nbio, sock, resolved_addr, on_done):
        if False:
            print('Hello World!')
        "\n        :param AbstractIOServices | AbstractFileDescriptorServices nbio:\n        :param socket.socket sock: non-blocking socket that needs to be\n            connected via `socket.socket.connect()`\n        :param tuple resolved_addr: resolved destination address/port two-tuple\n            which is compatible with the given's socket's address family\n        :param callable on_done: user callback that takes None upon successful\n            completion or exception upon error (check for `BaseException`) as\n            its only arg. It will not be called if the operation was cancelled.\n        :raises ValueError: if host portion of `resolved_addr` is not an IP\n            address or is inconsistent with the socket's address family as\n            validated via `socket.inet_pton()`\n        "
        check_callback_arg(on_done, 'on_done')
        try:
            socket.inet_pton(sock.family, resolved_addr[0])
        except Exception as error:
            if not hasattr(socket, 'inet_pton'):
                _LOGGER.debug('Unable to check resolved address: no socket.inet_pton().')
            else:
                msg = 'Invalid or unresolved IP address {!r} for socket {}: {!r}'.format(resolved_addr, sock, error)
                _LOGGER.error(msg)
                raise ValueError(msg)
        self._nbio = nbio
        self._sock = sock
        self._addr = resolved_addr
        self._on_done = on_done
        self._state = self._STATE_NOT_STARTED
        self._watching_socket_events = False

    @_log_exceptions
    def _cleanup(self):
        if False:
            i = 10
            return i + 15
        'Remove socket watcher, if any\n\n        '
        if self._watching_socket_events:
            self._watching_socket_events = False
            self._nbio.remove_writer(self._sock.fileno())

    def start(self):
        if False:
            while True:
                i = 10
        'Start asynchronous connection establishment.\n\n        :rtype: AbstractIOReference\n        '
        assert self._state == self._STATE_NOT_STARTED, ('_AsyncSocketConnector.start(): expected _STATE_NOT_STARTED', self._state)
        self._state = self._STATE_ACTIVE
        self._nbio.add_callback_threadsafe(self._start_async)
        return _AsyncServiceAsyncHandle(self)

    def cancel(self):
        if False:
            while True:
                i = 10
        "Cancel pending connection request without calling user's completion\n        callback.\n\n        :returns: False if was already done or cancelled; True otherwise\n        :rtype: bool\n\n        "
        if self._state == self._STATE_ACTIVE:
            self._state = self._STATE_CANCELED
            _LOGGER.debug('User canceled connection request for %s to %s', self._sock, self._addr)
            self._cleanup()
            return True
        _LOGGER.debug('_AsyncSocketConnector cancel requested when not ACTIVE: state=%s; %s', self._state, self._sock)
        return False

    @_log_exceptions
    def _report_completion(self, result):
        if False:
            print('Hello World!')
        "Advance to COMPLETED state, remove socket watcher, and invoke user's\n        completion callback.\n\n        :param BaseException | None result: value to pass in user's callback\n\n        "
        _LOGGER.debug('_AsyncSocketConnector._report_completion(%r); %s', result, self._sock)
        assert isinstance(result, (BaseException, type(None))), ('_AsyncSocketConnector._report_completion() expected exception or None as result.', result)
        assert self._state == self._STATE_ACTIVE, ('_AsyncSocketConnector._report_completion() expected _STATE_NOT_STARTED', self._state)
        self._state = self._STATE_COMPLETED
        self._cleanup()
        self._on_done(result)

    @_log_exceptions
    def _start_async(self):
        if False:
            return 10
        "Called as callback from I/O loop to kick-start the workflow, so it's\n        safe to call user's completion callback from here, if needed\n\n        "
        if self._state != self._STATE_ACTIVE:
            _LOGGER.debug('Abandoning sock=%s connection establishment to %s due to inactive state=%s', self._sock, self._addr, self._state)
            return
        try:
            self._sock.connect(self._addr)
        except (Exception, pika.compat.SOCKET_ERROR) as error:
            if isinstance(error, pika.compat.SOCKET_ERROR) and error.errno in _CONNECTION_IN_PROGRESS_SOCK_ERROR_CODES:
                pass
            else:
                _LOGGER.error('%s.connect(%s) failed: %r', self._sock, self._addr, error)
                self._report_completion(error)
                return
        try:
            self._nbio.set_writer(self._sock.fileno(), self._on_writable)
        except Exception as error:
            _LOGGER.exception('async.set_writer(%s) failed: %r', self._sock, error)
            self._report_completion(error)
            return
        else:
            self._watching_socket_events = True
            _LOGGER.debug('Connection-establishment is in progress for %s.', self._sock)

    @_log_exceptions
    def _on_writable(self):
        if False:
            while True:
                i = 10
        "Called when socket connects or fails to. Check for predicament and\n        invoke user's completion callback.\n\n        "
        if self._state != self._STATE_ACTIVE:
            _LOGGER.error('Socket connection-establishment event watcher called in inactive state (ignoring): %s; state=%s', self._sock, self._state)
            return
        error_code = self._sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        if not error_code:
            _LOGGER.info('Socket connected: %s', self._sock)
            result = None
        else:
            error_msg = os.strerror(error_code)
            _LOGGER.error('Socket failed to connect: %s; error=%s (%s)', self._sock, error_code, error_msg)
            result = pika.compat.SOCKET_ERROR(error_code, error_msg)
        self._report_completion(result)

class _AsyncStreamConnector:
    """Performs asynchronous SSL session establishment, if requested, on the
    already-connected socket and links the streaming transport to protocol.
    Used for implementing
    `.nbio_interface.AbstractIOServices.create_streaming_connection()`.

    """
    _STATE_NOT_STARTED = 0
    _STATE_ACTIVE = 1
    _STATE_CANCELED = 2
    _STATE_COMPLETED = 3

    def __init__(self, nbio, protocol_factory, sock, ssl_context, server_hostname, on_done):
        if False:
            while True:
                i = 10
        '\n        NOTE: We take ownership of the given socket upon successful completion\n        of the constructor.\n\n        See `AbstractIOServices.create_streaming_connection()` for detailed\n        documentation of the corresponding args.\n\n        :param AbstractIOServices | AbstractFileDescriptorServices nbio:\n        :param callable protocol_factory:\n        :param socket.socket sock:\n        :param ssl.SSLContext | None ssl_context:\n        :param str | None server_hostname:\n        :param callable on_done:\n\n        '
        check_callback_arg(protocol_factory, 'protocol_factory')
        check_callback_arg(on_done, 'on_done')
        if not isinstance(ssl_context, (type(None), ssl.SSLContext)):
            raise ValueError('Expected ssl_context=None | ssl.SSLContext, but got {!r}'.format(ssl_context))
        if server_hostname is not None and ssl_context is None:
            raise ValueError('Non-None server_hostname must not be passed without ssl context')
        try:
            sock.getpeername()
        except Exception as error:
            raise ValueError('Expected connected socket, but getpeername() failed: error={!r}; {}; '.format(error, sock))
        self._nbio = nbio
        self._protocol_factory = protocol_factory
        self._sock = sock
        self._ssl_context = ssl_context
        self._server_hostname = server_hostname
        self._on_done = on_done
        self._state = self._STATE_NOT_STARTED
        self._watching_socket = False

    @_log_exceptions
    def _cleanup(self, close):
        if False:
            print('Hello World!')
        'Cancel pending async operations, if any\n\n        :param bool close: close the socket if true\n        '
        _LOGGER.debug('_AsyncStreamConnector._cleanup(%r)', close)
        if self._watching_socket:
            _LOGGER.debug('_AsyncStreamConnector._cleanup(%r): removing RdWr; %s', close, self._sock)
            self._watching_socket = False
            self._nbio.remove_reader(self._sock.fileno())
            self._nbio.remove_writer(self._sock.fileno())
        try:
            if close:
                _LOGGER.debug('_AsyncStreamConnector._cleanup(%r): closing socket; %s', close, self._sock)
                try:
                    self._sock.close()
                except Exception as error:
                    _LOGGER.exception('_sock.close() failed: error=%r; %s', error, self._sock)
                    raise
        finally:
            self._sock = None
            self._nbio = None
            self._protocol_factory = None
            self._ssl_context = None
            self._server_hostname = None
            self._on_done = None

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        'Kick off the workflow\n\n        :rtype: AbstractIOReference\n        '
        _LOGGER.debug('_AsyncStreamConnector.start(); %s', self._sock)
        assert self._state == self._STATE_NOT_STARTED, ('_AsyncStreamConnector.start() expected _STATE_NOT_STARTED', self._state)
        self._state = self._STATE_ACTIVE
        self._nbio.add_callback_threadsafe(self._start_async)
        return _AsyncServiceAsyncHandle(self)

    def cancel(self):
        if False:
            return 10
        "Cancel pending connection request without calling user's completion\n        callback.\n\n        :returns: False if was already done or cancelled; True otherwise\n        :rtype: bool\n\n        "
        if self._state == self._STATE_ACTIVE:
            self._state = self._STATE_CANCELED
            _LOGGER.debug('User canceled streaming linkup for %s', self._sock)
            self._cleanup(close=True)
            return True
        _LOGGER.debug('_AsyncStreamConnector cancel requested when not ACTIVE: state=%s; %s', self._state, self._sock)
        return False

    @_log_exceptions
    def _report_completion(self, result):
        if False:
            return 10
        "Advance to COMPLETED state, cancel async operation(s), and invoke\n        user's completion callback.\n\n        :param BaseException | tuple result: value to pass in user's callback.\n            `tuple(transport, protocol)` on success, exception on error\n\n        "
        _LOGGER.debug('_AsyncStreamConnector._report_completion(%r); %s', result, self._sock)
        assert isinstance(result, (BaseException, tuple)), ('_AsyncStreamConnector._report_completion() expected exception or tuple as result.', result, self._state)
        assert self._state == self._STATE_ACTIVE, ('_AsyncStreamConnector._report_completion() expected _STATE_ACTIVE', self._state)
        self._state = self._STATE_COMPLETED
        try:
            self._on_done(result)
        except Exception:
            _LOGGER.exception('%r: _on_done(%r) failed.', self._report_completion, result)
            raise
        finally:
            self._cleanup(close=isinstance(result, BaseException))

    @_log_exceptions
    def _start_async(self):
        if False:
            return 10
        "Called as callback from I/O loop to kick-start the workflow, so it's\n        safe to call user's completion callback from here if needed\n\n        "
        _LOGGER.debug('_AsyncStreamConnector._start_async(); %s', self._sock)
        if self._state != self._STATE_ACTIVE:
            _LOGGER.debug('Abandoning streaming linkup due to inactive state transition; state=%s; %s; .', self._state, self._sock)
            return
        if self._ssl_context is None:
            self._linkup()
        else:
            _LOGGER.debug('Starting SSL handshake on %s', self._sock)
            try:
                self._sock = self._ssl_context.wrap_socket(self._sock, server_side=False, do_handshake_on_connect=False, suppress_ragged_eofs=False, server_hostname=self._server_hostname)
            except Exception as error:
                _LOGGER.exception('SSL wrap_socket(%s) failed: %r', self._sock, error)
                self._report_completion(error)
                return
            self._do_ssl_handshake()

    @_log_exceptions
    def _linkup(self):
        if False:
            return 10
        "Connection is ready: instantiate and link up transport and protocol,\n        and invoke user's completion callback.\n\n        "
        _LOGGER.debug('_AsyncStreamConnector._linkup()')
        transport = None
        try:
            try:
                protocol = self._protocol_factory()
            except Exception as error:
                _LOGGER.exception('protocol_factory() failed: error=%r; %s', error, self._sock)
                raise
            if self._ssl_context is None:
                try:
                    transport = _AsyncPlaintextTransport(self._sock, protocol, self._nbio)
                except Exception as error:
                    _LOGGER.exception('PlainTransport() failed: error=%r; %s', error, self._sock)
                    raise
            else:
                try:
                    transport = _AsyncSSLTransport(self._sock, protocol, self._nbio)
                except Exception as error:
                    _LOGGER.exception('SSLTransport() failed: error=%r; %s', error, self._sock)
                    raise
            _LOGGER.debug('_linkup(): created transport %r', transport)
            try:
                protocol.connection_made(transport)
            except Exception as error:
                _LOGGER.exception('protocol.connection_made(%r) failed: error=%r; %s', transport, error, self._sock)
                raise
            _LOGGER.debug('_linkup(): introduced transport to protocol %r; %r', transport, protocol)
        except Exception as error:
            result = error
        else:
            result = (transport, protocol)
        self._report_completion(result)

    @_log_exceptions
    def _do_ssl_handshake(self):
        if False:
            return 10
        'Perform asynchronous SSL handshake on the already wrapped socket\n\n        '
        _LOGGER.debug('_AsyncStreamConnector._do_ssl_handshake()')
        if self._state != self._STATE_ACTIVE:
            _LOGGER.debug('_do_ssl_handshake: Abandoning streaming linkup due to inactive state transition; state=%s; %s; .', self._state, self._sock)
            return
        done = False
        try:
            try:
                self._sock.do_handshake()
            except ssl.SSLError as error:
                if error.errno == ssl.SSL_ERROR_WANT_READ:
                    _LOGGER.debug('SSL handshake wants read; %s.', self._sock)
                    self._watching_socket = True
                    self._nbio.set_reader(self._sock.fileno(), self._do_ssl_handshake)
                    self._nbio.remove_writer(self._sock.fileno())
                elif error.errno == ssl.SSL_ERROR_WANT_WRITE:
                    _LOGGER.debug('SSL handshake wants write. %s', self._sock)
                    self._watching_socket = True
                    self._nbio.set_writer(self._sock.fileno(), self._do_ssl_handshake)
                    self._nbio.remove_reader(self._sock.fileno())
                else:
                    raise
            else:
                done = True
                _LOGGER.info('SSL handshake completed successfully: %s', self._sock)
        except Exception as error:
            _LOGGER.exception('SSL do_handshake failed: error=%r; %s', error, self._sock)
            self._report_completion(error)
            return
        if done:
            _LOGGER.debug('_do_ssl_handshake: removing watchers ahead of linkup: %s', self._sock)
            self._nbio.remove_reader(self._sock.fileno())
            self._nbio.remove_writer(self._sock.fileno())
            self._watching_socket = False
            _LOGGER.debug('_do_ssl_handshake: pre-linkup removal of watchers is done; %s', self._sock)
            self._linkup()

class _AsyncTransportBase(AbstractStreamTransport):
    """Base class for `_AsyncPlaintextTransport` and `_AsyncSSLTransport`.

    """
    _STATE_ACTIVE = 1
    _STATE_FAILED = 2
    _STATE_ABORTED_BY_USER = 3
    _STATE_COMPLETED = 4
    _MAX_RECV_BYTES = 4096
    _MAX_CONSUME_BYTES = 1024 * 100

    class RxEndOfFile(OSError):
        """We raise this internally when EOF (empty read) is detected on input.

        """

        def __init__(self):
            if False:
                print('Hello World!')
            super(_AsyncTransportBase.RxEndOfFile, self).__init__(-1, 'End of input stream (EOF)')

    def __init__(self, sock, protocol, nbio):
        if False:
            while True:
                i = 10
        '\n\n        :param socket.socket | ssl.SSLSocket sock: connected socket\n        :param pika.adapters.utils.nbio_interface.AbstractStreamProtocol protocol:\n            corresponding protocol in this transport/protocol pairing; the\n            protocol already had its `connection_made()` method called.\n        :param AbstractIOServices | AbstractFileDescriptorServices nbio:\n\n        '
        _LOGGER.debug('_AsyncTransportBase.__init__: %s', sock)
        self._sock = sock
        self._protocol = protocol
        self._nbio = nbio
        self._state = self._STATE_ACTIVE
        self._tx_buffers = collections.deque()
        self._tx_buffered_byte_count = 0

    def abort(self):
        if False:
            i = 10
            return i + 15
        "Close connection abruptly without waiting for pending I/O to\n        complete. Will invoke the corresponding protocol's `connection_lost()`\n        method asynchronously (not in context of the abort() call).\n\n        :raises Exception: Exception-based exception on error\n        "
        _LOGGER.info('Aborting transport connection: state=%s; %s', self._state, self._sock)
        self._initiate_abort(None)

    def get_protocol(self):
        if False:
            while True:
                i = 10
        'Return the protocol linked to this transport.\n\n        :rtype: pika.adapters.utils.nbio_interface.AbstractStreamProtocol\n        '
        return self._protocol

    def get_write_buffer_size(self):
        if False:
            return 10
        '\n        :returns: Current size of output data buffered by the transport\n        :rtype: int\n        '
        return self._tx_buffered_byte_count

    def _buffer_tx_data(self, data):
        if False:
            while True:
                i = 10
        'Buffer the given data until it can be sent asynchronously.\n\n        :param bytes data:\n        :raises ValueError: if called with empty data\n\n        '
        if not data:
            _LOGGER.error('write() called with empty data: state=%s; %s', self._state, self._sock)
            raise ValueError(f'write() called with empty data {data!r}')
        if self._state != self._STATE_ACTIVE:
            _LOGGER.debug('Ignoring write() called during inactive state: state=%s; %s', self._state, self._sock)
            return
        self._tx_buffers.append(data)
        self._tx_buffered_byte_count += len(data)

    def _consume(self):
        if False:
            for i in range(10):
                print('nop')
        'Utility method for use by subclasses to ingest data from socket and\n        dispatch it to protocol\'s `data_received()` method socket-specific\n        "try again" exception, per-event data consumption limit is reached,\n        transport becomes inactive, or a fatal failure.\n\n        Consumes up to `self._MAX_CONSUME_BYTES` to prevent event starvation or\n        until state becomes inactive (e.g., `protocol.data_received()` callback\n        aborts the transport)\n\n        :raises: Whatever the corresponding `sock.recv()` raises except the\n                 socket error with errno.EINTR\n        :raises: Whatever the `protocol.data_received()` callback raises\n        :raises _AsyncTransportBase.RxEndOfFile: upon shutdown of input stream\n\n        '
        bytes_consumed = 0
        while self._state == self._STATE_ACTIVE and bytes_consumed < self._MAX_CONSUME_BYTES:
            data = self._sigint_safe_recv(self._sock, self._MAX_RECV_BYTES)
            bytes_consumed += len(data)
            if not data:
                _LOGGER.error('Socket EOF; %s', self._sock)
                raise self.RxEndOfFile()
            try:
                self._protocol.data_received(data)
            except Exception as error:
                _LOGGER.exception('protocol.data_received() failed: error=%r; %s', error, self._sock)
                raise

    def _produce(self):
        if False:
            i = 10
            return i + 15
        'Utility method for use by subclasses to emit data from tx_buffers.\n        This method sends chunks from `tx_buffers` until all chunks are\n        exhausted or sending is interrupted by an exception. Maintains integrity\n        of `self.tx_buffers`.\n\n        :raises: whatever the corresponding `sock.send()` raises except the\n                 socket error with errno.EINTR\n\n        '
        while self._tx_buffers:
            num_bytes_sent = self._sigint_safe_send(self._sock, self._tx_buffers[0])
            chunk = self._tx_buffers.popleft()
            if num_bytes_sent < len(chunk):
                _LOGGER.debug('Partial send, requeing remaining data; %s of %s', num_bytes_sent, len(chunk))
                self._tx_buffers.appendleft(chunk[num_bytes_sent:])
            self._tx_buffered_byte_count -= num_bytes_sent
            assert self._tx_buffered_byte_count >= 0, ('_AsyncTransportBase._produce() tx buffer size underflow', self._tx_buffered_byte_count, self._state)

    @staticmethod
    @_retry_on_sigint
    def _sigint_safe_recv(sock, max_bytes):
        if False:
            return 10
        'Receive data from socket, retrying on SIGINT.\n\n        :param sock: stream or SSL socket\n        :param max_bytes: maximum number of bytes to receive\n        :returns: received data or empty bytes uppon end of file\n        :rtype: bytes\n        :raises: whatever the corresponding `sock.recv()` raises except socket\n                 error with errno.EINTR\n\n        '
        return sock.recv(max_bytes)

    @staticmethod
    @_retry_on_sigint
    def _sigint_safe_send(sock, data):
        if False:
            print('Hello World!')
        'Send data to socket, retrying on SIGINT.\n\n        :param sock: stream or SSL socket\n        :param data: data bytes to send\n        :returns: number of bytes actually sent\n        :rtype: int\n        :raises: whatever the corresponding `sock.send()` raises except socket\n                 error with errno.EINTR\n\n        '
        return sock.send(data)

    @_log_exceptions
    def _deactivate(self):
        if False:
            i = 10
            return i + 15
        'Unregister the transport from I/O events\n\n        '
        if self._state == self._STATE_ACTIVE:
            _LOGGER.info('Deactivating transport: state=%s; %s', self._state, self._sock)
            self._nbio.remove_reader(self._sock.fileno())
            self._nbio.remove_writer(self._sock.fileno())
            self._tx_buffers.clear()

    @_log_exceptions
    def _close_and_finalize(self):
        if False:
            print('Hello World!')
        "Close the transport's socket and unlink the transport it from\n        references to other assets (protocol, etc.)\n\n        "
        if self._state != self._STATE_COMPLETED:
            _LOGGER.info('Closing transport socket and unlinking: state=%s; %s', self._state, self._sock)
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except pika.compat.SOCKET_ERROR:
                pass
            self._sock.close()
            self._sock = None
            self._protocol = None
            self._nbio = None
            self._state = self._STATE_COMPLETED

    @_log_exceptions
    def _initiate_abort(self, error):
        if False:
            while True:
                i = 10
        "Initiate asynchronous abort of the transport that concludes with a\n        call to the protocol's `connection_lost()` method. No flushing of\n        output buffers will take place.\n\n        :param BaseException | None error: None if being canceled by user,\n            including via falsie return value from protocol.eof_received;\n            otherwise the exception corresponding to the the failed connection.\n        "
        _LOGGER.info('_AsyncTransportBase._initate_abort(): Initiating abrupt asynchronous transport shutdown: state=%s; error=%r; %s', self._state, error, self._sock)
        assert self._state != self._STATE_COMPLETED, ('_AsyncTransportBase._initate_abort() expected non-_STATE_COMPLETED', self._state)
        if self._state == self._STATE_COMPLETED:
            return
        self._deactivate()
        if error is None:
            if self._state == self._STATE_ABORTED_BY_USER:
                _LOGGER.debug('_AsyncTransportBase._initiate_abort(): ignoring - user-abort already pending.')
                return
            self._state = self._STATE_ABORTED_BY_USER
        else:
            if self._state != self._STATE_ACTIVE:
                assert self._state == self._STATE_ABORTED_BY_USER, ('_AsyncTransportBase._initate_abort() expected _STATE_ABORTED_BY_USER', self._state)
                return
            self._state = self._STATE_FAILED
        self._nbio.add_callback_threadsafe(functools.partial(self._connection_lost_notify_async, error))

    @_log_exceptions
    def _connection_lost_notify_async(self, error):
        if False:
            while True:
                i = 10
        "Handle aborting of transport either due to socket error or user-\n        initiated `abort()` call. Must be called from an I/O loop callback owned\n        by us in order to avoid reentry into user code from user's API call into\n        the transport.\n\n        :param BaseException | None error: None if being canceled by user;\n            otherwise the exception corresponding to the the failed connection.\n        "
        _LOGGER.debug('Concluding transport shutdown: state=%s; error=%r', self._state, error)
        if self._state == self._STATE_COMPLETED:
            return
        if error is not None and self._state != self._STATE_FAILED:
            assert self._state == self._STATE_ABORTED_BY_USER, ('_AsyncTransportBase._connection_lost_notify_async() expected _STATE_ABORTED_BY_USER', self._state)
            return
        try:
            self._protocol.connection_lost(error)
        except Exception as exc:
            _LOGGER.exception('protocol.connection_lost(%r) failed: exc=%r; %s', error, exc, self._sock)
            raise
        finally:
            self._close_and_finalize()

class _AsyncPlaintextTransport(_AsyncTransportBase):
    """Implementation of `nbio_interface.AbstractStreamTransport` for a
    plaintext connection.

    """

    def __init__(self, sock, protocol, nbio):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        :param socket.socket sock: non-blocking connected socket\n        :param pika.adapters.utils.nbio_interface.AbstractStreamProtocol protocol:\n            corresponding protocol in this transport/protocol pairing; the\n            protocol already had its `connection_made()` method called.\n        :param AbstractIOServices | AbstractFileDescriptorServices nbio:\n\n        '
        super().__init__(sock, protocol, nbio)
        self._nbio.set_reader(self._sock.fileno(), self._on_socket_readable)

    def write(self, data):
        if False:
            return 10
        'Buffer the given data until it can be sent asynchronously.\n\n        :param bytes data:\n        :raises ValueError: if called with empty data\n\n        '
        if self._state != self._STATE_ACTIVE:
            _LOGGER.debug('Ignoring write() called during inactive state: state=%s; %s', self._state, self._sock)
            return
        assert data, ('_AsyncPlaintextTransport.write(): empty data from user.', data, self._state)
        tx_buffer_was_empty = self.get_write_buffer_size() == 0
        self._buffer_tx_data(data)
        if tx_buffer_was_empty:
            self._nbio.set_writer(self._sock.fileno(), self._on_socket_writable)
            _LOGGER.debug('Turned on writability watcher: %s', self._sock)

    @_log_exceptions
    def _on_socket_readable(self):
        if False:
            return 10
        'Ingest data from socket and dispatch it to protocol until exception\n        occurs (typically EAGAIN or EWOULDBLOCK), per-event data consumption\n        limit is reached, transport becomes inactive, or failure.\n\n        '
        if self._state != self._STATE_ACTIVE:
            _LOGGER.debug('Ignoring readability notification due to inactive state: state=%s; %s', self._state, self._sock)
            return
        try:
            self._consume()
        except self.RxEndOfFile:
            try:
                keep_open = self._protocol.eof_received()
            except Exception as error:
                _LOGGER.exception('protocol.eof_received() failed: error=%r; %s', error, self._sock)
                self._initiate_abort(error)
            else:
                if keep_open:
                    _LOGGER.info('protocol.eof_received() elected to keep open: %s', self._sock)
                    self._nbio.remove_reader(self._sock.fileno())
                else:
                    _LOGGER.info('protocol.eof_received() elected to close: %s', self._sock)
                    self._initiate_abort(None)
        except (Exception, pika.compat.SOCKET_ERROR) as error:
            if isinstance(error, pika.compat.SOCKET_ERROR) and error.errno in _TRY_IO_AGAIN_SOCK_ERROR_CODES:
                _LOGGER.debug('Recv would block on %s', self._sock)
            else:
                _LOGGER.exception("_AsyncBaseTransport._consume() failed, aborting connection: error=%r; sock=%s; Caller's stack:\n%s", error, self._sock, ''.join(traceback.format_exception(*sys.exc_info())))
                self._initiate_abort(error)
        else:
            if self._state != self._STATE_ACTIVE:
                _LOGGER.debug('Leaving Plaintext consumer due to inactive state: state=%s; %s', self._state, self._sock)

    @_log_exceptions
    def _on_socket_writable(self):
        if False:
            for i in range(10):
                print('nop')
        'Handle writable socket notification\n\n        '
        if self._state != self._STATE_ACTIVE:
            _LOGGER.debug('Ignoring writability notification due to inactive state: state=%s; %s', self._state, self._sock)
            return
        assert self._tx_buffers, ('_AsyncPlaintextTransport._on_socket_writable() called, but _tx_buffers is empty.', self._state)
        try:
            self._produce()
        except (Exception, pika.compat.SOCKET_ERROR) as error:
            if isinstance(error, pika.compat.SOCKET_ERROR) and error.errno in _TRY_IO_AGAIN_SOCK_ERROR_CODES:
                _LOGGER.debug('Send would block on %s', self._sock)
            else:
                _LOGGER.exception("_AsyncBaseTransport._produce() failed, aborting connection: error=%r; sock=%s; Caller's stack:\n%s", error, self._sock, ''.join(traceback.format_exception(*sys.exc_info())))
                self._initiate_abort(error)
        else:
            if not self._tx_buffers:
                self._nbio.remove_writer(self._sock.fileno())
                _LOGGER.debug('Turned off writability watcher: %s', self._sock)

class _AsyncSSLTransport(_AsyncTransportBase):
    """Implementation of `.nbio_interface.AbstractStreamTransport` for an SSL
    connection.

    """

    def __init__(self, sock, protocol, nbio):
        if False:
            while True:
                i = 10
        '\n\n        :param ssl.SSLSocket sock: non-blocking connected socket\n        :param pika.adapters.utils.nbio_interface.AbstractStreamProtocol protocol:\n            corresponding protocol in this transport/protocol pairing; the\n            protocol already had its `connection_made()` method called.\n        :param AbstractIOServices | AbstractFileDescriptorServices nbio:\n\n        '
        super().__init__(sock, protocol, nbio)
        self._ssl_readable_action = self._consume
        self._ssl_writable_action = None
        self._nbio.set_reader(self._sock.fileno(), self._on_socket_readable)
        self._nbio.add_callback_threadsafe(self._on_socket_readable)

    def write(self, data):
        if False:
            while True:
                i = 10
        'Buffer the given data until it can be sent asynchronously.\n\n        :param bytes data:\n        :raises ValueError: if called with empty data\n\n        '
        if self._state != self._STATE_ACTIVE:
            _LOGGER.debug('Ignoring write() called during inactive state: state=%s; %s', self._state, self._sock)
            return
        assert data, ('_AsyncSSLTransport.write(): empty data from user.', data, self._state)
        tx_buffer_was_empty = self.get_write_buffer_size() == 0
        self._buffer_tx_data(data)
        if tx_buffer_was_empty and self._ssl_writable_action is None:
            self._ssl_writable_action = self._produce
            self._nbio.set_writer(self._sock.fileno(), self._on_socket_writable)
            _LOGGER.debug('Turned on writability watcher: %s', self._sock)

    @_log_exceptions
    def _on_socket_readable(self):
        if False:
            for i in range(10):
                print('nop')
        'Handle readable socket indication\n\n        '
        if self._state != self._STATE_ACTIVE:
            _LOGGER.debug('Ignoring readability notification due to inactive state: state=%s; %s', self._state, self._sock)
            return
        if self._ssl_readable_action:
            try:
                self._ssl_readable_action()
            except Exception as error:
                self._initiate_abort(error)
        else:
            _LOGGER.debug('SSL readable action was suppressed: ssl_writable_action=%r; %s', self._ssl_writable_action, self._sock)

    @_log_exceptions
    def _on_socket_writable(self):
        if False:
            print('Hello World!')
        'Handle writable socket notification\n\n        '
        if self._state != self._STATE_ACTIVE:
            _LOGGER.debug('Ignoring writability notification due to inactive state: state=%s; %s', self._state, self._sock)
            return
        if self._ssl_writable_action:
            try:
                self._ssl_writable_action()
            except Exception as error:
                self._initiate_abort(error)
        else:
            _LOGGER.debug('SSL writable action was suppressed: ssl_readable_action=%r; %s', self._ssl_readable_action, self._sock)

    @_log_exceptions
    def _consume(self):
        if False:
            for i in range(10):
                print('nop')
        '[override] Ingest data from socket and dispatch it to protocol until\n        exception occurs (typically ssl.SSLError with\n        SSL_ERROR_WANT_READ/WRITE), per-event data consumption limit is reached,\n        transport becomes inactive, or failure.\n\n        Update consumer/producer registration.\n\n        :raises Exception: error that signals that connection needs to be\n            aborted\n        '
        next_consume_on_readable = True
        try:
            super()._consume()
        except ssl.SSLError as error:
            if error.errno == ssl.SSL_ERROR_WANT_READ:
                _LOGGER.debug('SSL ingester wants read: %s', self._sock)
            elif error.errno == ssl.SSL_ERROR_WANT_WRITE:
                _LOGGER.debug('SSL ingester wants write: %s', self._sock)
                next_consume_on_readable = False
            else:
                _LOGGER.exception("_AsyncBaseTransport._consume() failed, aborting connection: error=%r; sock=%s; Caller's stack:\n%s", error, self._sock, ''.join(traceback.format_exception(*sys.exc_info())))
                raise
        else:
            if self._state != self._STATE_ACTIVE:
                _LOGGER.debug('Leaving SSL consumer due to inactive state: state=%s; %s', self._state, self._sock)
                return
            self._nbio.add_callback_threadsafe(self._on_socket_readable)
        if next_consume_on_readable:
            if not self._ssl_readable_action:
                self._nbio.set_reader(self._sock.fileno(), self._on_socket_readable)
            self._ssl_readable_action = self._consume
            if self._ssl_writable_action == self._consume:
                self._nbio.remove_writer(self._sock.fileno())
                self._ssl_writable_action = None
        else:
            if not self._ssl_writable_action:
                self._nbio.set_writer(self._sock.fileno(), self._on_socket_writable)
            self._ssl_writable_action = self._consume
            if self._ssl_readable_action:
                self._nbio.remove_reader(self._sock.fileno())
                self._ssl_readable_action = None
        if self._tx_buffers and (not self._ssl_writable_action):
            self._ssl_writable_action = self._produce
            self._nbio.set_writer(self._sock.fileno(), self._on_socket_writable)

    @_log_exceptions
    def _produce(self):
        if False:
            while True:
                i = 10
        '[override] Emit data from tx_buffers all chunks are exhausted or\n        sending is interrupted by an exception (typically ssl.SSLError with\n        SSL_ERROR_WANT_READ/WRITE).\n\n        Update consumer/producer registration.\n\n        :raises Exception: error that signals that connection needs to be\n            aborted\n\n        '
        next_produce_on_writable = None
        try:
            super()._produce()
        except ssl.SSLError as error:
            if error.errno == ssl.SSL_ERROR_WANT_READ:
                _LOGGER.debug('SSL emitter wants read: %s', self._sock)
                next_produce_on_writable = False
            elif error.errno == ssl.SSL_ERROR_WANT_WRITE:
                _LOGGER.debug('SSL emitter wants write: %s', self._sock)
                next_produce_on_writable = True
            else:
                _LOGGER.exception("_AsyncBaseTransport._produce() failed, aborting connection: error=%r; sock=%s; Caller's stack:\n%s", error, self._sock, ''.join(traceback.format_exception(*sys.exc_info())))
                raise
        else:
            assert not self._tx_buffers, ('_AsyncSSLTransport._produce(): no exception from parent class, but data remains in _tx_buffers.', len(self._tx_buffers))
        if self._tx_buffers:
            assert next_produce_on_writable is not None, ('_AsyncSSLTransport._produce(): next_produce_on_writable is still None', self._state)
            if next_produce_on_writable:
                if not self._ssl_writable_action:
                    self._nbio.set_writer(self._sock.fileno(), self._on_socket_writable)
                self._ssl_writable_action = self._produce
                if self._ssl_readable_action == self._produce:
                    self._nbio.remove_reader(self._sock.fileno())
                    self._ssl_readable_action = None
            else:
                if not self._ssl_readable_action:
                    self._nbio.set_reader(self._sock.fileno(), self._on_socket_readable)
                self._ssl_readable_action = self._produce
                if self._ssl_writable_action:
                    self._nbio.remove_writer(self._sock.fileno())
                    self._ssl_writable_action = None
        elif self._ssl_readable_action == self._produce:
            self._nbio.remove_reader(self._sock.fileno())
            self._ssl_readable_action = None
            assert self._ssl_writable_action != self._produce, ('_AsyncSSLTransport._produce(): with empty tx_buffers, writable_action cannot be _produce when readable is _produce', self._state)
        else:
            assert self._ssl_writable_action == self._produce, ('_AsyncSSLTransport._produce(): with empty tx_buffers, expected writable_action as _produce when readable_action is not _produce', 'writable_action:', self._ssl_writable_action, 'readable_action:', self._ssl_readable_action, 'state:', self._state)
            self._ssl_writable_action = None
            self._nbio.remove_writer(self._sock.fileno())
        if not self._ssl_readable_action:
            self._ssl_readable_action = self._consume
            self._nbio.set_reader(self._sock.fileno(), self._on_socket_readable)
            self._nbio.add_callback_threadsafe(self._on_socket_readable)
        elif self._sock.pending():
            self._nbio.add_callback_threadsafe(self._on_socket_readable)