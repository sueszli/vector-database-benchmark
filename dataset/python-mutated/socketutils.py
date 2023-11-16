"""At its heart, Python can be viewed as an extension of the C
programming language. Springing from the most popular systems
programming language has made Python itself a great language for
systems programming. One key to success in this domain is Python's
very serviceable :mod:`socket` module and its :class:`socket.socket`
type.

The ``socketutils`` module provides natural next steps to the ``socket``
builtin: straightforward, tested building blocks for higher-level
protocols.

The :class:`BufferedSocket` wraps an ordinary socket, providing a
layer of intuitive buffering for both sending and receiving. This
facilitates parsing messages from streams, i.e., all sockets with type
``SOCK_STREAM``. The BufferedSocket enables receiving until the next
relevant token, up to a certain size, or until the connection is
closed. For all of these, it provides consistent APIs to size
limiting, as well as timeouts that are compatible with multiple
concurrency paradigms. Use it to parse the next one-off text or binary
socket protocol you encounter.

This module also provides the :class:`NetstringSocket`, a pure-Python
implementation of `the Netstring protocol`_, built on top of the
:class:`BufferedSocket`, serving as a ready-made, production-grade example.

Special thanks to `Kurt Rose`_ for his original authorship and all his
contributions on this module. Also thanks to `Daniel J. Bernstein`_, the
original author of `Netstring`_.

.. _the Netstring protocol: https://en.wikipedia.org/wiki/Netstring
.. _Kurt Rose: https://github.com/doublereedkurt
.. _Daniel J. Bernstein: https://cr.yp.to/
.. _Netstring: https://cr.yp.to/proto/netstrings.txt

"""
import time
import socket
try:
    from threading import RLock
except Exception:

    class RLock(object):
        """Dummy reentrant lock for builds without threads"""

        def __enter__(self):
            if False:
                i = 10
                return i + 15
            pass

        def __exit__(self, exctype, excinst, exctb):
            if False:
                return 10
            pass
try:
    from .typeutils import make_sentinel
    _UNSET = make_sentinel(var_name='_UNSET')
except ImportError:
    _UNSET = object()
DEFAULT_TIMEOUT = 10
DEFAULT_MAXSIZE = 32 * 1024
_RECV_LARGE_MAXSIZE = 1024 ** 5

class BufferedSocket(object):
    """Mainly provides recv_until and recv_size. recv, send, sendall, and
    peek all function as similarly as possible to the built-in socket
    API.

    This type has been tested against both the built-in socket type as
    well as those from gevent and eventlet. It also features support
    for sockets with timeouts set to 0 (aka nonblocking), provided the
    caller is prepared to handle the EWOULDBLOCK exceptions.

    Args:
        sock (socket): The connected socket to be wrapped.
        timeout (float): The default timeout for sends and recvs, in
            seconds. Set to ``None`` for no timeout, and 0 for
            nonblocking. Defaults to *sock*'s own timeout if already set,
            and 10 seconds otherwise.
        maxsize (int): The default maximum number of bytes to be received
            into the buffer before it is considered full and raises an
            exception. Defaults to 32 kilobytes.
        recvsize (int): The number of bytes to recv for every
            lower-level :meth:`socket.recv` call. Defaults to *maxsize*.

    *timeout* and *maxsize* can both be overridden on individual socket
    operations.

    All ``recv`` methods return bytestrings (:class:`bytes`) and can
    raise :exc:`socket.error`. :exc:`Timeout`,
    :exc:`ConnectionClosed`, and :exc:`MessageTooLong` all inherit
    from :exc:`socket.error` and exist to provide better error
    messages. Received bytes are always buffered, even if an exception
    is raised. Use :meth:`BufferedSocket.getrecvbuffer` to retrieve
    partial recvs.

    BufferedSocket does not replace the built-in socket by any
    means. While the overlapping parts of the API are kept parallel to
    the built-in :class:`socket.socket`, BufferedSocket does not
    inherit from socket, and most socket functionality is only
    available on the underlying socket. :meth:`socket.getpeername`,
    :meth:`socket.getsockname`, :meth:`socket.fileno`, and others are
    only available on the underlying socket that is wrapped. Use the
    ``BufferedSocket.sock`` attribute to access it. See the examples
    for more information on how to use BufferedSockets with built-in
    sockets.

    The BufferedSocket is threadsafe, but consider the semantics of
    your protocol before accessing a single socket from multiple
    threads. Similarly, once the BufferedSocket is constructed, avoid
    using the underlying socket directly. Only use it for operations
    unrelated to messages, e.g., :meth:`socket.getpeername`.

    """

    def __init__(self, sock, timeout=_UNSET, maxsize=DEFAULT_MAXSIZE, recvsize=_UNSET):
        if False:
            while True:
                i = 10
        self.sock = sock
        self.rbuf = b''
        self.sbuf = []
        self.maxsize = int(maxsize)
        if timeout is _UNSET:
            if self.sock.gettimeout() is None:
                self.timeout = DEFAULT_TIMEOUT
            else:
                self.timeout = self.sock.gettimeout()
        elif timeout is None:
            self.timeout = timeout
        else:
            self.timeout = float(timeout)
        if recvsize is _UNSET:
            self._recvsize = self.maxsize
        else:
            self._recvsize = int(recvsize)
        self._send_lock = RLock()
        self._recv_lock = RLock()

    def settimeout(self, timeout):
        if False:
            i = 10
            return i + 15
        'Set the default *timeout* for future operations, in seconds.'
        self.timeout = timeout

    def gettimeout(self):
        if False:
            i = 10
            return i + 15
        return self.timeout

    def setblocking(self, blocking):
        if False:
            print('Hello World!')
        self.timeout = None if blocking else 0.0

    def setmaxsize(self, maxsize):
        if False:
            print('Hello World!')
        'Set the default maximum buffer size *maxsize* for future\n        operations, in bytes. Does not truncate the current buffer.\n        '
        self.maxsize = maxsize

    def getrecvbuffer(self):
        if False:
            i = 10
            return i + 15
        'Returns the receive buffer bytestring (rbuf).'
        with self._recv_lock:
            return self.rbuf

    def getsendbuffer(self):
        if False:
            return 10
        'Returns a copy of the send buffer list.'
        with self._send_lock:
            return b''.join(self.sbuf)

    def recv(self, size, flags=0, timeout=_UNSET):
        if False:
            for i in range(10):
                print('nop')
        'Returns **up to** *size* bytes, using the internal buffer before\n        performing a single :meth:`socket.recv` operation.\n\n        Args:\n            size (int): The maximum number of bytes to receive.\n            flags (int): Kept for API compatibility with sockets. Only\n                the default, ``0``, is valid.\n            timeout (float): The timeout for this operation. Can be\n                ``0`` for nonblocking and ``None`` for no\n                timeout. Defaults to the value set in the constructor\n                of BufferedSocket.\n\n        If the operation does not complete in *timeout* seconds, a\n        :exc:`Timeout` is raised. Much like the built-in\n        :class:`socket.socket`, if this method returns an empty string,\n        then the socket is closed and recv buffer is empty. Further\n        calls to recv will raise :exc:`socket.error`.\n\n        '
        with self._recv_lock:
            if timeout is _UNSET:
                timeout = self.timeout
            if flags:
                raise ValueError('non-zero flags not supported: %r' % flags)
            if len(self.rbuf) >= size:
                (data, self.rbuf) = (self.rbuf[:size], self.rbuf[size:])
                return data
            if self.rbuf:
                (ret, self.rbuf) = (self.rbuf, b'')
                return ret
            self.sock.settimeout(timeout)
            try:
                data = self.sock.recv(self._recvsize)
            except socket.timeout:
                raise Timeout(timeout)
            if len(data) > size:
                (data, self.rbuf) = (data[:size], data[size:])
        return data

    def peek(self, size, timeout=_UNSET):
        if False:
            return 10
        "Returns *size* bytes from the socket and/or internal buffer. Bytes\n        are retained in BufferedSocket's internal recv buffer. To only\n        see bytes in the recv buffer, use :meth:`getrecvbuffer`.\n\n        Args:\n            size (int): The exact number of bytes to peek at\n            timeout (float): The timeout for this operation. Can be 0 for\n                nonblocking and None for no timeout. Defaults to the value\n                set in the constructor of BufferedSocket.\n\n        If the appropriate number of bytes cannot be fetched from the\n        buffer and socket before *timeout* expires, then a\n        :exc:`Timeout` will be raised. If the connection is closed, a\n        :exc:`ConnectionClosed` will be raised.\n        "
        with self._recv_lock:
            if len(self.rbuf) >= size:
                return self.rbuf[:size]
            data = self.recv_size(size, timeout=timeout)
            self.rbuf = data + self.rbuf
        return data

    def recv_close(self, timeout=_UNSET, maxsize=_UNSET):
        if False:
            return 10
        'Receive until the connection is closed, up to *maxsize* bytes. If\n        more than *maxsize* bytes are received, raises :exc:`MessageTooLong`.\n        '
        with self._recv_lock:
            if maxsize is _UNSET:
                maxsize = self.maxsize
            if maxsize is None:
                maxsize = _RECV_LARGE_MAXSIZE
            try:
                recvd = self.recv_size(maxsize + 1, timeout)
            except ConnectionClosed:
                (ret, self.rbuf) = (self.rbuf, b'')
            else:
                self.rbuf = recvd + self.rbuf
                size_read = min(maxsize, len(self.rbuf))
                raise MessageTooLong(size_read)
        return ret

    def recv_until(self, delimiter, timeout=_UNSET, maxsize=_UNSET, with_delimiter=False):
        if False:
            for i in range(10):
                print('nop')
        'Receive until *delimiter* is found, *maxsize* bytes have been read,\n        or *timeout* is exceeded.\n\n        Args:\n            delimiter (bytes): One or more bytes to be searched for\n                in the socket stream.\n            timeout (float): The timeout for this operation. Can be 0 for\n                nonblocking and None for no timeout. Defaults to the value\n                set in the constructor of BufferedSocket.\n            maxsize (int): The maximum size for the internal buffer.\n                Defaults to the value set in the constructor.\n            with_delimiter (bool): Whether or not to include the\n                delimiter in the output. ``False`` by default, but\n                ``True`` is useful in cases where one is simply\n                forwarding the messages.\n\n        ``recv_until`` will raise the following exceptions:\n\n          * :exc:`Timeout` if more than *timeout* seconds expire.\n          * :exc:`ConnectionClosed` if the underlying socket is closed\n            by the sending end.\n          * :exc:`MessageTooLong` if the delimiter is not found in the\n            first *maxsize* bytes.\n          * :exc:`socket.error` if operating in nonblocking mode\n            (*timeout* equal to 0), or if some unexpected socket error\n            occurs, such as operating on a closed socket.\n\n        '
        with self._recv_lock:
            if maxsize is _UNSET:
                maxsize = self.maxsize
            if maxsize is None:
                maxsize = _RECV_LARGE_MAXSIZE
            if timeout is _UNSET:
                timeout = self.timeout
            len_delimiter = len(delimiter)
            sock = self.sock
            recvd = bytearray(self.rbuf)
            start = time.time()
            find_offset_start = 0
            if not timeout:
                sock.settimeout(timeout)
            try:
                while 1:
                    offset = recvd.find(delimiter, find_offset_start, maxsize)
                    if offset != -1:
                        if with_delimiter:
                            offset += len_delimiter
                            rbuf_offset = offset
                        else:
                            rbuf_offset = offset + len_delimiter
                        break
                    elif len(recvd) > maxsize:
                        raise MessageTooLong(maxsize, delimiter)
                    if timeout:
                        cur_timeout = timeout - (time.time() - start)
                        if cur_timeout <= 0.0:
                            raise socket.timeout()
                        sock.settimeout(cur_timeout)
                    nxt = sock.recv(self._recvsize)
                    if not nxt:
                        args = (len(recvd), delimiter)
                        msg = 'connection closed after reading %s bytes without finding symbol: %r' % args
                        raise ConnectionClosed(msg)
                    recvd.extend(nxt)
                    find_offset_start = -len(nxt) - len_delimiter + 1
            except socket.timeout:
                self.rbuf = bytes(recvd)
                msg = 'read %s bytes without finding delimiter: %r' % (len(recvd), delimiter)
                raise Timeout(timeout, msg)
            except Exception:
                self.rbuf = bytes(recvd)
                raise
            (val, self.rbuf) = (bytes(recvd[:offset]), bytes(recvd[rbuf_offset:]))
        return val

    def recv_size(self, size, timeout=_UNSET):
        if False:
            while True:
                i = 10
        'Read off of the internal buffer, then off the socket, until\n        *size* bytes have been read.\n\n        Args:\n            size (int): number of bytes to read before returning.\n            timeout (float): The timeout for this operation. Can be 0 for\n                nonblocking and None for no timeout. Defaults to the value\n                set in the constructor of BufferedSocket.\n\n        If the appropriate number of bytes cannot be fetched from the\n        buffer and socket before *timeout* expires, then a\n        :exc:`Timeout` will be raised. If the connection is closed, a\n        :exc:`ConnectionClosed` will be raised.\n        '
        with self._recv_lock:
            if timeout is _UNSET:
                timeout = self.timeout
            chunks = []
            total_bytes = 0
            try:
                start = time.time()
                self.sock.settimeout(timeout)
                nxt = self.rbuf or self.sock.recv(self._recvsize)
                while nxt:
                    total_bytes += len(nxt)
                    if total_bytes >= size:
                        break
                    chunks.append(nxt)
                    if timeout:
                        cur_timeout = timeout - (time.time() - start)
                        if cur_timeout <= 0.0:
                            raise socket.timeout()
                        self.sock.settimeout(cur_timeout)
                    nxt = self.sock.recv(self._recvsize)
                else:
                    msg = 'connection closed after reading %s of %s requested bytes' % (total_bytes, size)
                    raise ConnectionClosed(msg)
            except socket.timeout:
                self.rbuf = b''.join(chunks)
                msg = 'read %s of %s bytes' % (total_bytes, size)
                raise Timeout(timeout, msg)
            except Exception:
                self.rbuf = b''.join(chunks)
                raise
            extra_bytes = total_bytes - size
            if extra_bytes:
                (last, self.rbuf) = (nxt[:-extra_bytes], nxt[-extra_bytes:])
            else:
                (last, self.rbuf) = (nxt, b'')
            chunks.append(last)
        return b''.join(chunks)

    def send(self, data, flags=0, timeout=_UNSET):
        if False:
            for i in range(10):
                print('nop')
        'Send the contents of the internal send buffer, as well as *data*,\n        to the receiving end of the connection. Returns the total\n        number of bytes sent. If no exception is raised, all of *data* was\n        sent and the internal send buffer is empty.\n\n        Args:\n            data (bytes): The bytes to send.\n            flags (int): Kept for API compatibility with sockets. Only\n                the default 0 is valid.\n            timeout (float): The timeout for this operation. Can be 0 for\n                nonblocking and None for no timeout. Defaults to the value\n                set in the constructor of BufferedSocket.\n\n        Will raise :exc:`Timeout` if the send operation fails to\n        complete before *timeout*. In the event of an exception, use\n        :meth:`BufferedSocket.getsendbuffer` to see which data was\n        unsent.\n        '
        with self._send_lock:
            if timeout is _UNSET:
                timeout = self.timeout
            if flags:
                raise ValueError('non-zero flags not supported')
            sbuf = self.sbuf
            sbuf.append(data)
            if len(sbuf) > 1:
                sbuf[:] = [b''.join([s for s in sbuf if s])]
            self.sock.settimeout(timeout)
            (start, total_sent) = (time.time(), 0)
            try:
                while sbuf[0]:
                    sent = self.sock.send(sbuf[0])
                    total_sent += sent
                    sbuf[0] = sbuf[0][sent:]
                    if timeout:
                        cur_timeout = timeout - (time.time() - start)
                        if cur_timeout <= 0.0:
                            raise socket.timeout()
                        self.sock.settimeout(cur_timeout)
            except socket.timeout:
                raise Timeout(timeout, '%s bytes unsent' % len(sbuf[0]))
        return total_sent

    def sendall(self, data, flags=0, timeout=_UNSET):
        if False:
            i = 10
            return i + 15
        'A passthrough to :meth:`~BufferedSocket.send`, retained for\n        parallelism to the :class:`socket.socket` API.\n        '
        return self.send(data, flags, timeout)

    def flush(self):
        if False:
            return 10
        'Send the contents of the internal send buffer.'
        with self._send_lock:
            self.send(b'')
        return

    def buffer(self, data):
        if False:
            i = 10
            return i + 15
        'Buffer *data* bytes for the next send operation.'
        with self._send_lock:
            self.sbuf.append(data)
        return

    def getsockname(self):
        if False:
            print('Hello World!')
        "Convenience function to return the wrapped socket's own address.\n        See :meth:`socket.getsockname` for more details.\n        "
        return self.sock.getsockname()

    def getpeername(self):
        if False:
            return 10
        'Convenience function to return the remote address to which the\n        wrapped socket is connected.  See :meth:`socket.getpeername`\n        for more details.\n        '
        return self.sock.getpeername()

    def getsockopt(self, level, optname, buflen=None):
        if False:
            print('Hello World!')
        "Convenience function passing through to the wrapped socket's\n        :meth:`socket.getsockopt`.\n        "
        args = (level, optname)
        if buflen is not None:
            args += (buflen,)
        return self.sock.getsockopt(*args)

    def setsockopt(self, level, optname, value):
        if False:
            print('Hello World!')
        "Convenience function passing through to the wrapped socket's\n        :meth:`socket.setsockopt`.\n        "
        return self.sock.setsockopt(level, optname, value)

    @property
    def type(self):
        if False:
            return 10
        "A passthrough to the wrapped socket's type. Valid usages should\n        only ever see :data:`socket.SOCK_STREAM`.\n        "
        return self.sock.type

    @property
    def family(self):
        if False:
            for i in range(10):
                print('nop')
        "A passthrough to the wrapped socket's family. BufferedSocket\n        supports all widely-used families, so this read-only attribute\n        can be one of :data:`socket.AF_INET` for IP,\n        :data:`socket.AF_INET6` for IPv6, and :data:`socket.AF_UNIX`\n        for UDS.\n        "
        return self.sock.family

    @property
    def proto(self):
        if False:
            for i in range(10):
                print('nop')
        'A passthrough to the wrapped socket\'s protocol. The ``proto``\n        attribute is very rarely used, so it\'s always 0, meaning "the\n        default" protocol. Pretty much all the practical information\n        is in :attr:`~BufferedSocket.type` and\n        :attr:`~BufferedSocket.family`, so you can go back to never\n        thinking about this.\n        '
        return self.sock.proto

    def fileno(self):
        if False:
            for i in range(10):
                print('nop')
        "Returns the file descriptor of the wrapped socket. -1 if it has\n        been closed on this end.\n\n        Note that this makes the BufferedSocket selectable, i.e.,\n        usable for operating system event loops without any external\n        libraries. Keep in mind that the operating system cannot know\n        about data in BufferedSocket's internal buffer. Exercise\n        discipline with calling ``recv*`` functions.\n        "
        return self.sock.fileno()

    def close(self):
        if False:
            return 10
        'Closes the wrapped socket, and empties the internal buffers. The\n        send buffer is not flushed automatically, so if you have been\n        calling :meth:`~BufferedSocket.buffer`, be sure to call\n        :meth:`~BufferedSocket.flush` before calling this\n        method. After calling this method, future socket operations\n        will raise :exc:`socket.error`.\n        '
        with self._recv_lock:
            with self._send_lock:
                self.rbuf = b''
                self.rbuf_unconsumed = self.rbuf
                self.sbuf[:] = []
                self.sock.close()
        return

    def shutdown(self, how):
        if False:
            for i in range(10):
                print('nop')
        "Convenience method which passes through to the wrapped socket's\n        :meth:`~socket.shutdown`. Semantics vary by platform, so no\n        special internal handling is done with the buffers. This\n        method exists to facilitate the most common usage, wherein a\n        full ``shutdown`` is followed by a\n        :meth:`~BufferedSocket.close`. Developers requiring more\n        support, please open `an issue`_.\n\n        .. _an issue: https://github.com/mahmoud/boltons/issues\n        "
        with self._recv_lock:
            with self._send_lock:
                self.sock.shutdown(how)
        return

class Error(socket.error):
    """A subclass of :exc:`socket.error` from which all other
    ``socketutils`` exceptions inherit.

    When using :class:`BufferedSocket` and other ``socketutils``
    types, generally you want to catch one of the specific exception
    types below, or :exc:`socket.error`.
    """
    pass

class ConnectionClosed(Error):
    """Raised when receiving and the connection is unexpectedly closed
    from the sending end. Raised from :class:`BufferedSocket`'s
    :meth:`~BufferedSocket.peek`, :meth:`~BufferedSocket.recv_until`,
    and :meth:`~BufferedSocket.recv_size`, and never from its
    :meth:`~BufferedSocket.recv` or
    :meth:`~BufferedSocket.recv_close`.
    """
    pass

class MessageTooLong(Error):
    """Raised from :meth:`BufferedSocket.recv_until` and
    :meth:`BufferedSocket.recv_closed` when more than *maxsize* bytes are
    read without encountering the delimiter or a closed connection,
    respectively.
    """

    def __init__(self, bytes_read=None, delimiter=None):
        if False:
            i = 10
            return i + 15
        msg = 'message exceeded maximum size'
        if bytes_read is not None:
            msg += '. %s bytes read' % (bytes_read,)
        if delimiter is not None:
            msg += '. Delimiter not found: %r' % (delimiter,)
        super(MessageTooLong, self).__init__(msg)

class Timeout(socket.timeout, Error):
    """Inheriting from :exc:`socket.timeout`, Timeout is used to indicate
    when a socket operation did not complete within the time
    specified. Raised from any of :class:`BufferedSocket`'s ``recv``
    methods.
    """

    def __init__(self, timeout, extra=''):
        if False:
            print('Hello World!')
        msg = 'socket operation timed out'
        if timeout is not None:
            msg += ' after %sms.' % (timeout * 1000)
        if extra:
            msg += ' ' + extra
        super(Timeout, self).__init__(msg)

class NetstringSocket(object):
    """
    Reads and writes using the netstring protocol.

    More info: https://en.wikipedia.org/wiki/Netstring
    Even more info: http://cr.yp.to/proto/netstrings.txt
    """

    def __init__(self, sock, timeout=DEFAULT_TIMEOUT, maxsize=DEFAULT_MAXSIZE):
        if False:
            print('Hello World!')
        self.bsock = BufferedSocket(sock)
        self.timeout = timeout
        self.maxsize = maxsize
        self._msgsize_maxsize = len(str(maxsize)) + 1

    def fileno(self):
        if False:
            print('Hello World!')
        return self.bsock.fileno()

    def settimeout(self, timeout):
        if False:
            print('Hello World!')
        self.timeout = timeout

    def setmaxsize(self, maxsize):
        if False:
            print('Hello World!')
        self.maxsize = maxsize
        self._msgsize_maxsize = self._calc_msgsize_maxsize(maxsize)

    def _calc_msgsize_maxsize(self, maxsize):
        if False:
            return 10
        return len(str(maxsize)) + 1

    def read_ns(self, timeout=_UNSET, maxsize=_UNSET):
        if False:
            for i in range(10):
                print('nop')
        if timeout is _UNSET:
            timeout = self.timeout
        if maxsize is _UNSET:
            maxsize = self.maxsize
            msgsize_maxsize = self._msgsize_maxsize
        else:
            msgsize_maxsize = self._calc_msgsize_maxsize(maxsize)
        size_prefix = self.bsock.recv_until(b':', timeout=timeout, maxsize=msgsize_maxsize)
        try:
            size = int(size_prefix)
        except ValueError:
            raise NetstringInvalidSize('netstring message size must be valid integer, not %r' % size_prefix)
        if size > maxsize:
            raise NetstringMessageTooLong(size, maxsize)
        payload = self.bsock.recv_size(size)
        if self.bsock.recv(1) != b',':
            raise NetstringProtocolError("expected trailing ',' after message")
        return payload

    def write_ns(self, payload):
        if False:
            for i in range(10):
                print('nop')
        size = len(payload)
        if size > self.maxsize:
            raise NetstringMessageTooLong(size, self.maxsize)
        data = str(size).encode('ascii') + b':' + payload + b','
        self.bsock.send(data)

class NetstringProtocolError(Error):
    """Base class for all of socketutils' Netstring exception types."""
    pass

class NetstringInvalidSize(NetstringProtocolError):
    """NetstringInvalidSize is raised when the ``:``-delimited size prefix
    of the message does not contain a valid integer.

    Message showing valid size::

      5:hello,

    Here the ``5`` is the size. Anything in this prefix position that
    is not parsable as a Python integer (i.e., :class:`int`) will raise
    this exception.
    """

    def __init__(self, msg):
        if False:
            print('Hello World!')
        super(NetstringInvalidSize, self).__init__(msg)

class NetstringMessageTooLong(NetstringProtocolError):
    """NetstringMessageTooLong is raised when the size prefix contains a
    valid integer, but that integer is larger than the
    :class:`NetstringSocket`'s configured *maxsize*.

    When this exception is raised, it's recommended to simply close
    the connection instead of trying to recover.
    """

    def __init__(self, size, maxsize):
        if False:
            print('Hello World!')
        msg = 'netstring message length exceeds configured maxsize: %s > %s' % (size, maxsize)
        super(NetstringMessageTooLong, self).__init__(msg)
'\nattrs worth adding/passing through:\n\n\nproperties: type, proto\n\nFor its main functionality, BufferedSocket can wrap any object that\nhas the following methods:\n\n  - gettimeout()\n  - settimeout()\n  - recv(size)\n  - send(data)\n\nThe following methods are passed through:\n\n...\n\n'