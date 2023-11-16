"""
Python 3 socket module.
"""
from __future__ import absolute_import
import io
import os
from gevent import _socketcommon
from gevent._util import copy_globals
from gevent._compat import PYPY
import _socket
from os import dup
copy_globals(_socketcommon, globals(), names_to_ignore=_socketcommon.__extensions__, dunder_names_to_keep=())
__socket__ = _socketcommon.__socket__
__implements__ = _socketcommon._implements
__extensions__ = _socketcommon.__extensions__
__imports__ = _socketcommon.__imports__
__dns__ = _socketcommon.__dns__
SocketIO = __socket__.SocketIO

class _closedsocket(object):
    __slots__ = ('family', 'type', 'proto', 'orig_fileno', 'description')

    def __init__(self, family, type, proto, orig_fileno, description):
        if False:
            for i in range(10):
                print('nop')
        self.family = family
        self.type = type
        self.proto = proto
        self.orig_fileno = orig_fileno
        self.description = description

    def fileno(self):
        if False:
            for i in range(10):
                print('nop')
        return -1

    def close(self):
        if False:
            return 10
        'No-op'
    detach = fileno

    def _dummy(*args, **kwargs):
        if False:
            while True:
                i = 10
        raise OSError(EBADF, 'Bad file descriptor')
    send = recv = recv_into = sendto = recvfrom = recvfrom_into = _dummy
    getsockname = _dummy

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        return False
    __getattr__ = _dummy

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<socket object [closed proxy at 0x%x fd=%s %s]>' % (id(self), self.orig_fileno, self.description)

class _wrefsocket(_socket.socket):
    __slots__ = ('__weakref__',)
    if PYPY:
        timeout = property(lambda s: s.gettimeout(), lambda s, nv: s.settimeout(nv))

class socket(_socketcommon.SocketMixin):
    """
    gevent `socket.socket <https://docs.python.org/3/library/socket.html#socket-objects>`_
    for Python 3.

    This object should have the same API as the standard library socket linked to above. Not all
    methods are specifically documented here; when they are they may point out a difference
    to be aware of or may document a method the standard library does not.
    """
    _gevent_sock_class = _wrefsocket
    __slots__ = ('_io_refs', '_closed')

    def __init__(self, family=-1, type=-1, proto=-1, fileno=None):
        if False:
            print('Hello World!')
        super().__init__()
        self._closed = False
        if fileno is None:
            if family == -1:
                family = AddressFamily.AF_INET
            if type == -1:
                type = SOCK_STREAM
            if proto == -1:
                proto = 0
        self._sock = self._gevent_sock_class(family, type, proto, fileno)
        self.timeout = None
        self._io_refs = 0
        _socket.socket.setblocking(self._sock, False)
        fileno = _socket.socket.fileno(self._sock)
        self.hub = get_hub()
        io_class = self.hub.loop.io
        self._read_event = io_class(fileno, 1)
        self._write_event = io_class(fileno, 2)
        self.timeout = _socket.getdefaulttimeout()

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._sock, name)

    def _accept(self):
        if False:
            print('Hello World!')
        return self._sock._accept()
    if hasattr(_socket, 'SOCK_NONBLOCK'):

        @property
        def type(self):
            if False:
                i = 10
                return i + 15
            if self.timeout != 0.0:
                return self._sock.type & ~_socket.SOCK_NONBLOCK
            return self._sock.type

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        if not self._closed:
            self.close()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        'Wrap __repr__() to reveal the real class name.'
        try:
            s = repr(self._sock)
        except Exception as ex:
            s = '<socket [%r]>' % ex
        if s.startswith('<socket object'):
            s = '<%s.%s%s at 0x%x%s%s' % (self.__class__.__module__, self.__class__.__name__, getattr(self, '_closed', False) and ' [closed]' or '', id(self), self._extra_repr(), s[7:])
        return s

    def _extra_repr(self):
        if False:
            while True:
                i = 10
        return ''

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        raise TypeError('Cannot serialize socket object')

    def dup(self):
        if False:
            while True:
                i = 10
        'dup() -> socket object\n\n        Return a new socket object connected to the same system resource.\n        '
        fd = dup(self.fileno())
        sock = self.__class__(self.family, self.type, self.proto, fileno=fd)
        sock.settimeout(self.gettimeout())
        return sock

    def accept(self):
        if False:
            print('Hello World!')
        'accept() -> (socket object, address info)\n\n        Wait for an incoming connection.  Return a new socket\n        representing the connection, and the address of the client.\n        For IP sockets, the address info is a pair (hostaddr, port).\n        '
        while True:
            try:
                (fd, addr) = self._accept()
                break
            except BlockingIOError:
                if self.timeout == 0.0:
                    raise
            self._wait(self._read_event)
        sock = socket(self.family, self.type, self.proto, fileno=fd)
        if getdefaulttimeout() is None and self.gettimeout():
            sock.setblocking(True)
        return (sock, addr)

    def makefile(self, mode='r', buffering=None, *, encoding=None, errors=None, newline=None):
        if False:
            print('Hello World!')
        "Return an I/O stream connected to the socket\n\n        The arguments are as for io.open() after the filename,\n        except the only mode characters supported are 'r', 'w' and 'b'.\n        The semantics are similar too.\n        "
        for c in mode:
            if c not in {'r', 'w', 'b'}:
                raise ValueError('invalid mode %r (only r, w, b allowed)')
        writing = 'w' in mode
        reading = 'r' in mode or not writing
        assert reading or writing
        binary = 'b' in mode
        rawmode = ''
        if reading:
            rawmode += 'r'
        if writing:
            rawmode += 'w'
        raw = SocketIO(self, rawmode)
        self._io_refs += 1
        if buffering is None:
            buffering = -1
        if buffering < 0:
            buffering = io.DEFAULT_BUFFER_SIZE
        if buffering == 0:
            if not binary:
                raise ValueError('unbuffered streams must be binary')
            return raw
        if reading and writing:
            buffer = io.BufferedRWPair(raw, raw, buffering)
        elif reading:
            buffer = io.BufferedReader(raw, buffering)
        else:
            assert writing
            buffer = io.BufferedWriter(raw, buffering)
        if binary:
            return buffer
        text = io.TextIOWrapper(buffer, encoding, errors, newline)
        text.mode = mode
        return text

    def _decref_socketios(self):
        if False:
            while True:
                i = 10
        if self._io_refs > 0:
            self._io_refs -= 1
        if self._closed:
            self.close()

    def _drop_ref_on_close(self, sock):
        if False:
            print('Hello World!')
        scheduled_new = self.hub.loop.closing_fd(sock.fileno())
        if scheduled_new:
            self.hub.loop.run_callback(sock.close)
        else:
            sock.close()

    def _detach_socket(self, reason):
        if False:
            i = 10
            return i + 15
        if not self._sock:
            return
        sock = self._sock
        family = -1
        type = -1
        proto = -1
        fileno = None
        try:
            family = sock.family
            type = sock.type
            proto = sock.proto
            fileno = sock.fileno()
        except OSError:
            pass
        self._drop_events_and_close(closefd=reason == 'closed')
        self._sock = _closedsocket(family, type, proto, fileno, reason)

    def _real_close(self, _ss=_socket.socket):
        if False:
            return 10
        if not self._sock:
            return
        self._detach_socket('closed')

    def close(self):
        if False:
            return 10
        self._closed = True
        if self._io_refs <= 0:
            self._real_close()

    @property
    def closed(self):
        if False:
            print('Hello World!')
        return self._closed

    def detach(self):
        if False:
            i = 10
            return i + 15
        '\n        detach() -> file descriptor\n\n        Close the socket object without closing the underlying file\n        descriptor. The object cannot be used after this call; when the\n        real file descriptor is closed, the number that was previously\n        used here may be reused. The fileno() method, after this call,\n        will return an invalid socket id.\n\n        The previous descriptor is returned.\n\n        .. versionchanged:: 1.5\n\n           Also immediately drop any native event loop resources.\n        '
        self._closed = True
        sock = self._sock
        self._detach_socket('detached')
        return sock.detach()
    if hasattr(_socket.socket, 'recvmsg'):

        def recvmsg(self, *args):
            if False:
                return 10
            while True:
                try:
                    return self._sock.recvmsg(*args)
                except error as ex:
                    if ex.args[0] != EWOULDBLOCK or self.timeout == 0.0:
                        raise
                self._wait(self._read_event)
    if hasattr(_socket.socket, 'recvmsg_into'):

        def recvmsg_into(self, buffers, *args):
            if False:
                print('Hello World!')
            while True:
                try:
                    if args:
                        return self._sock.recvmsg_into(buffers, *args)
                    return self._sock.recvmsg_into(buffers)
                except error as ex:
                    if ex.args[0] != EWOULDBLOCK or self.timeout == 0.0:
                        raise
                self._wait(self._read_event)
    if hasattr(_socket.socket, 'sendmsg'):

        def sendmsg(self, buffers, ancdata=(), flags=0, address=None):
            if False:
                i = 10
                return i + 15
            try:
                return self._sock.sendmsg(buffers, ancdata, flags, address)
            except error as ex:
                if flags & getattr(_socket, 'MSG_DONTWAIT', 0):
                    raise
                if ex.args[0] != EWOULDBLOCK or self.timeout == 0.0:
                    raise
                self._wait(self._write_event)
                try:
                    return self._sock.sendmsg(buffers, ancdata, flags, address)
                except error as ex2:
                    if ex2.args[0] == EWOULDBLOCK:
                        return 0
                    raise

    def _sendfile_use_sendfile(self, file, offset=0, count=None):
        if False:
            while True:
                i = 10
        raise __socket__._GiveupOnSendfile()

    def _sendfile_use_send(self, file, offset=0, count=None):
        if False:
            print('Hello World!')
        self._check_sendfile_params(file, offset, count)
        if self.gettimeout() == 0:
            raise ValueError('non-blocking sockets are not supported')
        if offset:
            file.seek(offset)
        blocksize = min(count, 8192) if count else 8192
        total_sent = 0
        file_read = file.read
        sock_send = self.send
        try:
            while True:
                if count:
                    blocksize = min(count - total_sent, blocksize)
                    if blocksize <= 0:
                        break
                data = memoryview(file_read(blocksize))
                if not data:
                    break
                while True:
                    try:
                        sent = sock_send(data)
                    except BlockingIOError:
                        continue
                    else:
                        total_sent += sent
                        if sent < len(data):
                            data = data[sent:]
                        else:
                            break
            return total_sent
        finally:
            if total_sent > 0 and hasattr(file, 'seek'):
                file.seek(offset + total_sent)

    def _check_sendfile_params(self, file, offset, count):
        if False:
            print('Hello World!')
        if 'b' not in getattr(file, 'mode', 'b'):
            raise ValueError('file should be opened in binary mode')
        if not self.type & SOCK_STREAM:
            raise ValueError('only SOCK_STREAM type sockets are supported')
        if count is not None:
            if not isinstance(count, int):
                raise TypeError('count must be a positive integer (got {!r})'.format(count))
            if count <= 0:
                raise ValueError('count must be a positive integer (got {!r})'.format(count))

    def sendfile(self, file, offset=0, count=None):
        if False:
            return 10
        'sendfile(file[, offset[, count]]) -> sent\n\n        Send a file until EOF is reached by using high-performance\n        os.sendfile() and return the total number of bytes which\n        were sent.\n        *file* must be a regular file object opened in binary mode.\n        If os.sendfile() is not available (e.g. Windows) or file is\n        not a regular file socket.send() will be used instead.\n        *offset* tells from where to start reading the file.\n        If specified, *count* is the total number of bytes to transmit\n        as opposed to sending the file until EOF is reached.\n        File position is updated on return or also in case of error in\n        which case file.tell() can be used to figure out the number of\n        bytes which were sent.\n        The socket must be of SOCK_STREAM type.\n        Non-blocking sockets are not supported.\n\n        .. versionadded:: 1.1rc4\n           Added in Python 3.5, but available under all Python 3 versions in\n           gevent.\n        '
        return self._sendfile_use_send(file, offset, count)
    if os.name == 'nt':

        def get_inheritable(self):
            if False:
                while True:
                    i = 10
            return os.get_handle_inheritable(self.fileno())

        def set_inheritable(self, inheritable):
            if False:
                print('Hello World!')
            os.set_handle_inheritable(self.fileno(), inheritable)
    else:

        def get_inheritable(self):
            if False:
                for i in range(10):
                    print('nop')
            return os.get_inheritable(self.fileno())

        def set_inheritable(self, inheritable):
            if False:
                print('Hello World!')
            os.set_inheritable(self.fileno(), inheritable)
    get_inheritable.__doc__ = 'Get the inheritable flag of the socket'
    set_inheritable.__doc__ = 'Set the inheritable flag of the socket'
SocketType = socket

def fromfd(fd, family, type, proto=0):
    if False:
        i = 10
        return i + 15
    ' fromfd(fd, family, type[, proto]) -> socket object\n\n    Create a socket object from a duplicate of the given file\n    descriptor.  The remaining arguments are the same as for socket().\n    '
    nfd = dup(fd)
    return socket(family, type, proto, nfd)
if hasattr(_socket.socket, 'share'):

    def fromshare(info):
        if False:
            for i in range(10):
                print('nop')
        ' fromshare(info) -> socket object\n\n        Create a socket object from a the bytes object returned by\n        socket.share(pid).\n        '
        return socket(0, 0, 0, info)
    __implements__.append('fromshare')
if hasattr(_socket, 'socketpair'):

    def socketpair(family=None, type=SOCK_STREAM, proto=0):
        if False:
            for i in range(10):
                print('nop')
        'socketpair([family[, type[, proto]]]) -> (socket object, socket object)\n\n        Create a pair of socket objects from the sockets returned by the platform\n        socketpair() function.\n        The arguments are the same as for socket() except the default family is\n        AF_UNIX if defined on the platform; otherwise, the default is AF_INET.\n\n        .. versionchanged:: 1.2\n           All Python 3 versions on Windows supply this function (natively\n           supplied by Python 3.5 and above).\n        '
        if family is None:
            try:
                family = AF_UNIX
            except NameError:
                family = AF_INET
        (a, b) = _socket.socketpair(family, type, proto)
        a = socket(family, type, proto, a.detach())
        b = socket(family, type, proto, b.detach())
        return (a, b)
else:
    _LOCALHOST = '127.0.0.1'
    _LOCALHOST_V6 = '::1'

    def socketpair(family=AF_INET, type=SOCK_STREAM, proto=0):
        if False:
            return 10
        if family == AF_INET:
            host = _LOCALHOST
        elif family == AF_INET6:
            host = _LOCALHOST_V6
        else:
            raise ValueError('Only AF_INET and AF_INET6 socket address families are supported')
        if type != SOCK_STREAM:
            raise ValueError('Only SOCK_STREAM socket type is supported')
        if proto != 0:
            raise ValueError('Only protocol zero is supported')
        lsock = socket(family, type, proto)
        try:
            lsock.bind((host, 0))
            lsock.listen()
            (addr, port) = lsock.getsockname()[:2]
            csock = socket(family, type, proto)
            try:
                csock.setblocking(False)
                try:
                    csock.connect((addr, port))
                except (BlockingIOError, InterruptedError):
                    pass
                csock.setblocking(True)
                (ssock, _) = lsock.accept()
            except:
                csock.close()
                raise
        finally:
            lsock.close()
        return (ssock, csock)
__all__ = __implements__ + __extensions__ + __imports__
__version_specific__ = ('close', 'TCP_KEEPALIVE', 'TCP_KEEPCNT')
for _x in __version_specific__:
    if hasattr(__socket__, _x):
        vars()[_x] = getattr(__socket__, _x)
        if _x not in __all__:
            __all__.append(_x)
del _x