from __future__ import absolute_import
_implements = ['create_connection', 'socket', 'SocketType', 'fromfd', 'socketpair']
__dns__ = ['getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'gethostbyaddr', 'getnameinfo', 'getfqdn']
_implements += __dns__
__extensions__ = ['cancel_wait', 'wait_read', 'wait_write', 'wait_readwrite']
__imports__ = ['error', 'gaierror', 'herror', 'htonl', 'htons', 'ntohl', 'ntohs', 'inet_aton', 'inet_ntoa', 'inet_pton', 'inet_ntop', 'timeout', 'gethostname', 'getprotobyname', 'getservbyname', 'getservbyport', 'getdefaulttimeout', 'setdefaulttimeout', 'errorTab', 'AddressFamily', 'SocketKind', 'CMSG_LEN', 'CMSG_SPACE', 'dup', 'if_indextoname', 'if_nameindex', 'if_nametoindex', 'sethostname', 'create_server', 'has_dualstack_ipv6']
import time
from gevent._hub_local import get_hub_noargs as get_hub
from gevent._compat import string_types, integer_types
from gevent._compat import PY39
from gevent._compat import WIN as is_windows
from gevent._compat import OSX as is_macos
from gevent._compat import exc_clear
from gevent._util import copy_globals
from gevent._greenlet_primitives import get_memory as _get_memory
from gevent._hub_primitives import wait_on_socket as _wait_on_socket
from gevent.timeout import Timeout
if PY39:
    __imports__.extend(['recv_fds', 'send_fds'])
if is_windows:
    from errno import WSAEINVAL as EINVAL
    from errno import WSAEWOULDBLOCK as EWOULDBLOCK
    from errno import WSAEINPROGRESS as EINPROGRESS
    from errno import WSAEALREADY as EALREADY
    from errno import WSAEISCONN as EISCONN
    from gevent.win32util import formatError as strerror
    EAGAIN = EWOULDBLOCK
else:
    from errno import EINVAL
    from errno import EWOULDBLOCK
    from errno import EINPROGRESS
    from errno import EALREADY
    from errno import EAGAIN
    from errno import EISCONN
    from os import strerror
try:
    from errno import EBADF
except ImportError:
    EBADF = 9
try:
    from errno import EHOSTUNREACH
except ImportError:
    EHOSTUNREACH = -1
try:
    from errno import ECONNREFUSED
except ImportError:
    ECONNREFUSED = -1
GSENDAGAIN = (EWOULDBLOCK,)
if is_macos:
    from errno import EPROTOTYPE
    GSENDAGAIN += (EPROTOTYPE,)
import _socket
_realsocket = _socket.socket
import socket as __socket__
try:
    import backports.socketpair
except ImportError:
    pass
_SocketError = __socket__.error
_name = _value = None
__imports__ = copy_globals(__socket__, globals(), only_names=__imports__, ignore_missing_names=True)
for _name in __socket__.__all__:
    _value = getattr(__socket__, _name)
    if isinstance(_value, (integer_types, string_types)):
        globals()[_name] = _value
        __imports__.append(_name)
del _name, _value
_timeout_error = timeout
from gevent import _hub_primitives
_hub_primitives.set_default_timeout_error(_timeout_error)
wait = _hub_primitives.wait_on_watcher
wait_read = _hub_primitives.wait_read
wait_write = _hub_primitives.wait_write
wait_readwrite = _hub_primitives.wait_readwrite

class cancel_wait_ex(error):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(cancel_wait_ex, self).__init__(EBADF, 'File descriptor was closed in another greenlet')

def cancel_wait(watcher, error=cancel_wait_ex):
    if False:
        i = 10
        return i + 15
    'See :meth:`gevent.hub.Hub.cancel_wait`'
    get_hub().cancel_wait(watcher, error)

def gethostbyname(hostname):
    if False:
        i = 10
        return i + 15
    "\n    gethostbyname(host) -> address\n\n    Return the IP address (a string of the form '255.255.255.255') for a host.\n\n    .. seealso:: :doc:`/dns`\n    "
    return get_hub().resolver.gethostbyname(hostname)

def gethostbyname_ex(hostname):
    if False:
        i = 10
        return i + 15
    '\n    gethostbyname_ex(host) -> (name, aliaslist, addresslist)\n\n    Return the true host name, a list of aliases, and a list of IP addresses,\n    for a host.  The host argument is a string giving a host name or IP number.\n    Resolve host and port into list of address info entries.\n\n    .. seealso:: :doc:`/dns`\n    '
    return get_hub().resolver.gethostbyname_ex(hostname)

def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if False:
        for i in range(10):
            print('nop')
    "\n    Resolve host and port into list of address info entries.\n\n    Translate the host/port argument into a sequence of 5-tuples that contain\n    all the necessary arguments for creating a socket connected to that service.\n    host is a domain name, a string representation of an IPv4/v6 address or\n    None. port is a string service name such as 'http', a numeric port number or\n    None. By passing None as the value of host and port, you can pass NULL to\n    the underlying C API.\n\n    The family, type and proto arguments can be optionally specified in order to\n    narrow the list of addresses returned. Passing zero as a value for each of\n    these arguments selects the full range of results.\n\n    .. seealso:: :doc:`/dns`\n    "
    addrlist = get_hub().resolver.getaddrinfo(host, port, family, type, proto, flags)
    result = [(_intenum_converter(af, AddressFamily), _intenum_converter(socktype, SocketKind), proto, canonname, sa) for (af, socktype, proto, canonname, sa) in addrlist]
    return result

def _intenum_converter(value, enum_klass):
    if False:
        while True:
            i = 10
    try:
        return enum_klass(value)
    except ValueError:
        return value

def gethostbyaddr(ip_address):
    if False:
        i = 10
        return i + 15
    '\n    gethostbyaddr(ip_address) -> (name, aliaslist, addresslist)\n\n    Return the true host name, a list of aliases, and a list of IP addresses,\n    for a host.  The host argument is a string giving a host name or IP number.\n\n    .. seealso:: :doc:`/dns`\n    '
    return get_hub().resolver.gethostbyaddr(ip_address)

def getnameinfo(sockaddr, flags):
    if False:
        i = 10
        return i + 15
    '\n    getnameinfo(sockaddr, flags) -> (host, port)\n\n    Get host and port for a sockaddr.\n\n    .. seealso:: :doc:`/dns`\n    '
    return get_hub().resolver.getnameinfo(sockaddr, flags)

def getfqdn(name=''):
    if False:
        return 10
    "Get fully qualified domain name from name.\n\n    An empty argument is interpreted as meaning the local host.\n\n    First the hostname returned by gethostbyaddr() is checked, then\n    possibly existing aliases. In case no FQDN is available, hostname\n    from gethostname() is returned.\n\n    .. versionchanged:: 23.7.0\n       The IPv6 generic address '::' now returns the result of\n       ``gethostname``, like the IPv4 address '0.0.0.0'.\n    "
    name = name.strip()
    if not name or name in ('0.0.0.0', '::'):
        name = gethostname()
    try:
        (hostname, aliases, _) = gethostbyaddr(name)
    except error:
        pass
    else:
        aliases.insert(0, hostname)
        for name in aliases:
            if isinstance(name, bytes):
                if b'.' in name:
                    break
            elif '.' in name:
                break
        else:
            name = hostname
    return name

def __send_chunk(socket, data_memory, flags, timeleft, end, timeout=_timeout_error):
    if False:
        for i in range(10):
            print('nop')
    '\n    Send the complete contents of ``data_memory`` before returning.\n    This is the core loop around :meth:`send`.\n\n    :param timeleft: Either ``None`` if there is no timeout involved,\n       or a float indicating the timeout to use.\n    :param end: Either ``None`` if there is no timeout involved, or\n       a float giving the absolute end time.\n    :return: An updated value for ``timeleft`` (or None)\n    :raises timeout: If ``timeleft`` was given and elapsed while\n       sending this chunk.\n    '
    data_sent = 0
    len_data_memory = len(data_memory)
    started_timer = 0
    while data_sent < len_data_memory:
        chunk = data_memory[data_sent:]
        if timeleft is None:
            data_sent += socket.send(chunk, flags)
        elif started_timer and timeleft <= 0:
            raise timeout('timed out')
        else:
            started_timer = 1
            data_sent += socket.send(chunk, flags, timeout=timeleft)
            timeleft = end - time.time()
    return timeleft

def _sendall(socket, data_memory, flags, SOL_SOCKET=__socket__.SOL_SOCKET, SO_SNDBUF=__socket__.SO_SNDBUF):
    if False:
        i = 10
        return i + 15
    '\n    Send the *data_memory* (which should be a memoryview)\n    using the gevent *socket*, performing well on PyPy.\n    '
    len_data_memory = len(data_memory)
    if not len_data_memory:
        return 0
    chunk_size = max(socket.getsockopt(SOL_SOCKET, SO_SNDBUF), 1024 * 1024)
    data_sent = 0
    end = None
    timeleft = None
    if socket.timeout is not None:
        timeleft = socket.timeout
        end = time.time() + timeleft
    while data_sent < len_data_memory:
        chunk_end = min(data_sent + chunk_size, len_data_memory)
        chunk = data_memory[data_sent:chunk_end]
        timeleft = __send_chunk(socket, chunk, flags, timeleft, end)
        data_sent += len(chunk)
_RESOLVABLE_FAMILIES = (__socket__.AF_INET,)
if __socket__.has_ipv6:
    _RESOLVABLE_FAMILIES += (__socket__.AF_INET6,)

def _resolve_addr(sock, address):
    if False:
        while True:
            i = 10
    if sock.family not in _RESOLVABLE_FAMILIES or not isinstance(address, tuple):
        return address
    try:
        if __socket__.inet_pton(sock.family, address[0]):
            return address
    except AttributeError:
        pass
    except _SocketError:
        pass
    (host, port) = address[:2]
    r = getaddrinfo(host, None, sock.family)
    address = r[0][-1]
    if len(address) == 2:
        address = (address[0], port)
    else:
        address = (address[0], port, address[2], address[3])
    return address
timeout_default = object()

class SocketMixin(object):
    __slots__ = ('hub', 'timeout', '_read_event', '_write_event', '_sock', '__weakref__')

    def __init__(self):
        if False:
            return 10
        self._read_event = None
        self._write_event = None
        self._sock = None
        self.hub = None
        self.timeout = None

    def _drop_events_and_close(self, closefd=True, _cancel_wait_ex=cancel_wait_ex):
        if False:
            i = 10
            return i + 15
        hub = self.hub
        read_event = self._read_event
        write_event = self._write_event
        self._read_event = self._write_event = None
        hub.cancel_waits_close_and_then((read_event, write_event), _cancel_wait_ex, self._drop_ref_on_close if closefd else id, self._sock)

    def _drop_ref_on_close(self, sock):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def _get_ref(self):
        if False:
            for i in range(10):
                print('nop')
        return self._read_event.ref or self._write_event.ref

    def _set_ref(self, value):
        if False:
            while True:
                i = 10
        self._read_event.ref = value
        self._write_event.ref = value
    ref = property(_get_ref, _set_ref)
    _wait = _wait_on_socket

    def settimeout(self, howlong):
        if False:
            for i in range(10):
                print('nop')
        if howlong is not None:
            try:
                f = howlong.__float__
            except AttributeError:
                raise TypeError('a float is required', howlong, type(howlong))
            howlong = f()
            if howlong < 0.0:
                raise ValueError('Timeout value out of range')
        SocketMixin.timeout.__set__(self, howlong)

    def gettimeout(self):
        if False:
            i = 10
            return i + 15
        return SocketMixin.timeout.__get__(self, type(self))

    def setblocking(self, flag):
        if False:
            return 10
        if flag:
            self.timeout = None
        else:
            self.timeout = 0.0

    def shutdown(self, how):
        if False:
            while True:
                i = 10
        if how == 0:
            self.hub.cancel_wait(self._read_event, cancel_wait_ex)
        elif how == 1:
            self.hub.cancel_wait(self._write_event, cancel_wait_ex)
        else:
            self.hub.cancel_wait(self._read_event, cancel_wait_ex)
            self.hub.cancel_wait(self._write_event, cancel_wait_ex)
        self._sock.shutdown(how)
    family = property(lambda self: _intenum_converter(self._sock.family, AddressFamily))
    type = property(lambda self: _intenum_converter(self._sock.type, SocketKind))
    proto = property(lambda self: self._sock.proto)

    def fileno(self):
        if False:
            return 10
        return self._sock.fileno()

    def getsockname(self):
        if False:
            while True:
                i = 10
        return self._sock.getsockname()

    def getpeername(self):
        if False:
            print('Hello World!')
        return self._sock.getpeername()

    def bind(self, address):
        if False:
            while True:
                i = 10
        return self._sock.bind(address)

    def listen(self, *args):
        if False:
            while True:
                i = 10
        return self._sock.listen(*args)

    def getsockopt(self, *args):
        if False:
            i = 10
            return i + 15
        return self._sock.getsockopt(*args)

    def setsockopt(self, *args):
        if False:
            i = 10
            return i + 15
        return self._sock.setsockopt(*args)
    if hasattr(__socket__.socket, 'ioctl'):

        def ioctl(self, *args):
            if False:
                for i in range(10):
                    print('nop')
            return self._sock.ioctl(*args)
    if hasattr(__socket__.socket, 'sleeptaskw'):

        def sleeptaskw(self, *args):
            if False:
                for i in range(10):
                    print('nop')
            return self._sock.sleeptaskw(*args)

    def getblocking(self):
        if False:
            print('Hello World!')
        '\n        Returns whether the socket will approximate blocking\n        behaviour.\n\n        .. versionadded:: 1.3a2\n            Added in Python 3.7.\n        '
        return self.timeout != 0.0

    def connect(self, address):
        if False:
            i = 10
            return i + 15
        '\n        Connect to *address*.\n\n        .. versionchanged:: 20.6.0\n            If the host part of the address includes an IPv6 scope ID,\n            it will be used instead of ignored, if the platform supplies\n            :func:`socket.inet_pton`.\n        '
        self._internal_connect(address)

    def connect_ex(self, address):
        if False:
            i = 10
            return i + 15
        '\n        Connect to *address*, returning a result code.\n\n        .. versionchanged:: 23.7.0\n           No longer uses an overridden ``connect`` method on\n           this object. Instead, like the standard library, this method always\n           uses a non-replacable internal connection function.\n        '
        try:
            return self._internal_connect(address) or 0
        except __socket__.timeout:
            return EAGAIN
        except __socket__.gaierror:
            raise
        except _SocketError as ex:
            try:
                err = ex.errno
            except AttributeError:
                err = ex.args[0]
            if err:
                return err
            raise

    def _internal_connect(self, address):
        if False:
            while True:
                i = 10
        if self.timeout == 0.0:
            return self._sock.connect(address)
        address = _resolve_addr(self._sock, address)
        with Timeout._start_new_or_dummy(self.timeout, __socket__.timeout('timed out')):
            while 1:
                err = self.getsockopt(__socket__.SOL_SOCKET, __socket__.SO_ERROR)
                if err:
                    raise _SocketError(err, strerror(err))
                result = self._sock.connect_ex(address)
                if not result or result == EISCONN:
                    break
                if result in (EWOULDBLOCK, EINPROGRESS, EALREADY) or (result == EINVAL and is_windows):
                    self._wait(self._write_event)
                else:
                    if isinstance(address, tuple) and address[0] == 'fe80::1' and (result == EHOSTUNREACH):
                        result = ECONNREFUSED
                    raise _SocketError(result, strerror(result))

    def recv(self, *args):
        if False:
            return 10
        while 1:
            try:
                return self._sock.recv(*args)
            except _SocketError as ex:
                if ex.args[0] != EWOULDBLOCK or self.timeout == 0.0:
                    raise
                exc_clear()
            self._wait(self._read_event)

    def recvfrom(self, *args):
        if False:
            for i in range(10):
                print('nop')
        while 1:
            try:
                return self._sock.recvfrom(*args)
            except _SocketError as ex:
                if ex.args[0] != EWOULDBLOCK or self.timeout == 0.0:
                    raise
                exc_clear()
            self._wait(self._read_event)

    def recvfrom_into(self, *args):
        if False:
            return 10
        while 1:
            try:
                return self._sock.recvfrom_into(*args)
            except _SocketError as ex:
                if ex.args[0] != EWOULDBLOCK or self.timeout == 0.0:
                    raise
                exc_clear()
            self._wait(self._read_event)

    def recv_into(self, *args):
        if False:
            i = 10
            return i + 15
        while 1:
            try:
                return self._sock.recv_into(*args)
            except _SocketError as ex:
                if ex.args[0] != EWOULDBLOCK or self.timeout == 0.0:
                    raise
                exc_clear()
            self._wait(self._read_event)

    def sendall(self, data, flags=0):
        if False:
            for i in range(10):
                print('nop')
        data_memory = _get_memory(data)
        return _sendall(self, data_memory, flags)

    def sendto(self, *args):
        if False:
            return 10
        try:
            return self._sock.sendto(*args)
        except _SocketError as ex:
            if ex.args[0] != EWOULDBLOCK or self.timeout == 0.0:
                raise
            exc_clear()
            self._wait(self._write_event)
            try:
                return self._sock.sendto(*args)
            except _SocketError as ex2:
                if ex2.args[0] == EWOULDBLOCK:
                    exc_clear()
                    return 0
                raise

    def send(self, data, flags=0, timeout=timeout_default):
        if False:
            return 10
        if timeout is timeout_default:
            timeout = self.timeout
        try:
            return self._sock.send(data, flags)
        except _SocketError as ex:
            if ex.args[0] not in GSENDAGAIN or timeout == 0.0:
                raise
            exc_clear()
            self._wait(self._write_event)
            try:
                return self._sock.send(data, flags)
            except _SocketError as ex2:
                if ex2.args[0] == EWOULDBLOCK:
                    exc_clear()
                    return 0
                raise

    @classmethod
    def _fixup_docstrings(cls):
        if False:
            i = 10
            return i + 15
        for (k, v) in vars(cls).items():
            if k.startswith('_'):
                continue
            if not hasattr(v, '__doc__') or v.__doc__:
                continue
            smeth = getattr(__socket__.socket, k, None)
            if not smeth or not smeth.__doc__:
                continue
            try:
                v.__doc__ = smeth.__doc__
            except (AttributeError, TypeError):
                continue
SocketMixin._fixup_docstrings()
del SocketMixin._fixup_docstrings