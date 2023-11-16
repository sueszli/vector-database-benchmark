"""Miscellaneous network utility code."""
import asyncio
import concurrent.futures
import errno
import os
import sys
import socket
import ssl
import stat
from tornado.concurrent import dummy_executor, run_on_executor
from tornado.ioloop import IOLoop
from tornado.util import Configurable, errno_from_exception
from typing import List, Callable, Any, Type, Dict, Union, Tuple, Awaitable, Optional
_client_ssl_defaults = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
_server_ssl_defaults = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
if hasattr(ssl, 'OP_NO_COMPRESSION'):
    _client_ssl_defaults.options |= ssl.OP_NO_COMPRESSION
    _server_ssl_defaults.options |= ssl.OP_NO_COMPRESSION
'foo'.encode('idna')
'foo'.encode('latin1')
_DEFAULT_BACKLOG = 128

def bind_sockets(port: int, address: Optional[str]=None, family: socket.AddressFamily=socket.AF_UNSPEC, backlog: int=_DEFAULT_BACKLOG, flags: Optional[int]=None, reuse_port: bool=False) -> List[socket.socket]:
    if False:
        return 10
    "Creates listening sockets bound to the given port and address.\n\n    Returns a list of socket objects (multiple sockets are returned if\n    the given address maps to multiple IP addresses, which is most common\n    for mixed IPv4 and IPv6 use).\n\n    Address may be either an IP address or hostname.  If it's a hostname,\n    the server will listen on all IP addresses associated with the\n    name.  Address may be an empty string or None to listen on all\n    available interfaces.  Family may be set to either `socket.AF_INET`\n    or `socket.AF_INET6` to restrict to IPv4 or IPv6 addresses, otherwise\n    both will be used if available.\n\n    The ``backlog`` argument has the same meaning as for\n    `socket.listen() <socket.socket.listen>`.\n\n    ``flags`` is a bitmask of AI_* flags to `~socket.getaddrinfo`, like\n    ``socket.AI_PASSIVE | socket.AI_NUMERICHOST``.\n\n    ``reuse_port`` option sets ``SO_REUSEPORT`` option for every socket\n    in the list. If your platform doesn't support this option ValueError will\n    be raised.\n    "
    if reuse_port and (not hasattr(socket, 'SO_REUSEPORT')):
        raise ValueError("the platform doesn't support SO_REUSEPORT")
    sockets = []
    if address == '':
        address = None
    if not socket.has_ipv6 and family == socket.AF_UNSPEC:
        family = socket.AF_INET
    if flags is None:
        flags = socket.AI_PASSIVE
    bound_port = None
    unique_addresses = set()
    for res in sorted(socket.getaddrinfo(address, port, family, socket.SOCK_STREAM, 0, flags), key=lambda x: x[0]):
        if res in unique_addresses:
            continue
        unique_addresses.add(res)
        (af, socktype, proto, canonname, sockaddr) = res
        if sys.platform == 'darwin' and address == 'localhost' and (af == socket.AF_INET6) and (sockaddr[3] != 0):
            continue
        try:
            sock = socket.socket(af, socktype, proto)
        except socket.error as e:
            if errno_from_exception(e) == errno.EAFNOSUPPORT:
                continue
            raise
        if os.name != 'nt':
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except socket.error as e:
                if errno_from_exception(e) != errno.ENOPROTOOPT:
                    raise
        if reuse_port:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        if af == socket.AF_INET6:
            if hasattr(socket, 'IPPROTO_IPV6'):
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        (host, requested_port) = sockaddr[:2]
        if requested_port == 0 and bound_port is not None:
            sockaddr = tuple([host, bound_port] + list(sockaddr[2:]))
        sock.setblocking(False)
        try:
            sock.bind(sockaddr)
        except OSError as e:
            if errno_from_exception(e) == errno.EADDRNOTAVAIL and address == 'localhost' and (sockaddr[0] == '::1'):
                sock.close()
                continue
            else:
                raise
        bound_port = sock.getsockname()[1]
        sock.listen(backlog)
        sockets.append(sock)
    return sockets
if hasattr(socket, 'AF_UNIX'):

    def bind_unix_socket(file: str, mode: int=384, backlog: int=_DEFAULT_BACKLOG) -> socket.socket:
        if False:
            while True:
                i = 10
        'Creates a listening unix socket.\n\n        If a socket with the given name already exists, it will be deleted.\n        If any other file with that name exists, an exception will be\n        raised.\n\n        Returns a socket object (not a list of socket objects like\n        `bind_sockets`)\n        '
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except socket.error as e:
            if errno_from_exception(e) != errno.ENOPROTOOPT:
                raise
        sock.setblocking(False)
        try:
            st = os.stat(file)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISSOCK(st.st_mode):
                os.remove(file)
            else:
                raise ValueError('File %s exists and is not a socket', file)
        sock.bind(file)
        os.chmod(file, mode)
        sock.listen(backlog)
        return sock

def add_accept_handler(sock: socket.socket, callback: Callable[[socket.socket, Any], None]) -> Callable[[], None]:
    if False:
        return 10
    'Adds an `.IOLoop` event handler to accept new connections on ``sock``.\n\n    When a connection is accepted, ``callback(connection, address)`` will\n    be run (``connection`` is a socket object, and ``address`` is the\n    address of the other end of the connection).  Note that this signature\n    is different from the ``callback(fd, events)`` signature used for\n    `.IOLoop` handlers.\n\n    A callable is returned which, when called, will remove the `.IOLoop`\n    event handler and stop processing further incoming connections.\n\n    .. versionchanged:: 5.0\n       The ``io_loop`` argument (deprecated since version 4.1) has been removed.\n\n    .. versionchanged:: 5.0\n       A callable is returned (``None`` was returned before).\n    '
    io_loop = IOLoop.current()
    removed = [False]

    def accept_handler(fd: socket.socket, events: int) -> None:
        if False:
            print('Hello World!')
        for i in range(_DEFAULT_BACKLOG):
            if removed[0]:
                return
            try:
                (connection, address) = sock.accept()
            except BlockingIOError:
                return
            except ConnectionAbortedError:
                continue
            callback(connection, address)

    def remove_handler() -> None:
        if False:
            return 10
        io_loop.remove_handler(sock)
        removed[0] = True
    io_loop.add_handler(sock, accept_handler, IOLoop.READ)
    return remove_handler

def is_valid_ip(ip: str) -> bool:
    if False:
        i = 10
        return i + 15
    'Returns ``True`` if the given string is a well-formed IP address.\n\n    Supports IPv4 and IPv6.\n    '
    if not ip or '\x00' in ip:
        return False
    try:
        res = socket.getaddrinfo(ip, 0, socket.AF_UNSPEC, socket.SOCK_STREAM, 0, socket.AI_NUMERICHOST)
        return bool(res)
    except socket.gaierror as e:
        if e.args[0] == socket.EAI_NONAME:
            return False
        raise
    except UnicodeError:
        return False
    return True

class Resolver(Configurable):
    """Configurable asynchronous DNS resolver interface.

    By default, a blocking implementation is used (which simply calls
    `socket.getaddrinfo`).  An alternative implementation can be
    chosen with the `Resolver.configure <.Configurable.configure>`
    class method::

        Resolver.configure('tornado.netutil.ThreadedResolver')

    The implementations of this interface included with Tornado are

    * `tornado.netutil.DefaultLoopResolver`
    * `tornado.netutil.DefaultExecutorResolver` (deprecated)
    * `tornado.netutil.BlockingResolver` (deprecated)
    * `tornado.netutil.ThreadedResolver` (deprecated)
    * `tornado.netutil.OverrideResolver`
    * `tornado.platform.twisted.TwistedResolver` (deprecated)
    * `tornado.platform.caresresolver.CaresResolver` (deprecated)

    .. versionchanged:: 5.0
       The default implementation has changed from `BlockingResolver` to
       `DefaultExecutorResolver`.

    .. versionchanged:: 6.2
       The default implementation has changed from `DefaultExecutorResolver` to
       `DefaultLoopResolver`.
    """

    @classmethod
    def configurable_base(cls) -> Type['Resolver']:
        if False:
            while True:
                i = 10
        return Resolver

    @classmethod
    def configurable_default(cls) -> Type['Resolver']:
        if False:
            i = 10
            return i + 15
        return DefaultLoopResolver

    def resolve(self, host: str, port: int, family: socket.AddressFamily=socket.AF_UNSPEC) -> Awaitable[List[Tuple[int, Any]]]:
        if False:
            print('Hello World!')
        'Resolves an address.\n\n        The ``host`` argument is a string which may be a hostname or a\n        literal IP address.\n\n        Returns a `.Future` whose result is a list of (family,\n        address) pairs, where address is a tuple suitable to pass to\n        `socket.connect <socket.socket.connect>` (i.e. a ``(host,\n        port)`` pair for IPv4; additional fields may be present for\n        IPv6). If a ``callback`` is passed, it will be run with the\n        result as an argument when it is complete.\n\n        :raises IOError: if the address cannot be resolved.\n\n        .. versionchanged:: 4.4\n           Standardized all implementations to raise `IOError`.\n\n        .. versionchanged:: 6.0 The ``callback`` argument was removed.\n           Use the returned awaitable object instead.\n\n        '
        raise NotImplementedError()

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Closes the `Resolver`, freeing any resources used.\n\n        .. versionadded:: 3.1\n\n        '
        pass

def _resolve_addr(host: str, port: int, family: socket.AddressFamily=socket.AF_UNSPEC) -> List[Tuple[int, Any]]:
    if False:
        for i in range(10):
            print('nop')
    addrinfo = socket.getaddrinfo(host, port, family, socket.SOCK_STREAM)
    results = []
    for (fam, socktype, proto, canonname, address) in addrinfo:
        results.append((fam, address))
    return results

class DefaultExecutorResolver(Resolver):
    """Resolver implementation using `.IOLoop.run_in_executor`.

    .. versionadded:: 5.0

    .. deprecated:: 6.2

       Use `DefaultLoopResolver` instead.
    """

    async def resolve(self, host: str, port: int, family: socket.AddressFamily=socket.AF_UNSPEC) -> List[Tuple[int, Any]]:
        result = await IOLoop.current().run_in_executor(None, _resolve_addr, host, port, family)
        return result

class DefaultLoopResolver(Resolver):
    """Resolver implementation using `asyncio.loop.getaddrinfo`."""

    async def resolve(self, host: str, port: int, family: socket.AddressFamily=socket.AF_UNSPEC) -> List[Tuple[int, Any]]:
        return [(fam, address) for (fam, _, _, _, address) in await asyncio.get_running_loop().getaddrinfo(host, port, family=family, type=socket.SOCK_STREAM)]

class ExecutorResolver(Resolver):
    """Resolver implementation using a `concurrent.futures.Executor`.

    Use this instead of `ThreadedResolver` when you require additional
    control over the executor being used.

    The executor will be shut down when the resolver is closed unless
    ``close_resolver=False``; use this if you want to reuse the same
    executor elsewhere.

    .. versionchanged:: 5.0
       The ``io_loop`` argument (deprecated since version 4.1) has been removed.

    .. deprecated:: 5.0
       The default `Resolver` now uses `asyncio.loop.getaddrinfo`;
       use that instead of this class.
    """

    def initialize(self, executor: Optional[concurrent.futures.Executor]=None, close_executor: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        if executor is not None:
            self.executor = executor
            self.close_executor = close_executor
        else:
            self.executor = dummy_executor
            self.close_executor = False

    def close(self) -> None:
        if False:
            print('Hello World!')
        if self.close_executor:
            self.executor.shutdown()
        self.executor = None

    @run_on_executor
    def resolve(self, host: str, port: int, family: socket.AddressFamily=socket.AF_UNSPEC) -> List[Tuple[int, Any]]:
        if False:
            print('Hello World!')
        return _resolve_addr(host, port, family)

class BlockingResolver(ExecutorResolver):
    """Default `Resolver` implementation, using `socket.getaddrinfo`.

    The `.IOLoop` will be blocked during the resolution, although the
    callback will not be run until the next `.IOLoop` iteration.

    .. deprecated:: 5.0
       The default `Resolver` now uses `.IOLoop.run_in_executor`; use that instead
       of this class.
    """

    def initialize(self) -> None:
        if False:
            print('Hello World!')
        super().initialize()

class ThreadedResolver(ExecutorResolver):
    """Multithreaded non-blocking `Resolver` implementation.

    Requires the `concurrent.futures` package to be installed
    (available in the standard library since Python 3.2,
    installable with ``pip install futures`` in older versions).

    The thread pool size can be configured with::

        Resolver.configure('tornado.netutil.ThreadedResolver',
                           num_threads=10)

    .. versionchanged:: 3.1
       All ``ThreadedResolvers`` share a single thread pool, whose
       size is set by the first one to be created.

    .. deprecated:: 5.0
       The default `Resolver` now uses `.IOLoop.run_in_executor`; use that instead
       of this class.
    """
    _threadpool = None
    _threadpool_pid = None

    def initialize(self, num_threads: int=10) -> None:
        if False:
            i = 10
            return i + 15
        threadpool = ThreadedResolver._create_threadpool(num_threads)
        super().initialize(executor=threadpool, close_executor=False)

    @classmethod
    def _create_threadpool(cls, num_threads: int) -> concurrent.futures.ThreadPoolExecutor:
        if False:
            for i in range(10):
                print('nop')
        pid = os.getpid()
        if cls._threadpool_pid != pid:
            cls._threadpool = None
        if cls._threadpool is None:
            cls._threadpool = concurrent.futures.ThreadPoolExecutor(num_threads)
            cls._threadpool_pid = pid
        return cls._threadpool

class OverrideResolver(Resolver):
    """Wraps a resolver with a mapping of overrides.

    This can be used to make local DNS changes (e.g. for testing)
    without modifying system-wide settings.

    The mapping can be in three formats::

        {
            # Hostname to host or ip
            "example.com": "127.0.1.1",

            # Host+port to host+port
            ("login.example.com", 443): ("localhost", 1443),

            # Host+port+address family to host+port
            ("login.example.com", 443, socket.AF_INET6): ("::1", 1443),
        }

    .. versionchanged:: 5.0
       Added support for host-port-family triplets.
    """

    def initialize(self, resolver: Resolver, mapping: dict) -> None:
        if False:
            return 10
        self.resolver = resolver
        self.mapping = mapping

    def close(self) -> None:
        if False:
            while True:
                i = 10
        self.resolver.close()

    def resolve(self, host: str, port: int, family: socket.AddressFamily=socket.AF_UNSPEC) -> Awaitable[List[Tuple[int, Any]]]:
        if False:
            i = 10
            return i + 15
        if (host, port, family) in self.mapping:
            (host, port) = self.mapping[host, port, family]
        elif (host, port) in self.mapping:
            (host, port) = self.mapping[host, port]
        elif host in self.mapping:
            host = self.mapping[host]
        return self.resolver.resolve(host, port, family)
_SSL_CONTEXT_KEYWORDS = frozenset(['ssl_version', 'certfile', 'keyfile', 'cert_reqs', 'ca_certs', 'ciphers'])

def ssl_options_to_context(ssl_options: Union[Dict[str, Any], ssl.SSLContext], server_side: Optional[bool]=None) -> ssl.SSLContext:
    if False:
        print('Hello World!')
    'Try to convert an ``ssl_options`` dictionary to an\n    `~ssl.SSLContext` object.\n\n    The ``ssl_options`` dictionary contains keywords to be passed to\n    ``ssl.SSLContext.wrap_socket``.  In Python 2.7.9+, `ssl.SSLContext` objects can\n    be used instead.  This function converts the dict form to its\n    `~ssl.SSLContext` equivalent, and may be used when a component which\n    accepts both forms needs to upgrade to the `~ssl.SSLContext` version\n    to use features like SNI or NPN.\n\n    .. versionchanged:: 6.2\n\n       Added server_side argument. Omitting this argument will\n       result in a DeprecationWarning on Python 3.10.\n\n    '
    if isinstance(ssl_options, ssl.SSLContext):
        return ssl_options
    assert isinstance(ssl_options, dict)
    assert all((k in _SSL_CONTEXT_KEYWORDS for k in ssl_options)), ssl_options
    default_version = ssl.PROTOCOL_TLS
    if server_side:
        default_version = ssl.PROTOCOL_TLS_SERVER
    elif server_side is not None:
        default_version = ssl.PROTOCOL_TLS_CLIENT
    context = ssl.SSLContext(ssl_options.get('ssl_version', default_version))
    if 'certfile' in ssl_options:
        context.load_cert_chain(ssl_options['certfile'], ssl_options.get('keyfile', None))
    if 'cert_reqs' in ssl_options:
        if ssl_options['cert_reqs'] == ssl.CERT_NONE:
            context.check_hostname = False
        context.verify_mode = ssl_options['cert_reqs']
    if 'ca_certs' in ssl_options:
        context.load_verify_locations(ssl_options['ca_certs'])
    if 'ciphers' in ssl_options:
        context.set_ciphers(ssl_options['ciphers'])
    if hasattr(ssl, 'OP_NO_COMPRESSION'):
        context.options |= ssl.OP_NO_COMPRESSION
    return context

def ssl_wrap_socket(socket: socket.socket, ssl_options: Union[Dict[str, Any], ssl.SSLContext], server_hostname: Optional[str]=None, server_side: Optional[bool]=None, **kwargs: Any) -> ssl.SSLSocket:
    if False:
        while True:
            i = 10
    'Returns an ``ssl.SSLSocket`` wrapping the given socket.\n\n    ``ssl_options`` may be either an `ssl.SSLContext` object or a\n    dictionary (as accepted by `ssl_options_to_context`).  Additional\n    keyword arguments are passed to `ssl.SSLContext.wrap_socket`.\n\n    .. versionchanged:: 6.2\n\n       Added server_side argument. Omitting this argument will\n       result in a DeprecationWarning on Python 3.10.\n    '
    context = ssl_options_to_context(ssl_options, server_side=server_side)
    if server_side is None:
        server_side = False
    assert ssl.HAS_SNI
    return context.wrap_socket(socket, server_hostname=server_hostname, server_side=server_side, **kwargs)