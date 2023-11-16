"""A non-blocking TCP connection factory.
"""
import functools
import socket
import numbers
import datetime
import ssl
import typing
from tornado.concurrent import Future, future_add_done_callback
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream
from tornado import gen
from tornado.netutil import Resolver
from tornado.gen import TimeoutError
from typing import Any, Union, Dict, Tuple, List, Callable, Iterator, Optional
if typing.TYPE_CHECKING:
    from typing import Set
_INITIAL_CONNECT_TIMEOUT = 0.3

class _Connector(object):
    """A stateless implementation of the "Happy Eyeballs" algorithm.

    "Happy Eyeballs" is documented in RFC6555 as the recommended practice
    for when both IPv4 and IPv6 addresses are available.

    In this implementation, we partition the addresses by family, and
    make the first connection attempt to whichever address was
    returned first by ``getaddrinfo``.  If that connection fails or
    times out, we begin a connection in parallel to the first address
    of the other family.  If there are additional failures we retry
    with other addresses, keeping one connection attempt per family
    in flight at a time.

    http://tools.ietf.org/html/rfc6555

    """

    def __init__(self, addrinfo: List[Tuple], connect: Callable[[socket.AddressFamily, Tuple], Tuple[IOStream, 'Future[IOStream]']]) -> None:
        if False:
            while True:
                i = 10
        self.io_loop = IOLoop.current()
        self.connect = connect
        self.future = Future()
        self.timeout = None
        self.connect_timeout = None
        self.last_error = None
        self.remaining = len(addrinfo)
        (self.primary_addrs, self.secondary_addrs) = self.split(addrinfo)
        self.streams = set()

    @staticmethod
    def split(addrinfo: List[Tuple]) -> Tuple[List[Tuple[socket.AddressFamily, Tuple]], List[Tuple[socket.AddressFamily, Tuple]]]:
        if False:
            for i in range(10):
                print('nop')
        'Partition the ``addrinfo`` list by address family.\n\n        Returns two lists.  The first list contains the first entry from\n        ``addrinfo`` and all others with the same family, and the\n        second list contains all other addresses (normally one list will\n        be AF_INET and the other AF_INET6, although non-standard resolvers\n        may return additional families).\n        '
        primary = []
        secondary = []
        primary_af = addrinfo[0][0]
        for (af, addr) in addrinfo:
            if af == primary_af:
                primary.append((af, addr))
            else:
                secondary.append((af, addr))
        return (primary, secondary)

    def start(self, timeout: float=_INITIAL_CONNECT_TIMEOUT, connect_timeout: Optional[Union[float, datetime.timedelta]]=None) -> 'Future[Tuple[socket.AddressFamily, Any, IOStream]]':
        if False:
            return 10
        self.try_connect(iter(self.primary_addrs))
        self.set_timeout(timeout)
        if connect_timeout is not None:
            self.set_connect_timeout(connect_timeout)
        return self.future

    def try_connect(self, addrs: Iterator[Tuple[socket.AddressFamily, Tuple]]) -> None:
        if False:
            print('Hello World!')
        try:
            (af, addr) = next(addrs)
        except StopIteration:
            if self.remaining == 0 and (not self.future.done()):
                self.future.set_exception(self.last_error or IOError('connection failed'))
            return
        (stream, future) = self.connect(af, addr)
        self.streams.add(stream)
        future_add_done_callback(future, functools.partial(self.on_connect_done, addrs, af, addr))

    def on_connect_done(self, addrs: Iterator[Tuple[socket.AddressFamily, Tuple]], af: socket.AddressFamily, addr: Tuple, future: 'Future[IOStream]') -> None:
        if False:
            for i in range(10):
                print('nop')
        self.remaining -= 1
        try:
            stream = future.result()
        except Exception as e:
            if self.future.done():
                return
            self.last_error = e
            self.try_connect(addrs)
            if self.timeout is not None:
                self.io_loop.remove_timeout(self.timeout)
                self.on_timeout()
            return
        self.clear_timeouts()
        if self.future.done():
            stream.close()
        else:
            self.streams.discard(stream)
            self.future.set_result((af, addr, stream))
            self.close_streams()

    def set_timeout(self, timeout: float) -> None:
        if False:
            while True:
                i = 10
        self.timeout = self.io_loop.add_timeout(self.io_loop.time() + timeout, self.on_timeout)

    def on_timeout(self) -> None:
        if False:
            i = 10
            return i + 15
        self.timeout = None
        if not self.future.done():
            self.try_connect(iter(self.secondary_addrs))

    def clear_timeout(self) -> None:
        if False:
            i = 10
            return i + 15
        if self.timeout is not None:
            self.io_loop.remove_timeout(self.timeout)

    def set_connect_timeout(self, connect_timeout: Union[float, datetime.timedelta]) -> None:
        if False:
            print('Hello World!')
        self.connect_timeout = self.io_loop.add_timeout(connect_timeout, self.on_connect_timeout)

    def on_connect_timeout(self) -> None:
        if False:
            while True:
                i = 10
        if not self.future.done():
            self.future.set_exception(TimeoutError())
        self.close_streams()

    def clear_timeouts(self) -> None:
        if False:
            i = 10
            return i + 15
        if self.timeout is not None:
            self.io_loop.remove_timeout(self.timeout)
        if self.connect_timeout is not None:
            self.io_loop.remove_timeout(self.connect_timeout)

    def close_streams(self) -> None:
        if False:
            while True:
                i = 10
        for stream in self.streams:
            stream.close()

class TCPClient(object):
    """A non-blocking TCP connection factory.

    .. versionchanged:: 5.0
       The ``io_loop`` argument (deprecated since version 4.1) has been removed.
    """

    def __init__(self, resolver: Optional[Resolver]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        if resolver is not None:
            self.resolver = resolver
            self._own_resolver = False
        else:
            self.resolver = Resolver()
            self._own_resolver = True

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._own_resolver:
            self.resolver.close()

    async def connect(self, host: str, port: int, af: socket.AddressFamily=socket.AF_UNSPEC, ssl_options: Optional[Union[Dict[str, Any], ssl.SSLContext]]=None, max_buffer_size: Optional[int]=None, source_ip: Optional[str]=None, source_port: Optional[int]=None, timeout: Optional[Union[float, datetime.timedelta]]=None) -> IOStream:
        """Connect to the given host and port.

        Asynchronously returns an `.IOStream` (or `.SSLIOStream` if
        ``ssl_options`` is not None).

        Using the ``source_ip`` kwarg, one can specify the source
        IP address to use when establishing the connection.
        In case the user needs to resolve and
        use a specific interface, it has to be handled outside
        of Tornado as this depends very much on the platform.

        Raises `TimeoutError` if the input future does not complete before
        ``timeout``, which may be specified in any form allowed by
        `.IOLoop.add_timeout` (i.e. a `datetime.timedelta` or an absolute time
        relative to `.IOLoop.time`)

        Similarly, when the user requires a certain source port, it can
        be specified using the ``source_port`` arg.

        .. versionchanged:: 4.5
           Added the ``source_ip`` and ``source_port`` arguments.

        .. versionchanged:: 5.0
           Added the ``timeout`` argument.
        """
        if timeout is not None:
            if isinstance(timeout, numbers.Real):
                timeout = IOLoop.current().time() + timeout
            elif isinstance(timeout, datetime.timedelta):
                timeout = IOLoop.current().time() + timeout.total_seconds()
            else:
                raise TypeError('Unsupported timeout %r' % timeout)
        if timeout is not None:
            addrinfo = await gen.with_timeout(timeout, self.resolver.resolve(host, port, af))
        else:
            addrinfo = await self.resolver.resolve(host, port, af)
        connector = _Connector(addrinfo, functools.partial(self._create_stream, max_buffer_size, source_ip=source_ip, source_port=source_port))
        (af, addr, stream) = await connector.start(connect_timeout=timeout)
        if ssl_options is not None:
            if timeout is not None:
                stream = await gen.with_timeout(timeout, stream.start_tls(False, ssl_options=ssl_options, server_hostname=host))
            else:
                stream = await stream.start_tls(False, ssl_options=ssl_options, server_hostname=host)
        return stream

    def _create_stream(self, max_buffer_size: int, af: socket.AddressFamily, addr: Tuple, source_ip: Optional[str]=None, source_port: Optional[int]=None) -> Tuple[IOStream, 'Future[IOStream]']:
        if False:
            i = 10
            return i + 15
        source_port_bind = source_port if isinstance(source_port, int) else 0
        source_ip_bind = source_ip
        if source_port_bind and (not source_ip):
            source_ip_bind = '::1' if af == socket.AF_INET6 else '127.0.0.1'
        socket_obj = socket.socket(af)
        if source_port_bind or source_ip_bind:
            try:
                socket_obj.bind((source_ip_bind, source_port_bind))
            except socket.error:
                socket_obj.close()
                raise
        try:
            stream = IOStream(socket_obj, max_buffer_size=max_buffer_size)
        except socket.error as e:
            fu = Future()
            fu.set_exception(e)
            return (stream, fu)
        else:
            return (stream, stream.connect(addr))