"""trio async I/O library query support"""
import socket
import trio
import trio.socket
import dns._asyncbackend
import dns.exception
import dns.inet

def _maybe_timeout(timeout):
    if False:
        return 10
    if timeout is not None:
        return trio.move_on_after(timeout)
    else:
        return dns._asyncbackend.NullContext()
_lltuple = dns.inet.low_level_address_tuple

class DatagramSocket(dns._asyncbackend.DatagramSocket):

    def __init__(self, socket):
        if False:
            i = 10
            return i + 15
        super().__init__(socket.family)
        self.socket = socket

    async def sendto(self, what, destination, timeout):
        with _maybe_timeout(timeout):
            return await self.socket.sendto(what, destination)
        raise dns.exception.Timeout(timeout=timeout)

    async def recvfrom(self, size, timeout):
        with _maybe_timeout(timeout):
            return await self.socket.recvfrom(size)
        raise dns.exception.Timeout(timeout=timeout)

    async def close(self):
        self.socket.close()

    async def getpeername(self):
        return self.socket.getpeername()

    async def getsockname(self):
        return self.socket.getsockname()

    async def getpeercert(self, timeout):
        raise NotImplementedError

class StreamSocket(dns._asyncbackend.StreamSocket):

    def __init__(self, family, stream, tls=False):
        if False:
            for i in range(10):
                print('nop')
        self.family = family
        self.stream = stream
        self.tls = tls

    async def sendall(self, what, timeout):
        with _maybe_timeout(timeout):
            return await self.stream.send_all(what)
        raise dns.exception.Timeout(timeout=timeout)

    async def recv(self, size, timeout):
        with _maybe_timeout(timeout):
            return await self.stream.receive_some(size)
        raise dns.exception.Timeout(timeout=timeout)

    async def close(self):
        await self.stream.aclose()

    async def getpeername(self):
        if self.tls:
            return self.stream.transport_stream.socket.getpeername()
        else:
            return self.stream.socket.getpeername()

    async def getsockname(self):
        if self.tls:
            return self.stream.transport_stream.socket.getsockname()
        else:
            return self.stream.socket.getsockname()

    async def getpeercert(self, timeout):
        if self.tls:
            with _maybe_timeout(timeout):
                await self.stream.do_handshake()
            return self.stream.getpeercert()
        else:
            raise NotImplementedError
try:
    import httpcore
    import httpcore._backends.trio
    import httpx
    _CoreAsyncNetworkBackend = httpcore.AsyncNetworkBackend
    _CoreTrioStream = httpcore._backends.trio.TrioStream
    from dns.query import _compute_times, _expiration_for_this_attempt, _remaining

    class _NetworkBackend(_CoreAsyncNetworkBackend):

        def __init__(self, resolver, local_port, bootstrap_address, family):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self._local_port = local_port
            self._resolver = resolver
            self._bootstrap_address = bootstrap_address
            self._family = family

        async def connect_tcp(self, host, port, timeout, local_address, socket_options=None):
            addresses = []
            (_, expiration) = _compute_times(timeout)
            if dns.inet.is_address(host):
                addresses.append(host)
            elif self._bootstrap_address is not None:
                addresses.append(self._bootstrap_address)
            else:
                timeout = _remaining(expiration)
                family = self._family
                if local_address:
                    family = dns.inet.af_for_address(local_address)
                answers = await self._resolver.resolve_name(host, family=family, lifetime=timeout)
                addresses = answers.addresses()
            for address in addresses:
                try:
                    af = dns.inet.af_for_address(address)
                    if local_address is not None or self._local_port != 0:
                        source = (local_address, self._local_port)
                    else:
                        source = None
                    destination = (address, port)
                    attempt_expiration = _expiration_for_this_attempt(2.0, expiration)
                    timeout = _remaining(attempt_expiration)
                    sock = await Backend().make_socket(af, socket.SOCK_STREAM, 0, source, destination, timeout)
                    return _CoreTrioStream(sock.stream)
                except Exception:
                    continue
            raise httpcore.ConnectError

        async def connect_unix_socket(self, path, timeout, socket_options=None):
            raise NotImplementedError

        async def sleep(self, seconds):
            await trio.sleep(seconds)

    class _HTTPTransport(httpx.AsyncHTTPTransport):

        def __init__(self, *args, local_port=0, bootstrap_address=None, resolver=None, family=socket.AF_UNSPEC, **kwargs):
            if False:
                return 10
            if resolver is None:
                import dns.asyncresolver
                resolver = dns.asyncresolver.Resolver()
            super().__init__(*args, **kwargs)
            self._pool._network_backend = _NetworkBackend(resolver, local_port, bootstrap_address, family)
except ImportError:
    _HTTPTransport = dns._asyncbackend.NullTransport

class Backend(dns._asyncbackend.Backend):

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'trio'

    async def make_socket(self, af, socktype, proto=0, source=None, destination=None, timeout=None, ssl_context=None, server_hostname=None):
        s = trio.socket.socket(af, socktype, proto)
        stream = None
        try:
            if source:
                await s.bind(_lltuple(source, af))
            if socktype == socket.SOCK_STREAM:
                connected = False
                with _maybe_timeout(timeout):
                    await s.connect(_lltuple(destination, af))
                    connected = True
                if not connected:
                    raise dns.exception.Timeout(timeout=timeout)
        except Exception:
            s.close()
            raise
        if socktype == socket.SOCK_DGRAM:
            return DatagramSocket(s)
        elif socktype == socket.SOCK_STREAM:
            stream = trio.SocketStream(s)
            tls = False
            if ssl_context:
                tls = True
                try:
                    stream = trio.SSLStream(stream, ssl_context, server_hostname=server_hostname)
                except Exception:
                    await stream.aclose()
                    raise
            return StreamSocket(af, stream, tls)
        raise NotImplementedError('unsupported socket ' + f'type {socktype}')

    async def sleep(self, interval):
        await trio.sleep(interval)

    def get_transport_class(self):
        if False:
            print('Hello World!')
        return _HTTPTransport

    async def wait_for(self, awaitable, timeout):
        with _maybe_timeout(timeout):
            return await awaitable
        raise dns.exception.Timeout(timeout=timeout)