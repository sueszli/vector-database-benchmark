"""
This module defines "server instances", which manage
the TCP/UDP servers spawned by mitmproxy as specified by the proxy mode.

Example:

    mode = ProxyMode.parse("reverse:https://example.com")
    inst = ServerInstance.make(mode, manager_that_handles_callbacks)
    await inst.start()
    # TCP server is running now.
"""
from __future__ import annotations
import asyncio
import errno
import json
import logging
import os
import socket
import sys
import textwrap
import typing
from abc import ABCMeta
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from typing import ClassVar
from typing import Generic
from typing import get_args
from typing import TypeVar
import mitmproxy_rs
from mitmproxy import ctx
from mitmproxy import flow
from mitmproxy import platform
from mitmproxy.connection import Address
from mitmproxy.master import Master
from mitmproxy.net import local_ip
from mitmproxy.net import udp
from mitmproxy.proxy import commands
from mitmproxy.proxy import layers
from mitmproxy.proxy import mode_specs
from mitmproxy.proxy import server
from mitmproxy.proxy.context import Context
from mitmproxy.proxy.layer import Layer
from mitmproxy.utils import human
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self
logger = logging.getLogger(__name__)

class ProxyConnectionHandler(server.LiveConnectionHandler):
    master: Master

    def __init__(self, master, r, w, options, mode):
        if False:
            print('Hello World!')
        self.master = master
        super().__init__(r, w, options, mode)
        self.log_prefix = f'{human.format_address(self.client.peername)}: '

    async def handle_hook(self, hook: commands.StartHook) -> None:
        with self.timeout_watchdog.disarm():
            (data,) = hook.args()
            await self.master.addons.handle_lifecycle(hook)
            if isinstance(data, flow.Flow):
                await data.wait_for_resume()
M = TypeVar('M', bound=mode_specs.ProxyMode)

class ServerManager(typing.Protocol):
    connections: dict[tuple | str, ProxyConnectionHandler]

    @contextmanager
    def register_connection(self, connection_id: tuple | str, handler: ProxyConnectionHandler):
        if False:
            while True:
                i = 10
        ...

class ServerInstance(Generic[M], metaclass=ABCMeta):
    __modes: ClassVar[dict[str, type[ServerInstance]]] = {}
    last_exception: Exception | None = None

    def __init__(self, mode: M, manager: ServerManager):
        if False:
            return 10
        self.mode: M = mode
        self.manager: ServerManager = manager

    def __init_subclass__(cls, **kwargs):
        if False:
            print('Hello World!')
        'Register all subclasses so that make() finds them.'
        mode = get_args(cls.__orig_bases__[0])[0]
        if not isinstance(mode, TypeVar):
            assert issubclass(mode, mode_specs.ProxyMode)
            assert mode.type_name not in ServerInstance.__modes
            ServerInstance.__modes[mode.type_name] = cls

    @classmethod
    def make(cls, mode: mode_specs.ProxyMode | str, manager: ServerManager) -> Self:
        if False:
            i = 10
            return i + 15
        if isinstance(mode, str):
            mode = mode_specs.ProxyMode.parse(mode)
        inst = ServerInstance.__modes[mode.type_name](mode, manager)
        if not isinstance(inst, cls):
            raise ValueError(f'{mode!r} is not a spec for a {cls.__name__} server.')
        return inst

    @property
    @abstractmethod
    def is_running(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        pass

    async def start(self) -> None:
        try:
            await self._start()
        except Exception as e:
            self.last_exception = e
            raise
        else:
            self.last_exception = None
        if self.listen_addrs:
            addrs = ' and '.join({human.format_address(a) for a in self.listen_addrs})
            logger.info(f'{self.mode.description} listening at {addrs}.')
        else:
            logger.info(f'{self.mode.description} started.')

    async def stop(self) -> None:
        listen_addrs = self.listen_addrs
        try:
            await self._stop()
        except Exception as e:
            self.last_exception = e
            raise
        else:
            self.last_exception = None
        if listen_addrs:
            addrs = ' and '.join({human.format_address(a) for a in listen_addrs})
            logger.info(f'{self.mode.description} at {addrs} stopped.')
        else:
            logger.info(f'{self.mode.description} stopped.')

    @abstractmethod
    async def _start(self) -> None:
        pass

    @abstractmethod
    async def _stop(self) -> None:
        pass

    @property
    @abstractmethod
    def listen_addrs(self) -> tuple[Address, ...]:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def make_top_layer(self, context: Context) -> Layer:
        if False:
            while True:
                i = 10
        pass

    def to_json(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return {'type': self.mode.type_name, 'description': self.mode.description, 'full_spec': self.mode.full_spec, 'is_running': self.is_running, 'last_exception': str(self.last_exception) if self.last_exception else None, 'listen_addrs': self.listen_addrs}

    async def handle_tcp_connection(self, reader: asyncio.StreamReader | mitmproxy_rs.TcpStream, writer: asyncio.StreamWriter | mitmproxy_rs.TcpStream) -> None:
        handler = ProxyConnectionHandler(ctx.master, reader, writer, ctx.options, self.mode)
        handler.layer = self.make_top_layer(handler.layer.context)
        if isinstance(self.mode, mode_specs.TransparentMode):
            s = cast(socket.socket, writer.get_extra_info('socket'))
            try:
                assert platform.original_addr
                original_dst = platform.original_addr(s)
            except Exception as e:
                logger.error(f'Transparent mode failure: {e!r}')
                return
            else:
                handler.layer.context.client.sockname = original_dst
                handler.layer.context.server.address = original_dst
        elif isinstance(self.mode, (mode_specs.WireGuardMode, mode_specs.LocalMode)):
            handler.layer.context.server.address = writer.get_extra_info('destination_address', handler.layer.context.client.sockname)
        with self.manager.register_connection(handler.layer.context.client.id, handler):
            await handler.handle_client()

    def handle_udp_datagram(self, transport: asyncio.DatagramTransport | mitmproxy_rs.DatagramTransport, data: bytes, remote_addr: Address, local_addr: Address) -> None:
        if False:
            return 10
        connection_id = (remote_addr, local_addr)
        if connection_id not in self.manager.connections:
            reader = udp.DatagramReader()
            writer = udp.DatagramWriter(transport, remote_addr, reader)
            handler = ProxyConnectionHandler(ctx.master, reader, writer, ctx.options, self.mode)
            handler.timeout_watchdog.CONNECTION_TIMEOUT = 20
            handler.layer = self.make_top_layer(handler.layer.context)
            handler.layer.context.client.transport_protocol = 'udp'
            handler.layer.context.server.transport_protocol = 'udp'
            if isinstance(self.mode, (mode_specs.WireGuardMode, mode_specs.LocalMode)):
                handler.layer.context.server.address = local_addr
            self.manager.connections[connection_id] = handler
            t = asyncio.create_task(self.handle_udp_connection(connection_id, handler))
            handler._handle_udp_task = t
        else:
            handler = self.manager.connections[connection_id]
            reader = cast(udp.DatagramReader, handler.transports[handler.client].reader)
        reader.feed_data(data, remote_addr)

    async def handle_udp_connection(self, connection_id: tuple, handler: ProxyConnectionHandler) -> None:
        with self.manager.register_connection(connection_id, handler):
            await handler.handle_client()

class AsyncioServerInstance(ServerInstance[M], metaclass=ABCMeta):
    _servers: list[asyncio.Server | udp.UdpServer]

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        self._servers = []
        super().__init__(*args, **kwargs)

    @property
    def is_running(self) -> bool:
        if False:
            while True:
                i = 10
        return bool(self._servers)

    @property
    def listen_addrs(self) -> tuple[Address, ...]:
        if False:
            return 10
        return tuple((sock.getsockname() for serv in self._servers for sock in serv.sockets))

    async def _start(self) -> None:
        assert not self._servers
        host = self.mode.listen_host(ctx.options.listen_host)
        port = self.mode.listen_port(ctx.options.listen_port)
        try:
            self._servers = await self.listen(host, port)
        except OSError as e:
            message = f"{self.mode.description} failed to listen on {host or '*'}:{port} with {e}"
            if e.errno == errno.EADDRINUSE and self.mode.custom_listen_port is None:
                assert self.mode.custom_listen_host is None
                message += f'\nTry specifying a different port by using `--mode {self.mode.full_spec}@{port + 2}`.'
            raise OSError(e.errno, message, e.filename) from e

    async def _stop(self) -> None:
        assert self._servers
        try:
            for s in self._servers:
                s.close()
        finally:
            self._servers = []

    async def listen(self, host: str, port: int) -> list[asyncio.Server | udp.UdpServer]:
        if self.mode.transport_protocol == 'tcp':
            if port == 0:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.bind(('', 0))
                    fixed_port = s.getsockname()[1]
                    s.close()
                    return [await asyncio.start_server(self.handle_tcp_connection, host, fixed_port)]
                except Exception as e:
                    logger.debug(f'Failed to listen on a single port ({e!r}), falling back to default behavior.')
            return [await asyncio.start_server(self.handle_tcp_connection, host, port)]
        elif self.mode.transport_protocol == 'udp':
            if not host:
                ipv4 = await udp.start_server(self.handle_udp_datagram, '0.0.0.0', port)
                try:
                    ipv6 = await udp.start_server(self.handle_udp_datagram, '::', port or ipv4.sockets[0].getsockname()[1])
                except Exception:
                    logger.debug("Failed to listen on '::', listening on IPv4 only.")
                    return [ipv4]
                else:
                    return [ipv4, ipv6]
            return [await udp.start_server(self.handle_udp_datagram, host, port)]
        else:
            raise AssertionError(self.mode.transport_protocol)

class WireGuardServerInstance(ServerInstance[mode_specs.WireGuardMode]):
    _server: mitmproxy_rs.WireGuardServer | None = None
    server_key: str
    client_key: str

    def make_top_layer(self, context: Context) -> Layer:
        if False:
            print('Hello World!')
        return layers.modes.TransparentProxy(context)

    @property
    def is_running(self) -> bool:
        if False:
            while True:
                i = 10
        return self._server is not None

    @property
    def listen_addrs(self) -> tuple[Address, ...]:
        if False:
            print('Hello World!')
        if self._server:
            return (self._server.getsockname(),)
        else:
            return tuple()

    async def _start(self) -> None:
        assert self._server is None
        host = self.mode.listen_host(ctx.options.listen_host)
        port = self.mode.listen_port(ctx.options.listen_port)
        if self.mode.data:
            conf_path = Path(self.mode.data).expanduser()
        else:
            conf_path = Path(ctx.options.confdir).expanduser() / 'wireguard.conf'
        if not conf_path.exists():
            conf_path.parent.mkdir(parents=True, exist_ok=True)
            conf_path.write_text(json.dumps({'server_key': mitmproxy_rs.genkey(), 'client_key': mitmproxy_rs.genkey()}, indent=4))
        try:
            c = json.loads(conf_path.read_text())
            self.server_key = c['server_key']
            self.client_key = c['client_key']
        except Exception as e:
            raise ValueError(f'Invalid configuration file ({conf_path}): {e}') from e
        p = mitmproxy_rs.pubkey(self.client_key)
        _ = mitmproxy_rs.pubkey(self.server_key)
        self._server = await mitmproxy_rs.start_wireguard_server(host, port, self.server_key, [p], self.wg_handle_tcp_connection, self.handle_udp_datagram)
        conf = self.client_conf()
        assert conf
        logger.info('-' * 60 + '\n' + conf + '\n' + '-' * 60)

    def client_conf(self) -> str | None:
        if False:
            return 10
        if not self._server:
            return None
        host = local_ip.get_local_ip() or local_ip.get_local_ip6()
        port = self.mode.listen_port(ctx.options.listen_port)
        return textwrap.dedent(f'\n            [Interface]\n            PrivateKey = {self.client_key}\n            Address = 10.0.0.1/32\n            DNS = 10.0.0.53\n\n            [Peer]\n            PublicKey = {mitmproxy_rs.pubkey(self.server_key)}\n            AllowedIPs = 0.0.0.0/0\n            Endpoint = {host}:{port}\n            ').strip()

    def to_json(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return {'wireguard_conf': self.client_conf(), **super().to_json()}

    async def _stop(self) -> None:
        assert self._server is not None
        try:
            self._server.close()
            await self._server.wait_closed()
        finally:
            self._server = None

    async def wg_handle_tcp_connection(self, stream: mitmproxy_rs.TcpStream) -> None:
        await self.handle_tcp_connection(stream, stream)

class LocalRedirectorInstance(ServerInstance[mode_specs.LocalMode]):
    _server: ClassVar[mitmproxy_rs.LocalRedirector | None] = None
    'The local redirector daemon. Will be started once and then reused for all future instances.'
    _instance: ClassVar[LocalRedirectorInstance | None] = None
    'The current LocalRedirectorInstance. Will be unset again if an instance is stopped.'
    listen_addrs = ()

    @property
    def is_running(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._instance is not None

    def make_top_layer(self, context: Context) -> Layer:
        if False:
            print('Hello World!')
        return layers.modes.TransparentProxy(context)

    @classmethod
    async def redirector_handle_tcp_connection(cls, stream: mitmproxy_rs.TcpStream) -> None:
        if cls._instance is not None:
            await cls._instance.handle_tcp_connection(stream, stream)

    @classmethod
    def redirector_handle_datagram(cls, transport: mitmproxy_rs.DatagramTransport, data: bytes, remote_addr: Address, local_addr: Address) -> None:
        if False:
            while True:
                i = 10
        if cls._instance is not None:
            cls._instance.handle_udp_datagram(transport=transport, data=data, remote_addr=remote_addr, local_addr=local_addr)

    async def _start(self) -> None:
        if self._instance:
            raise RuntimeError('Cannot spawn more than one local redirector.')
        if self.mode.data.startswith('!'):
            spec = f'{self.mode.data},{os.getpid()}'
        elif self.mode.data:
            spec = self.mode.data
        else:
            spec = f'!{os.getpid()}'
        cls = self.__class__
        cls._instance = self
        if cls._server is None:
            try:
                cls._server = await mitmproxy_rs.start_local_redirector(cls.redirector_handle_tcp_connection, cls.redirector_handle_datagram)
            except Exception:
                cls._instance = None
                raise
        cls._server.set_intercept(spec)

    async def _stop(self) -> None:
        assert self._instance
        assert self._server
        self.__class__._instance = None
        self._server.set_intercept('')

class RegularInstance(AsyncioServerInstance[mode_specs.RegularMode]):

    def make_top_layer(self, context: Context) -> Layer:
        if False:
            i = 10
            return i + 15
        return layers.modes.HttpProxy(context)

class UpstreamInstance(AsyncioServerInstance[mode_specs.UpstreamMode]):

    def make_top_layer(self, context: Context) -> Layer:
        if False:
            return 10
        return layers.modes.HttpUpstreamProxy(context)

class TransparentInstance(AsyncioServerInstance[mode_specs.TransparentMode]):

    def make_top_layer(self, context: Context) -> Layer:
        if False:
            for i in range(10):
                print('nop')
        return layers.modes.TransparentProxy(context)

class ReverseInstance(AsyncioServerInstance[mode_specs.ReverseMode]):

    def make_top_layer(self, context: Context) -> Layer:
        if False:
            for i in range(10):
                print('nop')
        return layers.modes.ReverseProxy(context)

class Socks5Instance(AsyncioServerInstance[mode_specs.Socks5Mode]):

    def make_top_layer(self, context: Context) -> Layer:
        if False:
            while True:
                i = 10
        return layers.modes.Socks5Proxy(context)

class DnsInstance(AsyncioServerInstance[mode_specs.DnsMode]):

    def make_top_layer(self, context: Context) -> Layer:
        if False:
            for i in range(10):
                print('nop')
        return layers.DNSLayer(context)