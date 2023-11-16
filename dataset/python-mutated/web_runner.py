import asyncio
import signal
import socket
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, List, Optional, Set, Type
from yarl import URL
from .abc import AbstractAccessLogger, AbstractStreamWriter
from .http_parser import RawRequestMessage
from .streams import StreamReader
from .typedefs import PathLike
from .web_app import Application
from .web_log import AccessLogger
from .web_protocol import RequestHandler
from .web_request import Request
from .web_server import Server
try:
    from ssl import SSLContext
except ImportError:
    SSLContext = object
__all__ = ('BaseSite', 'TCPSite', 'UnixSite', 'NamedPipeSite', 'SockSite', 'BaseRunner', 'AppRunner', 'ServerRunner', 'GracefulExit')

class GracefulExit(SystemExit):
    code = 1

def _raise_graceful_exit() -> None:
    if False:
        i = 10
        return i + 15
    raise GracefulExit()

class BaseSite(ABC):
    __slots__ = ('_runner', '_ssl_context', '_backlog', '_server')

    def __init__(self, runner: 'BaseRunner', *, ssl_context: Optional[SSLContext]=None, backlog: int=128) -> None:
        if False:
            i = 10
            return i + 15
        if runner.server is None:
            raise RuntimeError('Call runner.setup() before making a site')
        self._runner = runner
        self._ssl_context = ssl_context
        self._backlog = backlog
        self._server: Optional[asyncio.AbstractServer] = None

    @property
    @abstractmethod
    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    async def start(self) -> None:
        self._runner._reg_site(self)

    async def stop(self) -> None:
        self._runner._check_site(self)
        if self._server is not None:
            self._server.close()
        self._runner._unreg_site(self)

class TCPSite(BaseSite):
    __slots__ = ('_host', '_port', '_reuse_address', '_reuse_port')

    def __init__(self, runner: 'BaseRunner', host: Optional[str]=None, port: Optional[int]=None, *, ssl_context: Optional[SSLContext]=None, backlog: int=128, reuse_address: Optional[bool]=None, reuse_port: Optional[bool]=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(runner, ssl_context=ssl_context, backlog=backlog)
        self._host = host
        if port is None:
            port = 8443 if self._ssl_context else 8080
        self._port = port
        self._reuse_address = reuse_address
        self._reuse_port = reuse_port

    @property
    def name(self) -> str:
        if False:
            return 10
        scheme = 'https' if self._ssl_context else 'http'
        host = '0.0.0.0' if self._host is None else self._host
        return str(URL.build(scheme=scheme, host=host, port=self._port))

    async def start(self) -> None:
        await super().start()
        loop = asyncio.get_event_loop()
        server = self._runner.server
        assert server is not None
        self._server = await loop.create_server(server, self._host, self._port, ssl=self._ssl_context, backlog=self._backlog, reuse_address=self._reuse_address, reuse_port=self._reuse_port)

class UnixSite(BaseSite):
    __slots__ = ('_path',)

    def __init__(self, runner: 'BaseRunner', path: PathLike, *, ssl_context: Optional[SSLContext]=None, backlog: int=128) -> None:
        if False:
            print('Hello World!')
        super().__init__(runner, ssl_context=ssl_context, backlog=backlog)
        self._path = path

    @property
    def name(self) -> str:
        if False:
            i = 10
            return i + 15
        scheme = 'https' if self._ssl_context else 'http'
        return f'{scheme}://unix:{self._path}:'

    async def start(self) -> None:
        await super().start()
        loop = asyncio.get_event_loop()
        server = self._runner.server
        assert server is not None
        self._server = await loop.create_unix_server(server, self._path, ssl=self._ssl_context, backlog=self._backlog)

class NamedPipeSite(BaseSite):
    __slots__ = ('_path',)

    def __init__(self, runner: 'BaseRunner', path: str) -> None:
        if False:
            return 10
        loop = asyncio.get_event_loop()
        if not isinstance(loop, asyncio.ProactorEventLoop):
            raise RuntimeError('Named Pipes only available in proactorloop under windows')
        super().__init__(runner)
        self._path = path

    @property
    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._path

    async def start(self) -> None:
        await super().start()
        loop = asyncio.get_event_loop()
        server = self._runner.server
        assert server is not None
        _server = await loop.start_serving_pipe(server, self._path)
        self._server = _server[0]

class SockSite(BaseSite):
    __slots__ = ('_sock', '_name')

    def __init__(self, runner: 'BaseRunner', sock: socket.socket, *, ssl_context: Optional[SSLContext]=None, backlog: int=128) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(runner, ssl_context=ssl_context, backlog=backlog)
        self._sock = sock
        scheme = 'https' if self._ssl_context else 'http'
        if hasattr(socket, 'AF_UNIX') and sock.family == socket.AF_UNIX:
            name = f'{scheme}://unix:{sock.getsockname()}:'
        else:
            (host, port) = sock.getsockname()[:2]
            name = str(URL.build(scheme=scheme, host=host, port=port))
        self._name = name

    @property
    def name(self) -> str:
        if False:
            print('Hello World!')
        return self._name

    async def start(self) -> None:
        await super().start()
        loop = asyncio.get_event_loop()
        server = self._runner.server
        assert server is not None
        self._server = await loop.create_server(server, sock=self._sock, ssl=self._ssl_context, backlog=self._backlog)

class BaseRunner(ABC):
    __slots__ = ('shutdown_callback', '_handle_signals', '_kwargs', '_server', '_sites', '_shutdown_timeout')

    def __init__(self, *, handle_signals: bool=False, shutdown_timeout: float=60.0, **kwargs: Any) -> None:
        if False:
            return 10
        self.shutdown_callback: Optional[Callable[[], Awaitable[None]]] = None
        self._handle_signals = handle_signals
        self._kwargs = kwargs
        self._server: Optional[Server] = None
        self._sites: List[BaseSite] = []
        self._shutdown_timeout = shutdown_timeout

    @property
    def server(self) -> Optional[Server]:
        if False:
            return 10
        return self._server

    @property
    def addresses(self) -> List[Any]:
        if False:
            i = 10
            return i + 15
        ret: List[Any] = []
        for site in self._sites:
            server = site._server
            if server is not None:
                sockets = server.sockets
                if sockets is not None:
                    for sock in sockets:
                        ret.append(sock.getsockname())
        return ret

    @property
    def sites(self) -> Set[BaseSite]:
        if False:
            i = 10
            return i + 15
        return set(self._sites)

    async def setup(self) -> None:
        loop = asyncio.get_event_loop()
        if self._handle_signals:
            try:
                loop.add_signal_handler(signal.SIGINT, _raise_graceful_exit)
                loop.add_signal_handler(signal.SIGTERM, _raise_graceful_exit)
            except NotImplementedError:
                pass
        self._server = await self._make_server()

    @abstractmethod
    async def shutdown(self) -> None:
        """Call any shutdown hooks to help server close gracefully."""

    async def cleanup(self) -> None:
        for site in list(self._sites):
            await site.stop()
        if self._server:
            self._server.pre_shutdown()
            await self.shutdown()
            if self.shutdown_callback:
                await self.shutdown_callback()
            await self._server.shutdown(self._shutdown_timeout)
        await self._cleanup_server()
        self._server = None
        if self._handle_signals:
            loop = asyncio.get_running_loop()
            try:
                loop.remove_signal_handler(signal.SIGINT)
                loop.remove_signal_handler(signal.SIGTERM)
            except NotImplementedError:
                pass

    @abstractmethod
    async def _make_server(self) -> Server:
        pass

    @abstractmethod
    async def _cleanup_server(self) -> None:
        pass

    def _reg_site(self, site: BaseSite) -> None:
        if False:
            return 10
        if site in self._sites:
            raise RuntimeError(f'Site {site} is already registered in runner {self}')
        self._sites.append(site)

    def _check_site(self, site: BaseSite) -> None:
        if False:
            print('Hello World!')
        if site not in self._sites:
            raise RuntimeError(f'Site {site} is not registered in runner {self}')

    def _unreg_site(self, site: BaseSite) -> None:
        if False:
            return 10
        if site not in self._sites:
            raise RuntimeError(f'Site {site} is not registered in runner {self}')
        self._sites.remove(site)

class ServerRunner(BaseRunner):
    """Low-level web server runner"""
    __slots__ = ('_web_server',)

    def __init__(self, web_server: Server, *, handle_signals: bool=False, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        super().__init__(handle_signals=handle_signals, **kwargs)
        self._web_server = web_server

    async def shutdown(self) -> None:
        pass

    async def _make_server(self) -> Server:
        return self._web_server

    async def _cleanup_server(self) -> None:
        pass

class AppRunner(BaseRunner):
    """Web Application runner"""
    __slots__ = ('_app',)

    def __init__(self, app: Application, *, handle_signals: bool=False, access_log_class: Type[AbstractAccessLogger]=AccessLogger, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        if not isinstance(app, Application):
            raise TypeError('The first argument should be web.Application instance, got {!r}'.format(app))
        kwargs['access_log_class'] = access_log_class
        if app._handler_args:
            for (k, v) in app._handler_args.items():
                kwargs[k] = v
        if not issubclass(kwargs['access_log_class'], AbstractAccessLogger):
            raise TypeError('access_log_class must be subclass of aiohttp.abc.AbstractAccessLogger, got {}'.format(kwargs['access_log_class']))
        super().__init__(handle_signals=handle_signals, **kwargs)
        self._app = app

    @property
    def app(self) -> Application:
        if False:
            while True:
                i = 10
        return self._app

    async def shutdown(self) -> None:
        await self._app.shutdown()

    async def _make_server(self) -> Server:
        self._app.on_startup.freeze()
        await self._app.startup()
        self._app.freeze()
        return Server(self._app._handle, request_factory=self._make_request, **self._kwargs)

    def _make_request(self, message: RawRequestMessage, payload: StreamReader, protocol: RequestHandler, writer: AbstractStreamWriter, task: 'asyncio.Task[None]', _cls: Type[Request]=Request) -> Request:
        if False:
            return 10
        loop = asyncio.get_running_loop()
        return _cls(message, payload, protocol, writer, task, loop, client_max_size=self.app._client_max_size)

    async def _cleanup_server(self) -> None:
        await self._app.cleanup()