import asyncio
import threading
from asyncio import AbstractEventLoop
from hypercorn import Config
from hypercorn.asyncio import serve
from hypercorn.typing import ASGIFramework
from localstack.aws.gateway import Gateway
from localstack.aws.handlers.proxy import ProxyHandler
from localstack.aws.serving.asgi import AsgiGateway
from localstack.config import HostAndPort
from localstack.logging.setup import setup_hypercorn_logger
from localstack.utils.collections import ensure_list
from localstack.utils.functions import call_safe
from localstack.utils.serving import Server
from localstack.utils.ssl import create_ssl_cert, install_predefined_cert_if_available

class HypercornServer(Server):
    """
    A sync wrapper around Hypercorn that implements the ``Server`` interface.
    """

    def __init__(self, app: ASGIFramework, config: Config, loop: AbstractEventLoop=None):
        if False:
            i = 10
            return i + 15
        '\n        Create a new Hypercorn server instance. Note that, if you pass an event loop to the constructor,\n        you are yielding control of that event loop to the server, as it will invoke `run_until_complete` and\n        shutdown the loop.\n\n        :param app: the ASGI3 app\n        :param config: the hypercorn config\n        :param loop: optionally the event loop, otherwise ``asyncio.new_event_loop`` will be called\n        '
        self.app = app
        self.config = config
        self.loop = loop or asyncio.new_event_loop()
        self._close = asyncio.Event()
        self._closed = threading.Event()
        parts = config.bind[0].split(':')
        if len(parts) == 1:
            host = parts[0]
            port = 443 if config.ssl_enabled else 80
        else:
            (host, port) = (parts[0], int(parts[1]))
        super().__init__(port, host)

    @property
    def protocol(self):
        if False:
            print('Hello World!')
        return 'https' if self.config.ssl_enabled else 'http'

    def do_run(self):
        if False:
            i = 10
            return i + 15
        self.loop.run_until_complete(serve(self.app, self.config, shutdown_trigger=self._shutdown_trigger))
        self._closed.set()

    def do_shutdown(self):
        if False:
            i = 10
            return i + 15
        asyncio.run_coroutine_threadsafe(self._set_closed(), self.loop)
        self._closed.wait(timeout=10)
        asyncio.run_coroutine_threadsafe(self.loop.shutdown_asyncgens(), self.loop)
        self.loop.shutdown_default_executor()
        self.loop.stop()
        call_safe(self.loop.close)

    async def _set_closed(self):
        self._close.set()

    async def _shutdown_trigger(self):
        await self._close.wait()

class GatewayServer(HypercornServer):
    """
    A Hypercorn-based server implementation which serves a given Gateway.
    It can be used to easily spawn new gateway servers, defining their individual request-, response-, and
    exception-handlers.
    """

    def __init__(self, gateway: Gateway, listen: HostAndPort | list[HostAndPort], use_ssl: bool=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new GatewayServer instance.\n\n        :param gateway: which will be served by this server\n        :param port: defining the port of this server instance\n        :param bind_address: to bind this server instance to. Can be a host string or a list of host strings.\n        :param use_ssl: True if the LocalStack cert should be loaded and HTTP/HTTPS multiplexing should be enabled.\n        '
        config = Config()
        config.h11_pass_raw_headers = True
        setup_hypercorn_logger(config)
        listens = ensure_list(listen)
        config.bind = [str(host_and_port) for host_and_port in listens]
        if use_ssl:
            install_predefined_cert_if_available()
            serial_number = listens[0].port
            (_, cert_file_name, key_file_name) = create_ssl_cert(serial_number=serial_number)
            config.certfile = cert_file_name
            config.keyfile = key_file_name
        loop = asyncio.new_event_loop()
        app = AsgiGateway(gateway, event_loop=loop)
        super().__init__(app, config, loop)

    def do_shutdown(self):
        if False:
            i = 10
            return i + 15
        super().do_shutdown()
        self.app.close()

class ProxyServer(GatewayServer):
    """
    Proxy server implementation which uses the localstack.http.proxy module.
    These server instances can be spawned easily, while implementing HTTP/HTTPS multiplexing (if enabled),
    and just forward all incoming requests to a backend.
    """

    def __init__(self, forward_base_url: str, listen: HostAndPort | list[HostAndPort], use_ssl: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        Creates a new ProxyServer instance.\n\n        :param forward_base_url: URL of the backend system all requests this server receives should be forwarded to\n        :param port: defining the port of this server instance\n        :param bind_address: to bind this server instance to. Can be a host string or a list of host strings.\n        :param use_ssl: True if the LocalStack cert should be loaded and HTTP/HTTPS multiplexing should be enabled.\n        '
        gateway = Gateway()
        gateway.request_handlers.append(ProxyHandler(forward_base_url=forward_base_url))
        super().__init__(gateway, listen, use_ssl)