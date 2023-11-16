import asyncio
import http
import logging
import re
import urllib
from asyncio.events import TimerHandle
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Literal, Optional, Tuple, Union, cast
import httptools
from uvicorn._types import ASGI3Application, ASGIReceiveEvent, ASGISendEvent, HTTPDisconnectEvent, HTTPRequestEvent, HTTPResponseBodyEvent, HTTPResponseStartEvent, HTTPScope
from uvicorn.config import Config
from uvicorn.logging import TRACE_LOG_LEVEL
from uvicorn.protocols.http.flow_control import CLOSE_HEADER, HIGH_WATER_LIMIT, FlowControl, service_unavailable
from uvicorn.protocols.utils import get_client_addr, get_local_addr, get_path_with_query_string, get_remote_addr, is_ssl
from uvicorn.server import ServerState
HEADER_RE = re.compile(b'[\x00-\x1f\x7f()<>@,;:[]={} \t\\"]')
HEADER_VALUE_RE = re.compile(b'[\x00-\x1f\x7f]')

def _get_status_line(status_code: int) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    try:
        phrase = http.HTTPStatus(status_code).phrase.encode()
    except ValueError:
        phrase = b''
    return b''.join([b'HTTP/1.1 ', str(status_code).encode(), b' ', phrase, b'\r\n'])
STATUS_LINE = {status_code: _get_status_line(status_code) for status_code in range(100, 600)}

class HttpToolsProtocol(asyncio.Protocol):

    def __init__(self, config: Config, server_state: ServerState, app_state: Dict[str, Any], _loop: Optional[asyncio.AbstractEventLoop]=None) -> None:
        if False:
            print('Hello World!')
        if not config.loaded:
            config.load()
        self.config = config
        self.app = config.loaded_app
        self.loop = _loop or asyncio.get_event_loop()
        self.logger = logging.getLogger('uvicorn.error')
        self.access_logger = logging.getLogger('uvicorn.access')
        self.access_log = self.access_logger.hasHandlers()
        self.parser = httptools.HttpRequestParser(self)
        self.ws_protocol_class = config.ws_protocol_class
        self.root_path = config.root_path
        self.limit_concurrency = config.limit_concurrency
        self.app_state = app_state
        self.timeout_keep_alive_task: Optional[TimerHandle] = None
        self.timeout_keep_alive = config.timeout_keep_alive
        self.server_state = server_state
        self.connections = server_state.connections
        self.tasks = server_state.tasks
        self.transport: asyncio.Transport = None
        self.flow: FlowControl = None
        self.server: Optional[Tuple[str, int]] = None
        self.client: Optional[Tuple[str, int]] = None
        self.scheme: Optional[Literal['http', 'https']] = None
        self.pipeline: Deque[Tuple[RequestResponseCycle, ASGI3Application]] = deque()
        self.scope: HTTPScope = None
        self.headers: List[Tuple[bytes, bytes]] = None
        self.expect_100_continue = False
        self.cycle: RequestResponseCycle = None

    def connection_made(self, transport: asyncio.Transport) -> None:
        if False:
            while True:
                i = 10
        self.connections.add(self)
        self.transport = transport
        self.flow = FlowControl(transport)
        self.server = get_local_addr(transport)
        self.client = get_remote_addr(transport)
        self.scheme = 'https' if is_ssl(transport) else 'http'
        if self.logger.level <= TRACE_LOG_LEVEL:
            prefix = '%s:%d - ' % self.client if self.client else ''
            self.logger.log(TRACE_LOG_LEVEL, '%sHTTP connection made', prefix)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.connections.discard(self)
        if self.logger.level <= TRACE_LOG_LEVEL:
            prefix = '%s:%d - ' % self.client if self.client else ''
            self.logger.log(TRACE_LOG_LEVEL, '%sHTTP connection lost', prefix)
        if self.cycle and (not self.cycle.response_complete):
            self.cycle.disconnected = True
        if self.cycle is not None:
            self.cycle.message_event.set()
        if self.flow is not None:
            self.flow.resume_writing()
        if exc is None:
            self.transport.close()
            self._unset_keepalive_if_required()
        self.parser = None

    def eof_received(self) -> None:
        if False:
            print('Hello World!')
        pass

    def _unset_keepalive_if_required(self) -> None:
        if False:
            print('Hello World!')
        if self.timeout_keep_alive_task is not None:
            self.timeout_keep_alive_task.cancel()
            self.timeout_keep_alive_task = None

    def _get_upgrade(self) -> Optional[bytes]:
        if False:
            for i in range(10):
                print('nop')
        connection = []
        upgrade = None
        for (name, value) in self.headers:
            if name == b'connection':
                connection = [token.lower().strip() for token in value.split(b',')]
            if name == b'upgrade':
                upgrade = value.lower()
        if b'upgrade' in connection:
            return upgrade
        return None

    def _should_upgrade_to_ws(self, upgrade: Optional[bytes]) -> bool:
        if False:
            i = 10
            return i + 15
        if upgrade == b'websocket' and self.ws_protocol_class is not None:
            return True
        if self.config.ws == 'auto':
            msg = 'Unsupported upgrade request.'
            self.logger.warning(msg)
            msg = 'No supported WebSocket library detected. Please use "pip install \'uvicorn[standard]\'", or install \'websockets\' or \'wsproto\' manually.'
            self.logger.warning(msg)
        return False

    def _should_upgrade(self) -> bool:
        if False:
            while True:
                i = 10
        upgrade = self._get_upgrade()
        return self._should_upgrade_to_ws(upgrade)

    def data_received(self, data: bytes) -> None:
        if False:
            return 10
        self._unset_keepalive_if_required()
        try:
            self.parser.feed_data(data)
        except httptools.HttpParserError:
            msg = 'Invalid HTTP request received.'
            self.logger.warning(msg)
            self.send_400_response(msg)
            return
        except httptools.HttpParserUpgrade:
            upgrade = self._get_upgrade()
            if self._should_upgrade_to_ws(upgrade):
                self.handle_websocket_upgrade()

    def handle_websocket_upgrade(self) -> None:
        if False:
            while True:
                i = 10
        if self.logger.level <= TRACE_LOG_LEVEL:
            prefix = '%s:%d - ' % self.client if self.client else ''
            self.logger.log(TRACE_LOG_LEVEL, '%sUpgrading to WebSocket', prefix)
        self.connections.discard(self)
        method = self.scope['method'].encode()
        output = [method, b' ', self.url, b' HTTP/1.1\r\n']
        for (name, value) in self.scope['headers']:
            output += [name, b': ', value, b'\r\n']
        output.append(b'\r\n')
        protocol = self.ws_protocol_class(config=self.config, server_state=self.server_state, app_state=self.app_state)
        protocol.connection_made(self.transport)
        protocol.data_received(b''.join(output))
        self.transport.set_protocol(protocol)

    def send_400_response(self, msg: str) -> None:
        if False:
            print('Hello World!')
        content = [STATUS_LINE[400]]
        for (name, value) in self.server_state.default_headers:
            content.extend([name, b': ', value, b'\r\n'])
        content.extend([b'content-type: text/plain; charset=utf-8\r\n', b'content-length: ' + str(len(msg)).encode('ascii') + b'\r\n', b'connection: close\r\n', b'\r\n', msg.encode('ascii')])
        self.transport.write(b''.join(content))
        self.transport.close()

    def on_message_begin(self) -> None:
        if False:
            while True:
                i = 10
        self.url = b''
        self.expect_100_continue = False
        self.headers = []
        self.scope = {'type': 'http', 'asgi': {'version': self.config.asgi_version, 'spec_version': '2.3'}, 'http_version': '1.1', 'server': self.server, 'client': self.client, 'scheme': self.scheme, 'root_path': self.root_path, 'headers': self.headers, 'state': self.app_state.copy()}

    def on_url(self, url: bytes) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.url += url

    def on_header(self, name: bytes, value: bytes) -> None:
        if False:
            i = 10
            return i + 15
        name = name.lower()
        if name == b'expect' and value.lower() == b'100-continue':
            self.expect_100_continue = True
        self.headers.append((name, value))

    def on_headers_complete(self) -> None:
        if False:
            i = 10
            return i + 15
        http_version = self.parser.get_http_version()
        method = self.parser.get_method()
        self.scope['method'] = method.decode('ascii')
        if http_version != '1.1':
            self.scope['http_version'] = http_version
        if self.parser.should_upgrade() and self._should_upgrade():
            return
        parsed_url = httptools.parse_url(self.url)
        raw_path = parsed_url.path
        path = raw_path.decode('ascii')
        if '%' in path:
            path = urllib.parse.unquote(path)
        self.scope['path'] = path
        self.scope['raw_path'] = raw_path
        self.scope['query_string'] = parsed_url.query or b''
        if self.limit_concurrency is not None and (len(self.connections) >= self.limit_concurrency or len(self.tasks) >= self.limit_concurrency):
            app = service_unavailable
            message = 'Exceeded concurrency limit.'
            self.logger.warning(message)
        else:
            app = self.app
        existing_cycle = self.cycle
        self.cycle = RequestResponseCycle(scope=self.scope, transport=self.transport, flow=self.flow, logger=self.logger, access_logger=self.access_logger, access_log=self.access_log, default_headers=self.server_state.default_headers, message_event=asyncio.Event(), expect_100_continue=self.expect_100_continue, keep_alive=http_version != '1.0', on_response=self.on_response_complete)
        if existing_cycle is None or existing_cycle.response_complete:
            task = self.loop.create_task(self.cycle.run_asgi(app))
            task.add_done_callback(self.tasks.discard)
            self.tasks.add(task)
        else:
            self.flow.pause_reading()
            self.pipeline.appendleft((self.cycle, app))

    def on_body(self, body: bytes) -> None:
        if False:
            print('Hello World!')
        if self.parser.should_upgrade() and self._should_upgrade() or self.cycle.response_complete:
            return
        self.cycle.body += body
        if len(self.cycle.body) > HIGH_WATER_LIMIT:
            self.flow.pause_reading()
        self.cycle.message_event.set()

    def on_message_complete(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.parser.should_upgrade() and self._should_upgrade() or self.cycle.response_complete:
            return
        self.cycle.more_body = False
        self.cycle.message_event.set()

    def on_response_complete(self) -> None:
        if False:
            i = 10
            return i + 15
        self.server_state.total_requests += 1
        if self.transport.is_closing():
            return
        self._unset_keepalive_if_required()
        self.timeout_keep_alive_task = self.loop.call_later(self.timeout_keep_alive, self.timeout_keep_alive_handler)
        self.flow.resume_reading()
        if self.pipeline:
            (cycle, app) = self.pipeline.pop()
            task = self.loop.create_task(cycle.run_asgi(app))
            task.add_done_callback(self.tasks.discard)
            self.tasks.add(task)

    def shutdown(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Called by the server to commence a graceful shutdown.\n        '
        if self.cycle is None or self.cycle.response_complete:
            self.transport.close()
        else:
            self.cycle.keep_alive = False

    def pause_writing(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Called by the transport when the write buffer exceeds the high water mark.\n        '
        self.flow.pause_writing()

    def resume_writing(self) -> None:
        if False:
            print('Hello World!')
        '\n        Called by the transport when the write buffer drops below the low water mark.\n        '
        self.flow.resume_writing()

    def timeout_keep_alive_handler(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Called on a keep-alive connection if no new data is received after a short\n        delay.\n        '
        if not self.transport.is_closing():
            self.transport.close()

class RequestResponseCycle:

    def __init__(self, scope: 'HTTPScope', transport: asyncio.Transport, flow: FlowControl, logger: logging.Logger, access_logger: logging.Logger, access_log: bool, default_headers: List[Tuple[bytes, bytes]], message_event: asyncio.Event, expect_100_continue: bool, keep_alive: bool, on_response: Callable[..., None]):
        if False:
            print('Hello World!')
        self.scope = scope
        self.transport = transport
        self.flow = flow
        self.logger = logger
        self.access_logger = access_logger
        self.access_log = access_log
        self.default_headers = default_headers
        self.message_event = message_event
        self.on_response = on_response
        self.disconnected = False
        self.keep_alive = keep_alive
        self.waiting_for_100_continue = expect_100_continue
        self.body = b''
        self.more_body = True
        self.response_started = False
        self.response_complete = False
        self.chunked_encoding: Optional[bool] = None
        self.expected_content_length = 0

    async def run_asgi(self, app: 'ASGI3Application') -> None:
        try:
            result = await app(self.scope, self.receive, self.send)
        except BaseException as exc:
            msg = 'Exception in ASGI application\n'
            self.logger.error(msg, exc_info=exc)
            if not self.response_started:
                await self.send_500_response()
            else:
                self.transport.close()
        else:
            if result is not None:
                msg = "ASGI callable should return None, but returned '%s'."
                self.logger.error(msg, result)
                self.transport.close()
            elif not self.response_started and (not self.disconnected):
                msg = 'ASGI callable returned without starting response.'
                self.logger.error(msg)
                await self.send_500_response()
            elif not self.response_complete and (not self.disconnected):
                msg = 'ASGI callable returned without completing response.'
                self.logger.error(msg)
                self.transport.close()
        finally:
            self.on_response = lambda : None

    async def send_500_response(self) -> None:
        response_start_event: 'HTTPResponseStartEvent' = {'type': 'http.response.start', 'status': 500, 'headers': [(b'content-type', b'text/plain; charset=utf-8'), (b'connection', b'close')]}
        await self.send(response_start_event)
        response_body_event: 'HTTPResponseBodyEvent' = {'type': 'http.response.body', 'body': b'Internal Server Error', 'more_body': False}
        await self.send(response_body_event)

    async def send(self, message: 'ASGISendEvent') -> None:
        message_type = message['type']
        if self.flow.write_paused and (not self.disconnected):
            await self.flow.drain()
        if self.disconnected:
            return
        if not self.response_started:
            if message_type != 'http.response.start':
                msg = "Expected ASGI message 'http.response.start', but got '%s'."
                raise RuntimeError(msg % message_type)
            message = cast('HTTPResponseStartEvent', message)
            self.response_started = True
            self.waiting_for_100_continue = False
            status_code = message['status']
            headers = self.default_headers + list(message.get('headers', []))
            if CLOSE_HEADER in self.scope['headers'] and CLOSE_HEADER not in headers:
                headers = headers + [CLOSE_HEADER]
            if self.access_log:
                self.access_logger.info('%s - "%s %s HTTP/%s" %d', get_client_addr(self.scope), self.scope['method'], get_path_with_query_string(self.scope), self.scope['http_version'], status_code)
            content = [STATUS_LINE[status_code]]
            for (name, value) in headers:
                if HEADER_RE.search(name):
                    raise RuntimeError('Invalid HTTP header name.')
                if HEADER_VALUE_RE.search(value):
                    raise RuntimeError('Invalid HTTP header value.')
                name = name.lower()
                if name == b'content-length' and self.chunked_encoding is None:
                    self.expected_content_length = int(value.decode())
                    self.chunked_encoding = False
                elif name == b'transfer-encoding' and value.lower() == b'chunked':
                    self.expected_content_length = 0
                    self.chunked_encoding = True
                elif name == b'connection' and value.lower() == b'close':
                    self.keep_alive = False
                content.extend([name, b': ', value, b'\r\n'])
            if self.chunked_encoding is None and self.scope['method'] != 'HEAD' and (status_code not in (204, 304)):
                self.chunked_encoding = True
                content.append(b'transfer-encoding: chunked\r\n')
            content.append(b'\r\n')
            self.transport.write(b''.join(content))
        elif not self.response_complete:
            if message_type != 'http.response.body':
                msg = "Expected ASGI message 'http.response.body', but got '%s'."
                raise RuntimeError(msg % message_type)
            body = cast(bytes, message.get('body', b''))
            more_body = message.get('more_body', False)
            if self.scope['method'] == 'HEAD':
                self.expected_content_length = 0
            elif self.chunked_encoding:
                if body:
                    content = [b'%x\r\n' % len(body), body, b'\r\n']
                else:
                    content = []
                if not more_body:
                    content.append(b'0\r\n\r\n')
                self.transport.write(b''.join(content))
            else:
                num_bytes = len(body)
                if num_bytes > self.expected_content_length:
                    raise RuntimeError('Response content longer than Content-Length')
                else:
                    self.expected_content_length -= num_bytes
                self.transport.write(body)
            if not more_body:
                if self.expected_content_length != 0:
                    raise RuntimeError('Response content shorter than Content-Length')
                self.response_complete = True
                self.message_event.set()
                if not self.keep_alive:
                    self.transport.close()
                self.on_response()
        else:
            msg = "Unexpected ASGI message '%s' sent, after response already completed."
            raise RuntimeError(msg % message_type)

    async def receive(self) -> 'ASGIReceiveEvent':
        if self.waiting_for_100_continue and (not self.transport.is_closing()):
            self.transport.write(b'HTTP/1.1 100 Continue\r\n\r\n')
            self.waiting_for_100_continue = False
        if not self.disconnected and (not self.response_complete):
            self.flow.resume_reading()
            await self.message_event.wait()
            self.message_event.clear()
        message: 'Union[HTTPDisconnectEvent, HTTPRequestEvent]'
        if self.disconnected or self.response_complete:
            message = {'type': 'http.disconnect'}
        else:
            message = {'type': 'http.request', 'body': self.body, 'more_body': self.more_body}
            self.body = b''
        return message