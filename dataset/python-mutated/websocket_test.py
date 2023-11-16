import asyncio
import contextlib
import functools
import socket
import traceback
import typing
import unittest
from tornado.concurrent import Future
from tornado import gen
from tornado.httpclient import HTTPError, HTTPRequest
from tornado.locks import Event
from tornado.log import gen_log, app_log
from tornado.netutil import Resolver
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.template import DictLoader
from tornado.testing import AsyncHTTPTestCase, gen_test, bind_unused_port, ExpectLog
from tornado.web import Application, RequestHandler
try:
    import tornado.websocket
    from tornado.util import _websocket_mask_python
except ImportError:
    traceback.print_exc()
    raise
from tornado.websocket import WebSocketHandler, websocket_connect, WebSocketError, WebSocketClosedError
try:
    from tornado import speedups
except ImportError:
    speedups = None

class TestWebSocketHandler(WebSocketHandler):
    """Base class for testing handlers that exposes the on_close event.

    This allows for tests to see the close code and reason on the
    server side.

    """

    def initialize(self, close_future=None, compression_options=None):
        if False:
            print('Hello World!')
        self.close_future = close_future
        self.compression_options = compression_options

    def get_compression_options(self):
        if False:
            for i in range(10):
                print('nop')
        return self.compression_options

    def on_close(self):
        if False:
            for i in range(10):
                print('nop')
        if self.close_future is not None:
            self.close_future.set_result((self.close_code, self.close_reason))

class EchoHandler(TestWebSocketHandler):

    @gen.coroutine
    def on_message(self, message):
        if False:
            for i in range(10):
                print('nop')
        try:
            yield self.write_message(message, isinstance(message, bytes))
        except asyncio.CancelledError:
            pass
        except WebSocketClosedError:
            pass

class ErrorInOnMessageHandler(TestWebSocketHandler):

    def on_message(self, message):
        if False:
            return 10
        1 / 0

class HeaderHandler(TestWebSocketHandler):

    def open(self):
        if False:
            return 10
        methods_to_test = [functools.partial(self.write, 'This should not work'), functools.partial(self.redirect, 'http://localhost/elsewhere'), functools.partial(self.set_header, 'X-Test', ''), functools.partial(self.set_cookie, 'Chocolate', 'Chip'), functools.partial(self.set_status, 503), self.flush, self.finish]
        for method in methods_to_test:
            try:
                method()
                raise Exception('did not get expected exception')
            except RuntimeError:
                pass
        self.write_message(self.request.headers.get('X-Test', ''))

class HeaderEchoHandler(TestWebSocketHandler):

    def set_default_headers(self):
        if False:
            print('Hello World!')
        self.set_header('X-Extra-Response-Header', 'Extra-Response-Value')

    def prepare(self):
        if False:
            while True:
                i = 10
        for (k, v) in self.request.headers.get_all():
            if k.lower().startswith('x-test'):
                self.set_header(k, v)

class NonWebSocketHandler(RequestHandler):

    def get(self):
        if False:
            print('Hello World!')
        self.write('ok')

class RedirectHandler(RequestHandler):

    def get(self):
        if False:
            i = 10
            return i + 15
        self.redirect('/echo')

class CloseReasonHandler(TestWebSocketHandler):

    def open(self):
        if False:
            print('Hello World!')
        self.on_close_called = False
        self.close(1001, 'goodbye')

class AsyncPrepareHandler(TestWebSocketHandler):

    @gen.coroutine
    def prepare(self):
        if False:
            print('Hello World!')
        yield gen.moment

    def on_message(self, message):
        if False:
            print('Hello World!')
        self.write_message(message)

class PathArgsHandler(TestWebSocketHandler):

    def open(self, arg):
        if False:
            for i in range(10):
                print('nop')
        self.write_message(arg)

class CoroutineOnMessageHandler(TestWebSocketHandler):

    def initialize(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super().initialize(**kwargs)
        self.sleeping = 0

    @gen.coroutine
    def on_message(self, message):
        if False:
            print('Hello World!')
        if self.sleeping > 0:
            self.write_message('another coroutine is already sleeping')
        self.sleeping += 1
        yield gen.sleep(0.01)
        self.sleeping -= 1
        self.write_message(message)

class RenderMessageHandler(TestWebSocketHandler):

    def on_message(self, message):
        if False:
            return 10
        self.write_message(self.render_string('message.html', message=message))

class SubprotocolHandler(TestWebSocketHandler):

    def initialize(self, **kwargs):
        if False:
            print('Hello World!')
        super().initialize(**kwargs)
        self.select_subprotocol_called = False

    def select_subprotocol(self, subprotocols):
        if False:
            for i in range(10):
                print('nop')
        if self.select_subprotocol_called:
            raise Exception('select_subprotocol called twice')
        self.select_subprotocol_called = True
        if 'goodproto' in subprotocols:
            return 'goodproto'
        return None

    def open(self):
        if False:
            while True:
                i = 10
        if not self.select_subprotocol_called:
            raise Exception('select_subprotocol not called')
        self.write_message('subprotocol=%s' % self.selected_subprotocol)

class OpenCoroutineHandler(TestWebSocketHandler):

    def initialize(self, test, **kwargs):
        if False:
            print('Hello World!')
        super().initialize(**kwargs)
        self.test = test
        self.open_finished = False

    @gen.coroutine
    def open(self):
        if False:
            i = 10
            return i + 15
        yield self.test.message_sent.wait()
        yield gen.sleep(0.01)
        self.open_finished = True

    def on_message(self, message):
        if False:
            return 10
        if not self.open_finished:
            raise Exception('on_message called before open finished')
        self.write_message('ok')

class ErrorInOpenHandler(TestWebSocketHandler):

    def open(self):
        if False:
            return 10
        raise Exception('boom')

class ErrorInAsyncOpenHandler(TestWebSocketHandler):

    async def open(self):
        await asyncio.sleep(0)
        raise Exception('boom')

class NoDelayHandler(TestWebSocketHandler):

    def open(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_nodelay(True)
        self.write_message('hello')

class WebSocketBaseTestCase(AsyncHTTPTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.conns_to_close = []

    def tearDown(self):
        if False:
            return 10
        for conn in self.conns_to_close:
            conn.close()
        super().tearDown()

    @gen.coroutine
    def ws_connect(self, path, **kwargs):
        if False:
            i = 10
            return i + 15
        ws = (yield websocket_connect('ws://127.0.0.1:%d%s' % (self.get_http_port(), path), **kwargs))
        self.conns_to_close.append(ws)
        raise gen.Return(ws)

class WebSocketTest(WebSocketBaseTestCase):

    def get_app(self):
        if False:
            for i in range(10):
                print('nop')
        self.close_future = Future()
        return Application([('/echo', EchoHandler, dict(close_future=self.close_future)), ('/non_ws', NonWebSocketHandler), ('/redirect', RedirectHandler), ('/header', HeaderHandler, dict(close_future=self.close_future)), ('/header_echo', HeaderEchoHandler, dict(close_future=self.close_future)), ('/close_reason', CloseReasonHandler, dict(close_future=self.close_future)), ('/error_in_on_message', ErrorInOnMessageHandler, dict(close_future=self.close_future)), ('/async_prepare', AsyncPrepareHandler, dict(close_future=self.close_future)), ('/path_args/(.*)', PathArgsHandler, dict(close_future=self.close_future)), ('/coroutine', CoroutineOnMessageHandler, dict(close_future=self.close_future)), ('/render', RenderMessageHandler, dict(close_future=self.close_future)), ('/subprotocol', SubprotocolHandler, dict(close_future=self.close_future)), ('/open_coroutine', OpenCoroutineHandler, dict(close_future=self.close_future, test=self)), ('/error_in_open', ErrorInOpenHandler), ('/error_in_async_open', ErrorInAsyncOpenHandler), ('/nodelay', NoDelayHandler)], template_loader=DictLoader({'message.html': '<b>{{ message }}</b>'}))

    def get_http_client(self):
        if False:
            i = 10
            return i + 15
        return SimpleAsyncHTTPClient()

    def tearDown(self):
        if False:
            return 10
        super().tearDown()
        RequestHandler._template_loaders.clear()

    def test_http_request(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/echo')
        self.assertEqual(response.code, 400)

    def test_missing_websocket_key(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/echo', headers={'Connection': 'Upgrade', 'Upgrade': 'WebSocket', 'Sec-WebSocket-Version': '13'})
        self.assertEqual(response.code, 400)

    def test_bad_websocket_version(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/echo', headers={'Connection': 'Upgrade', 'Upgrade': 'WebSocket', 'Sec-WebSocket-Version': '12'})
        self.assertEqual(response.code, 426)

    @gen_test
    def test_websocket_gen(self):
        if False:
            while True:
                i = 10
        ws = (yield self.ws_connect('/echo'))
        yield ws.write_message('hello')
        response = (yield ws.read_message())
        self.assertEqual(response, 'hello')

    def test_websocket_callbacks(self):
        if False:
            for i in range(10):
                print('nop')
        websocket_connect('ws://127.0.0.1:%d/echo' % self.get_http_port(), callback=self.stop)
        ws = self.wait().result()
        ws.write_message('hello')
        ws.read_message(self.stop)
        response = self.wait().result()
        self.assertEqual(response, 'hello')
        self.close_future.add_done_callback(lambda f: self.stop())
        ws.close()
        self.wait()

    @gen_test
    def test_binary_message(self):
        if False:
            while True:
                i = 10
        ws = (yield self.ws_connect('/echo'))
        ws.write_message(b'hello \xe9', binary=True)
        response = (yield ws.read_message())
        self.assertEqual(response, b'hello \xe9')

    @gen_test
    def test_unicode_message(self):
        if False:
            i = 10
            return i + 15
        ws = (yield self.ws_connect('/echo'))
        ws.write_message('hello é')
        response = (yield ws.read_message())
        self.assertEqual(response, 'hello é')

    @gen_test
    def test_error_in_closed_client_write_message(self):
        if False:
            i = 10
            return i + 15
        ws = (yield self.ws_connect('/echo'))
        ws.close()
        with self.assertRaises(WebSocketClosedError):
            ws.write_message('hello é')

    @gen_test
    def test_render_message(self):
        if False:
            return 10
        ws = (yield self.ws_connect('/render'))
        ws.write_message('hello')
        response = (yield ws.read_message())
        self.assertEqual(response, '<b>hello</b>')

    @gen_test
    def test_error_in_on_message(self):
        if False:
            for i in range(10):
                print('nop')
        ws = (yield self.ws_connect('/error_in_on_message'))
        ws.write_message('hello')
        with ExpectLog(app_log, 'Uncaught exception'):
            response = (yield ws.read_message())
        self.assertIs(response, None)

    @gen_test
    def test_websocket_http_fail(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(HTTPError) as cm:
            yield self.ws_connect('/notfound')
        self.assertEqual(cm.exception.code, 404)

    @gen_test
    def test_websocket_http_success(self):
        if False:
            return 10
        with self.assertRaises(WebSocketError):
            yield self.ws_connect('/non_ws')

    @gen_test
    def test_websocket_http_redirect(self):
        if False:
            return 10
        with self.assertRaises(HTTPError):
            yield self.ws_connect('/redirect')

    @gen_test
    def test_websocket_network_fail(self):
        if False:
            print('Hello World!')
        (sock, port) = bind_unused_port()
        sock.close()
        with self.assertRaises(IOError):
            with ExpectLog(gen_log, '.*', required=False):
                yield websocket_connect('ws://127.0.0.1:%d/' % port, connect_timeout=3600)

    @gen_test
    def test_websocket_close_buffered_data(self):
        if False:
            i = 10
            return i + 15
        with contextlib.closing((yield websocket_connect('ws://127.0.0.1:%d/echo' % self.get_http_port()))) as ws:
            ws.write_message('hello')
            ws.write_message('world')
            ws.stream.close()

    @gen_test
    def test_websocket_headers(self):
        if False:
            while True:
                i = 10
        with contextlib.closing((yield websocket_connect(HTTPRequest('ws://127.0.0.1:%d/header' % self.get_http_port(), headers={'X-Test': 'hello'})))) as ws:
            response = (yield ws.read_message())
            self.assertEqual(response, 'hello')

    @gen_test
    def test_websocket_header_echo(self):
        if False:
            return 10
        with contextlib.closing((yield websocket_connect(HTTPRequest('ws://127.0.0.1:%d/header_echo' % self.get_http_port(), headers={'X-Test-Hello': 'hello'})))) as ws:
            self.assertEqual(ws.headers.get('X-Test-Hello'), 'hello')
            self.assertEqual(ws.headers.get('X-Extra-Response-Header'), 'Extra-Response-Value')

    @gen_test
    def test_server_close_reason(self):
        if False:
            for i in range(10):
                print('nop')
        ws = (yield self.ws_connect('/close_reason'))
        msg = (yield ws.read_message())
        self.assertIs(msg, None)
        self.assertEqual(ws.close_code, 1001)
        self.assertEqual(ws.close_reason, 'goodbye')
        (code, reason) = (yield self.close_future)
        self.assertEqual(code, 1001)

    @gen_test
    def test_client_close_reason(self):
        if False:
            i = 10
            return i + 15
        ws = (yield self.ws_connect('/echo'))
        ws.close(1001, 'goodbye')
        (code, reason) = (yield self.close_future)
        self.assertEqual(code, 1001)
        self.assertEqual(reason, 'goodbye')

    @gen_test
    def test_write_after_close(self):
        if False:
            print('Hello World!')
        ws = (yield self.ws_connect('/close_reason'))
        msg = (yield ws.read_message())
        self.assertIs(msg, None)
        with self.assertRaises(WebSocketClosedError):
            ws.write_message('hello')

    @gen_test
    def test_async_prepare(self):
        if False:
            print('Hello World!')
        ws = (yield self.ws_connect('/async_prepare'))
        ws.write_message('hello')
        res = (yield ws.read_message())
        self.assertEqual(res, 'hello')

    @gen_test
    def test_path_args(self):
        if False:
            for i in range(10):
                print('nop')
        ws = (yield self.ws_connect('/path_args/hello'))
        res = (yield ws.read_message())
        self.assertEqual(res, 'hello')

    @gen_test
    def test_coroutine(self):
        if False:
            print('Hello World!')
        ws = (yield self.ws_connect('/coroutine'))
        yield ws.write_message('hello1')
        yield ws.write_message('hello2')
        res = (yield ws.read_message())
        self.assertEqual(res, 'hello1')
        res = (yield ws.read_message())
        self.assertEqual(res, 'hello2')

    @gen_test
    def test_check_origin_valid_no_path(self):
        if False:
            for i in range(10):
                print('nop')
        port = self.get_http_port()
        url = 'ws://127.0.0.1:%d/echo' % port
        headers = {'Origin': 'http://127.0.0.1:%d' % port}
        with contextlib.closing((yield websocket_connect(HTTPRequest(url, headers=headers)))) as ws:
            ws.write_message('hello')
            response = (yield ws.read_message())
            self.assertEqual(response, 'hello')

    @gen_test
    def test_check_origin_valid_with_path(self):
        if False:
            for i in range(10):
                print('nop')
        port = self.get_http_port()
        url = 'ws://127.0.0.1:%d/echo' % port
        headers = {'Origin': 'http://127.0.0.1:%d/something' % port}
        with contextlib.closing((yield websocket_connect(HTTPRequest(url, headers=headers)))) as ws:
            ws.write_message('hello')
            response = (yield ws.read_message())
            self.assertEqual(response, 'hello')

    @gen_test
    def test_check_origin_invalid_partial_url(self):
        if False:
            return 10
        port = self.get_http_port()
        url = 'ws://127.0.0.1:%d/echo' % port
        headers = {'Origin': '127.0.0.1:%d' % port}
        with self.assertRaises(HTTPError) as cm:
            yield websocket_connect(HTTPRequest(url, headers=headers))
        self.assertEqual(cm.exception.code, 403)

    @gen_test
    def test_check_origin_invalid(self):
        if False:
            i = 10
            return i + 15
        port = self.get_http_port()
        url = 'ws://127.0.0.1:%d/echo' % port
        headers = {'Origin': 'http://somewhereelse.com'}
        with self.assertRaises(HTTPError) as cm:
            yield websocket_connect(HTTPRequest(url, headers=headers))
        self.assertEqual(cm.exception.code, 403)

    @gen_test
    def test_check_origin_invalid_subdomains(self):
        if False:
            return 10
        port = self.get_http_port()
        addrinfo = (yield Resolver().resolve('localhost', port))
        families = set((addr[0] for addr in addrinfo))
        if socket.AF_INET not in families:
            self.skipTest('localhost does not resolve to ipv4')
            return
        url = 'ws://localhost:%d/echo' % port
        headers = {'Origin': 'http://subtenant.localhost'}
        with self.assertRaises(HTTPError) as cm:
            yield websocket_connect(HTTPRequest(url, headers=headers))
        self.assertEqual(cm.exception.code, 403)

    @gen_test
    def test_subprotocols(self):
        if False:
            return 10
        ws = (yield self.ws_connect('/subprotocol', subprotocols=['badproto', 'goodproto']))
        self.assertEqual(ws.selected_subprotocol, 'goodproto')
        res = (yield ws.read_message())
        self.assertEqual(res, 'subprotocol=goodproto')

    @gen_test
    def test_subprotocols_not_offered(self):
        if False:
            for i in range(10):
                print('nop')
        ws = (yield self.ws_connect('/subprotocol'))
        self.assertIs(ws.selected_subprotocol, None)
        res = (yield ws.read_message())
        self.assertEqual(res, 'subprotocol=None')

    @gen_test
    def test_open_coroutine(self):
        if False:
            while True:
                i = 10
        self.message_sent = Event()
        ws = (yield self.ws_connect('/open_coroutine'))
        yield ws.write_message('hello')
        self.message_sent.set()
        res = (yield ws.read_message())
        self.assertEqual(res, 'ok')

    @gen_test
    def test_error_in_open(self):
        if False:
            print('Hello World!')
        with ExpectLog(app_log, 'Uncaught exception'):
            ws = (yield self.ws_connect('/error_in_open'))
            res = (yield ws.read_message())
        self.assertIsNone(res)

    @gen_test
    def test_error_in_async_open(self):
        if False:
            return 10
        with ExpectLog(app_log, 'Uncaught exception'):
            ws = (yield self.ws_connect('/error_in_async_open'))
            res = (yield ws.read_message())
        self.assertIsNone(res)

    @gen_test
    def test_nodelay(self):
        if False:
            i = 10
            return i + 15
        ws = (yield self.ws_connect('/nodelay'))
        res = (yield ws.read_message())
        self.assertEqual(res, 'hello')

class NativeCoroutineOnMessageHandler(TestWebSocketHandler):

    def initialize(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().initialize(**kwargs)
        self.sleeping = 0

    async def on_message(self, message):
        if self.sleeping > 0:
            self.write_message('another coroutine is already sleeping')
        self.sleeping += 1
        await gen.sleep(0.01)
        self.sleeping -= 1
        self.write_message(message)

class WebSocketNativeCoroutineTest(WebSocketBaseTestCase):

    def get_app(self):
        if False:
            return 10
        return Application([('/native', NativeCoroutineOnMessageHandler)])

    @gen_test
    def test_native_coroutine(self):
        if False:
            print('Hello World!')
        ws = (yield self.ws_connect('/native'))
        yield ws.write_message('hello1')
        yield ws.write_message('hello2')
        res = (yield ws.read_message())
        self.assertEqual(res, 'hello1')
        res = (yield ws.read_message())
        self.assertEqual(res, 'hello2')

class CompressionTestMixin(object):
    MESSAGE = 'Hello world. Testing 123 123'

    def get_app(self):
        if False:
            for i in range(10):
                print('nop')

        class LimitedHandler(TestWebSocketHandler):

            @property
            def max_message_size(self):
                if False:
                    print('Hello World!')
                return 1024

            def on_message(self, message):
                if False:
                    i = 10
                    return i + 15
                self.write_message(str(len(message)))
        return Application([('/echo', EchoHandler, dict(compression_options=self.get_server_compression_options())), ('/limited', LimitedHandler, dict(compression_options=self.get_server_compression_options()))])

    def get_server_compression_options(self):
        if False:
            while True:
                i = 10
        return None

    def get_client_compression_options(self):
        if False:
            i = 10
            return i + 15
        return None

    def verify_wire_bytes(self, bytes_in: int, bytes_out: int) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @gen_test
    def test_message_sizes(self: typing.Any):
        if False:
            print('Hello World!')
        ws = (yield self.ws_connect('/echo', compression_options=self.get_client_compression_options()))
        for i in range(3):
            ws.write_message(self.MESSAGE)
            response = (yield ws.read_message())
            self.assertEqual(response, self.MESSAGE)
        self.assertEqual(ws.protocol._message_bytes_out, len(self.MESSAGE) * 3)
        self.assertEqual(ws.protocol._message_bytes_in, len(self.MESSAGE) * 3)
        self.verify_wire_bytes(ws.protocol._wire_bytes_in, ws.protocol._wire_bytes_out)

    @gen_test
    def test_size_limit(self: typing.Any):
        if False:
            while True:
                i = 10
        ws = (yield self.ws_connect('/limited', compression_options=self.get_client_compression_options()))
        ws.write_message('a' * 128)
        response = (yield ws.read_message())
        self.assertEqual(response, '128')
        ws.write_message('a' * 2048)
        response = (yield ws.read_message())
        self.assertIsNone(response)

class UncompressedTestMixin(CompressionTestMixin):
    """Specialization of CompressionTestMixin when we expect no compression."""

    def verify_wire_bytes(self: typing.Any, bytes_in, bytes_out):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(bytes_out, 3 * (len(self.MESSAGE) + 6))
        self.assertEqual(bytes_in, 3 * (len(self.MESSAGE) + 2))

class NoCompressionTest(UncompressedTestMixin, WebSocketBaseTestCase):
    pass

class ServerOnlyCompressionTest(UncompressedTestMixin, WebSocketBaseTestCase):

    def get_server_compression_options(self):
        if False:
            for i in range(10):
                print('nop')
        return {}

class ClientOnlyCompressionTest(UncompressedTestMixin, WebSocketBaseTestCase):

    def get_client_compression_options(self):
        if False:
            return 10
        return {}

class DefaultCompressionTest(CompressionTestMixin, WebSocketBaseTestCase):

    def get_server_compression_options(self):
        if False:
            while True:
                i = 10
        return {}

    def get_client_compression_options(self):
        if False:
            i = 10
            return i + 15
        return {}

    def verify_wire_bytes(self, bytes_in, bytes_out):
        if False:
            i = 10
            return i + 15
        self.assertLess(bytes_out, 3 * (len(self.MESSAGE) + 6))
        self.assertLess(bytes_in, 3 * (len(self.MESSAGE) + 2))
        self.assertEqual(bytes_out, bytes_in + 12)

class MaskFunctionMixin(object):

    def mask(self, mask: bytes, data: bytes) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def test_mask(self: typing.Any):
        if False:
            return 10
        self.assertEqual(self.mask(b'abcd', b''), b'')
        self.assertEqual(self.mask(b'abcd', b'b'), b'\x03')
        self.assertEqual(self.mask(b'abcd', b'54321'), b'TVPVP')
        self.assertEqual(self.mask(b'ZXCV', b'98765432'), b'c`t`olpd')
        self.assertEqual(self.mask(b'\x00\x01\x02\x03', b'\xff\xfb\xfd\xfc\xfe\xfa'), b'\xff\xfa\xff\xff\xfe\xfb')
        self.assertEqual(self.mask(b'\xff\xfb\xfd\xfc', b'\x00\x01\x02\x03\x04\x05'), b'\xff\xfa\xff\xff\xfb\xfe')

class PythonMaskFunctionTest(MaskFunctionMixin, unittest.TestCase):

    def mask(self, mask, data):
        if False:
            i = 10
            return i + 15
        return _websocket_mask_python(mask, data)

@unittest.skipIf(speedups is None, 'tornado.speedups module not present')
class CythonMaskFunctionTest(MaskFunctionMixin, unittest.TestCase):

    def mask(self, mask, data):
        if False:
            print('Hello World!')
        return speedups.websocket_mask(mask, data)

class ServerPeriodicPingTest(WebSocketBaseTestCase):

    def get_app(self):
        if False:
            i = 10
            return i + 15

        class PingHandler(TestWebSocketHandler):

            def on_pong(self, data):
                if False:
                    while True:
                        i = 10
                self.write_message('got pong')
        return Application([('/', PingHandler)], websocket_ping_interval=0.01)

    @gen_test
    def test_server_ping(self):
        if False:
            i = 10
            return i + 15
        ws = (yield self.ws_connect('/'))
        for i in range(3):
            response = (yield ws.read_message())
            self.assertEqual(response, 'got pong')

class ClientPeriodicPingTest(WebSocketBaseTestCase):

    def get_app(self):
        if False:
            i = 10
            return i + 15

        class PingHandler(TestWebSocketHandler):

            def on_ping(self, data):
                if False:
                    return 10
                self.write_message('got ping')
        return Application([('/', PingHandler)])

    @gen_test
    def test_client_ping(self):
        if False:
            print('Hello World!')
        ws = (yield self.ws_connect('/', ping_interval=0.01))
        for i in range(3):
            response = (yield ws.read_message())
            self.assertEqual(response, 'got ping')
        ws.close()

class ManualPingTest(WebSocketBaseTestCase):

    def get_app(self):
        if False:
            for i in range(10):
                print('nop')

        class PingHandler(TestWebSocketHandler):

            def on_ping(self, data):
                if False:
                    i = 10
                    return i + 15
                self.write_message(data, binary=isinstance(data, bytes))
        return Application([('/', PingHandler)])

    @gen_test
    def test_manual_ping(self):
        if False:
            return 10
        ws = (yield self.ws_connect('/'))
        self.assertRaises(ValueError, ws.ping, 'a' * 126)
        ws.ping('hello')
        resp = (yield ws.read_message())
        self.assertEqual(resp, b'hello')
        ws.ping(b'binary hello')
        resp = (yield ws.read_message())
        self.assertEqual(resp, b'binary hello')

class MaxMessageSizeTest(WebSocketBaseTestCase):

    def get_app(self):
        if False:
            i = 10
            return i + 15
        return Application([('/', EchoHandler)], websocket_max_message_size=1024)

    @gen_test
    def test_large_message(self):
        if False:
            print('Hello World!')
        ws = (yield self.ws_connect('/'))
        msg = 'a' * 1024
        ws.write_message(msg)
        resp = (yield ws.read_message())
        self.assertEqual(resp, msg)
        ws.write_message(msg + 'b')
        resp = (yield ws.read_message())
        self.assertIs(resp, None)
        self.assertEqual(ws.close_code, 1009)
        self.assertEqual(ws.close_reason, 'message too big')