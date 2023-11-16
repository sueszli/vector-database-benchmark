from tornado.concurrent import Future
from tornado import gen
from tornado import netutil
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream, SSLIOStream, PipeIOStream, StreamClosedError, _StreamBuffer
from tornado.httputil import HTTPHeaders
from tornado.locks import Condition, Event
from tornado.log import gen_log
from tornado.netutil import ssl_options_to_context, ssl_wrap_socket
from tornado.platform.asyncio import AddThreadSelectorEventLoop
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncHTTPTestCase, AsyncHTTPSTestCase, AsyncTestCase, bind_unused_port, ExpectLog, gen_test
from tornado.test.util import skipIfNonUnix, refusing_port, skipPypy3V58, ignore_deprecation
from tornado.web import RequestHandler, Application
import asyncio
import errno
import hashlib
import logging
import os
import platform
import random
import socket
import ssl
import typing
from unittest import mock
import unittest

def _server_ssl_options():
    if False:
        i = 10
        return i + 15
    return dict(certfile=os.path.join(os.path.dirname(__file__), 'test.crt'), keyfile=os.path.join(os.path.dirname(__file__), 'test.key'))

class HelloHandler(RequestHandler):

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        self.write('Hello')

class TestIOStreamWebMixin(object):

    def _make_client_iostream(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def get_app(self):
        if False:
            print('Hello World!')
        return Application([('/', HelloHandler)])

    def test_connection_closed(self: typing.Any):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/', headers={'Connection': 'close'})
        response.rethrow()

    @gen_test
    def test_read_until_close(self: typing.Any):
        if False:
            while True:
                i = 10
        stream = self._make_client_iostream()
        yield stream.connect(('127.0.0.1', self.get_http_port()))
        stream.write(b'GET / HTTP/1.0\r\n\r\n')
        data = (yield stream.read_until_close())
        self.assertTrue(data.startswith(b'HTTP/1.1 200'))
        self.assertTrue(data.endswith(b'Hello'))

    @gen_test
    def test_read_zero_bytes(self: typing.Any):
        if False:
            for i in range(10):
                print('nop')
        self.stream = self._make_client_iostream()
        yield self.stream.connect(('127.0.0.1', self.get_http_port()))
        self.stream.write(b'GET / HTTP/1.0\r\n\r\n')
        data = (yield self.stream.read_bytes(9))
        self.assertEqual(data, b'HTTP/1.1 ')
        data = (yield self.stream.read_bytes(0))
        self.assertEqual(data, b'')
        data = (yield self.stream.read_bytes(3))
        self.assertEqual(data, b'200')
        self.stream.close()

    @gen_test
    def test_write_while_connecting(self: typing.Any):
        if False:
            return 10
        stream = self._make_client_iostream()
        connect_fut = stream.connect(('127.0.0.1', self.get_http_port()))
        write_fut = stream.write(b'GET / HTTP/1.0\r\nConnection: close\r\n\r\n')
        self.assertFalse(connect_fut.done())
        it = gen.WaitIterator(connect_fut, write_fut)
        resolved_order = []
        while not it.done():
            yield it.next()
            resolved_order.append(it.current_future)
        self.assertEqual(resolved_order, [connect_fut, write_fut])
        data = (yield stream.read_until_close())
        self.assertTrue(data.endswith(b'Hello'))
        stream.close()

    @gen_test
    def test_future_interface(self: typing.Any):
        if False:
            for i in range(10):
                print('nop')
        "Basic test of IOStream's ability to return Futures."
        stream = self._make_client_iostream()
        connect_result = (yield stream.connect(('127.0.0.1', self.get_http_port())))
        self.assertIs(connect_result, stream)
        yield stream.write(b'GET / HTTP/1.0\r\n\r\n')
        first_line = (yield stream.read_until(b'\r\n'))
        self.assertEqual(first_line, b'HTTP/1.1 200 OK\r\n')
        header_data = (yield stream.read_until(b'\r\n\r\n'))
        headers = HTTPHeaders.parse(header_data.decode('latin1'))
        content_length = int(headers['Content-Length'])
        body = (yield stream.read_bytes(content_length))
        self.assertEqual(body, b'Hello')
        stream.close()

    @gen_test
    def test_future_close_while_reading(self: typing.Any):
        if False:
            return 10
        stream = self._make_client_iostream()
        yield stream.connect(('127.0.0.1', self.get_http_port()))
        yield stream.write(b'GET / HTTP/1.0\r\n\r\n')
        with self.assertRaises(StreamClosedError):
            yield stream.read_bytes(1024 * 1024)
        stream.close()

    @gen_test
    def test_future_read_until_close(self: typing.Any):
        if False:
            print('Hello World!')
        stream = self._make_client_iostream()
        yield stream.connect(('127.0.0.1', self.get_http_port()))
        yield stream.write(b'GET / HTTP/1.0\r\nConnection: close\r\n\r\n')
        yield stream.read_until(b'\r\n\r\n')
        body = (yield stream.read_until_close())
        self.assertEqual(body, b'Hello')
        with self.assertRaises(StreamClosedError):
            stream.read_bytes(1)

class TestReadWriteMixin(object):

    def make_iostream_pair(self, **kwargs):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def iostream_pair(self, **kwargs):
        if False:
            print('Hello World!')
        'Like make_iostream_pair, but called by ``async with``.\n\n        In py37 this becomes simpler with contextlib.asynccontextmanager.\n        '

        class IOStreamPairContext:

            def __init__(self, test, kwargs):
                if False:
                    return 10
                self.test = test
                self.kwargs = kwargs

            async def __aenter__(self):
                self.pair = await self.test.make_iostream_pair(**self.kwargs)
                return self.pair

            async def __aexit__(self, typ, value, tb):
                for s in self.pair:
                    s.close()
        return IOStreamPairContext(self, kwargs)

    @gen_test
    def test_write_zero_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        (rs, ws) = (yield self.make_iostream_pair())
        yield ws.write(b'')
        ws.close()
        rs.close()

    @gen_test
    def test_future_delayed_close_callback(self: typing.Any):
        if False:
            while True:
                i = 10
        (rs, ws) = (yield self.make_iostream_pair())
        try:
            ws.write(b'12')
            chunks = []
            chunks.append((yield rs.read_bytes(1)))
            ws.close()
            chunks.append((yield rs.read_bytes(1)))
            self.assertEqual(chunks, [b'1', b'2'])
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_close_buffered_data(self: typing.Any):
        if False:
            print('Hello World!')
        (rs, ws) = (yield self.make_iostream_pair(read_chunk_size=256))
        try:
            ws.write(b'A' * 512)
            data = (yield rs.read_bytes(256))
            self.assertEqual(b'A' * 256, data)
            ws.close()
            yield gen.sleep(0.01)
            data = (yield rs.read_bytes(256))
            self.assertEqual(b'A' * 256, data)
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_close_after_close(self: typing.Any):
        if False:
            for i in range(10):
                print('nop')
        (rs, ws) = (yield self.make_iostream_pair())
        try:
            ws.write(b'1234')
            data = (yield rs.read_bytes(1))
            ws.close()
            self.assertEqual(data, b'1')
            data = (yield rs.read_until_close())
            self.assertEqual(data, b'234')
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_large_read_until(self: typing.Any):
        if False:
            print('Hello World!')
        (rs, ws) = (yield self.make_iostream_pair())
        try:
            if isinstance(rs, SSLIOStream) and platform.python_implementation() == 'PyPy':
                raise unittest.SkipTest('pypy gc causes problems with openssl')
            NUM_KB = 4096
            for i in range(NUM_KB):
                ws.write(b'A' * 1024)
            ws.write(b'\r\n')
            data = (yield rs.read_until(b'\r\n'))
            self.assertEqual(len(data), NUM_KB * 1024 + 2)
        finally:
            ws.close()
            rs.close()

    @gen_test
    async def test_read_until_with_close_after_second_packet(self):
        async with self.iostream_pair() as (rs, ws):
            rf = asyncio.ensure_future(rs.read_until(b'done'))
            await asyncio.sleep(0.1)
            await ws.write(b'x' * 2048)
            ws.write(b'done')
            ws.close()
            await rf

    @gen_test
    async def test_read_until_unsatisfied_after_close(self: typing.Any):
        async with self.iostream_pair() as (rs, ws):
            rf = asyncio.ensure_future(rs.read_until(b'done'))
            await ws.write(b'x' * 2048)
            ws.write(b'foo')
            ws.close()
            with self.assertRaises(StreamClosedError):
                await rf

    @gen_test
    def test_close_callback_with_pending_read(self: typing.Any):
        if False:
            i = 10
            return i + 15
        OK = b'OK\r\n'
        (rs, ws) = (yield self.make_iostream_pair())
        event = Event()
        rs.set_close_callback(event.set)
        try:
            ws.write(OK)
            res = (yield rs.read_until(b'\r\n'))
            self.assertEqual(res, OK)
            ws.close()
            rs.read_until(b'\r\n')
            yield event.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_future_close_callback(self: typing.Any):
        if False:
            while True:
                i = 10
        (rs, ws) = (yield self.make_iostream_pair())
        closed = [False]
        cond = Condition()

        def close_callback():
            if False:
                return 10
            closed[0] = True
            cond.notify()
        rs.set_close_callback(close_callback)
        try:
            ws.write(b'a')
            res = (yield rs.read_bytes(1))
            self.assertEqual(res, b'a')
            self.assertFalse(closed[0])
            ws.close()
            yield cond.wait()
            self.assertTrue(closed[0])
        finally:
            rs.close()
            ws.close()

    @gen_test
    def test_write_memoryview(self: typing.Any):
        if False:
            print('Hello World!')
        (rs, ws) = (yield self.make_iostream_pair())
        try:
            fut = rs.read_bytes(4)
            ws.write(memoryview(b'hello'))
            data = (yield fut)
            self.assertEqual(data, b'hell')
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_bytes_partial(self: typing.Any):
        if False:
            return 10
        (rs, ws) = (yield self.make_iostream_pair())
        try:
            fut = rs.read_bytes(50, partial=True)
            ws.write(b'hello')
            data = (yield fut)
            self.assertEqual(data, b'hello')
            fut = rs.read_bytes(3, partial=True)
            ws.write(b'world')
            data = (yield fut)
            self.assertEqual(data, b'wor')
            data = (yield rs.read_bytes(0, partial=True))
            self.assertEqual(data, b'')
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_max_bytes(self: typing.Any):
        if False:
            for i in range(10):
                print('nop')
        (rs, ws) = (yield self.make_iostream_pair())
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            fut = rs.read_until(b'def', max_bytes=50)
            ws.write(b'abcdef')
            data = (yield fut)
            self.assertEqual(data, b'abcdef')
            fut = rs.read_until(b'def', max_bytes=6)
            ws.write(b'abcdef')
            data = (yield fut)
            self.assertEqual(data, b'abcdef')
            with ExpectLog(gen_log, 'Unsatisfiable read', level=logging.INFO):
                fut = rs.read_until(b'def', max_bytes=5)
                ws.write(b'123456')
                yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_max_bytes_inline(self: typing.Any):
        if False:
            print('Hello World!')
        (rs, ws) = (yield self.make_iostream_pair())
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            ws.write(b'123456')
            with ExpectLog(gen_log, 'Unsatisfiable read', level=logging.INFO):
                with self.assertRaises(StreamClosedError):
                    yield rs.read_until(b'def', max_bytes=5)
            yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_max_bytes_ignores_extra(self: typing.Any):
        if False:
            for i in range(10):
                print('nop')
        (rs, ws) = (yield self.make_iostream_pair())
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            ws.write(b'abcdef')
            with ExpectLog(gen_log, 'Unsatisfiable read', level=logging.INFO):
                rs.read_until(b'def', max_bytes=5)
                yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_regex_max_bytes(self: typing.Any):
        if False:
            for i in range(10):
                print('nop')
        (rs, ws) = (yield self.make_iostream_pair())
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            fut = rs.read_until_regex(b'def', max_bytes=50)
            ws.write(b'abcdef')
            data = (yield fut)
            self.assertEqual(data, b'abcdef')
            fut = rs.read_until_regex(b'def', max_bytes=6)
            ws.write(b'abcdef')
            data = (yield fut)
            self.assertEqual(data, b'abcdef')
            with ExpectLog(gen_log, 'Unsatisfiable read', level=logging.INFO):
                rs.read_until_regex(b'def', max_bytes=5)
                ws.write(b'123456')
                yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_regex_max_bytes_inline(self: typing.Any):
        if False:
            for i in range(10):
                print('nop')
        (rs, ws) = (yield self.make_iostream_pair())
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            ws.write(b'123456')
            with ExpectLog(gen_log, 'Unsatisfiable read', level=logging.INFO):
                rs.read_until_regex(b'def', max_bytes=5)
                yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_regex_max_bytes_ignores_extra(self):
        if False:
            i = 10
            return i + 15
        (rs, ws) = (yield self.make_iostream_pair())
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            ws.write(b'abcdef')
            with ExpectLog(gen_log, 'Unsatisfiable read', level=logging.INFO):
                rs.read_until_regex(b'def', max_bytes=5)
                yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_small_reads_from_large_buffer(self: typing.Any):
        if False:
            i = 10
            return i + 15
        (rs, ws) = (yield self.make_iostream_pair(max_buffer_size=10 * 1024))
        try:
            ws.write(b'a' * 1024 * 100)
            for i in range(100):
                data = (yield rs.read_bytes(1024))
                self.assertEqual(data, b'a' * 1024)
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_small_read_untils_from_large_buffer(self: typing.Any):
        if False:
            print('Hello World!')
        (rs, ws) = (yield self.make_iostream_pair(max_buffer_size=10 * 1024))
        try:
            ws.write((b'a' * 1023 + b'\n') * 100)
            for i in range(100):
                data = (yield rs.read_until(b'\n', max_bytes=4096))
                self.assertEqual(data, b'a' * 1023 + b'\n')
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_flow_control(self):
        if False:
            i = 10
            return i + 15
        MB = 1024 * 1024
        (rs, ws) = (yield self.make_iostream_pair(max_buffer_size=5 * MB))
        try:
            ws.write(b'a' * 10 * MB)
            yield rs.read_bytes(MB)
            yield gen.sleep(0.1)
            for i in range(9):
                yield rs.read_bytes(MB)
        finally:
            rs.close()
            ws.close()

    @gen_test
    def test_read_into(self: typing.Any):
        if False:
            i = 10
            return i + 15
        (rs, ws) = (yield self.make_iostream_pair())

        def sleep_some():
            if False:
                return 10
            self.io_loop.run_sync(lambda : gen.sleep(0.05))
        try:
            buf = bytearray(10)
            fut = rs.read_into(buf)
            ws.write(b'hello')
            yield gen.sleep(0.05)
            self.assertTrue(rs.reading())
            ws.write(b'world!!')
            data = (yield fut)
            self.assertFalse(rs.reading())
            self.assertEqual(data, 10)
            self.assertEqual(bytes(buf), b'helloworld')
            fut = rs.read_into(buf)
            yield gen.sleep(0.05)
            self.assertTrue(rs.reading())
            ws.write(b'1234567890')
            data = (yield fut)
            self.assertFalse(rs.reading())
            self.assertEqual(data, 10)
            self.assertEqual(bytes(buf), b'!!12345678')
            buf = bytearray(4)
            ws.write(b'abcdefghi')
            data = (yield rs.read_into(buf))
            self.assertEqual(data, 4)
            self.assertEqual(bytes(buf), b'90ab')
            data = (yield rs.read_bytes(7))
            self.assertEqual(data, b'cdefghi')
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_into_partial(self: typing.Any):
        if False:
            i = 10
            return i + 15
        (rs, ws) = (yield self.make_iostream_pair())
        try:
            buf = bytearray(10)
            fut = rs.read_into(buf, partial=True)
            ws.write(b'hello')
            data = (yield fut)
            self.assertFalse(rs.reading())
            self.assertEqual(data, 5)
            self.assertEqual(bytes(buf), b'hello\x00\x00\x00\x00\x00')
            ws.write(b'world!1234567890')
            data = (yield rs.read_into(buf, partial=True))
            self.assertEqual(data, 10)
            self.assertEqual(bytes(buf), b'world!1234')
            data = (yield rs.read_into(buf, partial=True))
            self.assertEqual(data, 6)
            self.assertEqual(bytes(buf), b'5678901234')
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_into_zero_bytes(self: typing.Any):
        if False:
            print('Hello World!')
        (rs, ws) = (yield self.make_iostream_pair())
        try:
            buf = bytearray()
            fut = rs.read_into(buf)
            self.assertEqual(fut.result(), 0)
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_many_mixed_reads(self):
        if False:
            i = 10
            return i + 15
        r = random.Random(42)
        nbytes = 1000000
        (rs, ws) = (yield self.make_iostream_pair())
        produce_hash = hashlib.sha1()
        consume_hash = hashlib.sha1()

        @gen.coroutine
        def produce():
            if False:
                i = 10
                return i + 15
            remaining = nbytes
            while remaining > 0:
                size = r.randint(1, min(1000, remaining))
                data = os.urandom(size)
                produce_hash.update(data)
                yield ws.write(data)
                remaining -= size
            assert remaining == 0

        @gen.coroutine
        def consume():
            if False:
                for i in range(10):
                    print('nop')
            remaining = nbytes
            while remaining > 0:
                if r.random() > 0.5:
                    size = r.randint(1, min(1000, remaining))
                    data = (yield rs.read_bytes(size))
                    consume_hash.update(data)
                    remaining -= size
                else:
                    size = r.randint(1, min(1000, remaining))
                    buf = bytearray(size)
                    n = (yield rs.read_into(buf))
                    assert n == size
                    consume_hash.update(buf)
                    remaining -= size
            assert remaining == 0
        try:
            yield [produce(), consume()]
            assert produce_hash.hexdigest() == consume_hash.hexdigest()
        finally:
            ws.close()
            rs.close()

class TestIOStreamMixin(TestReadWriteMixin):

    def _make_server_iostream(self, connection, **kwargs):
        if False:
            return 10
        raise NotImplementedError()

    def _make_client_iostream(self, connection, **kwargs):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @gen.coroutine
    def make_iostream_pair(self: typing.Any, **kwargs):
        if False:
            return 10
        (listener, port) = bind_unused_port()
        server_stream_fut = Future()

        def accept_callback(connection, address):
            if False:
                i = 10
                return i + 15
            server_stream_fut.set_result(self._make_server_iostream(connection, **kwargs))
        netutil.add_accept_handler(listener, accept_callback)
        client_stream = self._make_client_iostream(socket.socket(), **kwargs)
        connect_fut = client_stream.connect(('127.0.0.1', port))
        (server_stream, client_stream) = (yield [server_stream_fut, connect_fut])
        self.io_loop.remove_handler(listener.fileno())
        listener.close()
        raise gen.Return((server_stream, client_stream))

    @gen_test
    def test_connection_refused(self: typing.Any):
        if False:
            return 10
        (cleanup_func, port) = refusing_port()
        self.addCleanup(cleanup_func)
        stream = IOStream(socket.socket())
        stream.set_close_callback(self.stop)
        with ExpectLog(gen_log, '.*', required=False):
            with self.assertRaises(StreamClosedError):
                yield stream.connect(('127.0.0.1', port))
        self.assertTrue(isinstance(stream.error, ConnectionRefusedError), stream.error)

    @gen_test
    def test_gaierror(self: typing.Any):
        if False:
            i = 10
            return i + 15
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        stream = IOStream(s)
        stream.set_close_callback(self.stop)
        with mock.patch('socket.socket.connect', side_effect=socket.gaierror(errno.EIO, 'boom')):
            with self.assertRaises(StreamClosedError):
                yield stream.connect(('localhost', 80))
            self.assertTrue(isinstance(stream.error, socket.gaierror))

    @gen_test
    def test_read_until_close_with_error(self: typing.Any):
        if False:
            return 10
        (server, client) = (yield self.make_iostream_pair())
        try:
            with mock.patch('tornado.iostream.BaseIOStream._try_inline_read', side_effect=IOError('boom')):
                with self.assertRaisesRegex(IOError, 'boom'):
                    client.read_until_close()
        finally:
            server.close()
            client.close()

    @skipIfNonUnix
    @skipPypy3V58
    @gen_test
    def test_inline_read_error(self: typing.Any):
        if False:
            for i in range(10):
                print('nop')
        io_loop = IOLoop.current()
        if isinstance(io_loop.selector_loop, AddThreadSelectorEventLoop):
            self.skipTest('AddThreadSelectorEventLoop not supported')
        (server, client) = (yield self.make_iostream_pair())
        try:
            os.close(server.socket.fileno())
            with self.assertRaises(socket.error):
                server.read_bytes(1)
        finally:
            server.close()
            client.close()

    @skipPypy3V58
    @gen_test
    def test_async_read_error_logging(self):
        if False:
            print('Hello World!')
        (server, client) = (yield self.make_iostream_pair())
        closed = Event()
        server.set_close_callback(closed.set)
        try:
            server.read_bytes(1)
            client.write(b'a')

            def fake_read_from_fd():
                if False:
                    for i in range(10):
                        print('nop')
                os.close(server.socket.fileno())
                server.__class__.read_from_fd(server)
            server.read_from_fd = fake_read_from_fd
            with ExpectLog(gen_log, 'error on read'):
                yield closed.wait()
        finally:
            server.close()
            client.close()

    @gen_test
    def test_future_write(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that write() Futures are never orphaned.\n        '
        (m, n) = (5000, 1000)
        nproducers = 10
        total_bytes = m * n * nproducers
        (server, client) = (yield self.make_iostream_pair(max_buffer_size=total_bytes))

        @gen.coroutine
        def produce():
            if False:
                while True:
                    i = 10
            data = b'x' * m
            for i in range(n):
                yield server.write(data)

        @gen.coroutine
        def consume():
            if False:
                i = 10
                return i + 15
            nread = 0
            while nread < total_bytes:
                res = (yield client.read_bytes(m))
                nread += len(res)
        try:
            yield ([produce() for i in range(nproducers)] + [consume()])
        finally:
            server.close()
            client.close()

class TestIOStreamWebHTTP(TestIOStreamWebMixin, AsyncHTTPTestCase):

    def _make_client_iostream(self):
        if False:
            for i in range(10):
                print('nop')
        return IOStream(socket.socket())

class TestIOStreamWebHTTPS(TestIOStreamWebMixin, AsyncHTTPSTestCase):

    def _make_client_iostream(self):
        if False:
            print('Hello World!')
        return SSLIOStream(socket.socket(), ssl_options=dict(cert_reqs=ssl.CERT_NONE))

class TestIOStream(TestIOStreamMixin, AsyncTestCase):

    def _make_server_iostream(self, connection, **kwargs):
        if False:
            return 10
        return IOStream(connection, **kwargs)

    def _make_client_iostream(self, connection, **kwargs):
        if False:
            while True:
                i = 10
        return IOStream(connection, **kwargs)

class TestIOStreamSSL(TestIOStreamMixin, AsyncTestCase):

    def _make_server_iostream(self, connection, **kwargs):
        if False:
            print('Hello World!')
        ssl_ctx = ssl_options_to_context(_server_ssl_options(), server_side=True)
        connection = ssl_ctx.wrap_socket(connection, server_side=True, do_handshake_on_connect=False)
        return SSLIOStream(connection, **kwargs)

    def _make_client_iostream(self, connection, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return SSLIOStream(connection, ssl_options=dict(cert_reqs=ssl.CERT_NONE), **kwargs)

class TestIOStreamSSLContext(TestIOStreamMixin, AsyncTestCase):

    def _make_server_iostream(self, connection, **kwargs):
        if False:
            i = 10
            return i + 15
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(os.path.join(os.path.dirname(__file__), 'test.crt'), os.path.join(os.path.dirname(__file__), 'test.key'))
        connection = ssl_wrap_socket(connection, context, server_side=True, do_handshake_on_connect=False)
        return SSLIOStream(connection, **kwargs)

    def _make_client_iostream(self, connection, **kwargs):
        if False:
            i = 10
            return i + 15
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return SSLIOStream(connection, ssl_options=context, **kwargs)

class TestIOStreamStartTLS(AsyncTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            super().setUp()
            (self.listener, self.port) = bind_unused_port()
            self.server_stream = None
            self.server_accepted = Future()
            netutil.add_accept_handler(self.listener, self.accept)
            self.client_stream = IOStream(socket.socket())
            self.io_loop.add_future(self.client_stream.connect(('127.0.0.1', self.port)), self.stop)
            self.wait()
            self.io_loop.add_future(self.server_accepted, self.stop)
            self.wait()
        except Exception as e:
            print(e)
            raise

    def tearDown(self):
        if False:
            while True:
                i = 10
        if self.server_stream is not None:
            self.server_stream.close()
        if self.client_stream is not None:
            self.client_stream.close()
        self.io_loop.remove_handler(self.listener.fileno())
        self.listener.close()
        super().tearDown()

    def accept(self, connection, address):
        if False:
            i = 10
            return i + 15
        if self.server_stream is not None:
            self.fail('should only get one connection')
        self.server_stream = IOStream(connection)
        self.server_accepted.set_result(None)

    @gen.coroutine
    def client_send_line(self, line):
        if False:
            i = 10
            return i + 15
        assert self.client_stream is not None
        self.client_stream.write(line)
        assert self.server_stream is not None
        recv_line = (yield self.server_stream.read_until(b'\r\n'))
        self.assertEqual(line, recv_line)

    @gen.coroutine
    def server_send_line(self, line):
        if False:
            for i in range(10):
                print('nop')
        assert self.server_stream is not None
        self.server_stream.write(line)
        assert self.client_stream is not None
        recv_line = (yield self.client_stream.read_until(b'\r\n'))
        self.assertEqual(line, recv_line)

    def client_start_tls(self, ssl_options=None, server_hostname=None):
        if False:
            for i in range(10):
                print('nop')
        assert self.client_stream is not None
        client_stream = self.client_stream
        self.client_stream = None
        return client_stream.start_tls(False, ssl_options, server_hostname)

    def server_start_tls(self, ssl_options=None):
        if False:
            i = 10
            return i + 15
        assert self.server_stream is not None
        server_stream = self.server_stream
        self.server_stream = None
        return server_stream.start_tls(True, ssl_options)

    @gen_test
    def test_start_tls_smtp(self):
        if False:
            i = 10
            return i + 15
        yield self.server_send_line(b'220 mail.example.com ready\r\n')
        yield self.client_send_line(b'EHLO mail.example.com\r\n')
        yield self.server_send_line(b'250-mail.example.com welcome\r\n')
        yield self.server_send_line(b'250 STARTTLS\r\n')
        yield self.client_send_line(b'STARTTLS\r\n')
        yield self.server_send_line(b'220 Go ahead\r\n')
        client_future = self.client_start_tls(dict(cert_reqs=ssl.CERT_NONE))
        server_future = self.server_start_tls(_server_ssl_options())
        self.client_stream = (yield client_future)
        self.server_stream = (yield server_future)
        self.assertTrue(isinstance(self.client_stream, SSLIOStream))
        self.assertTrue(isinstance(self.server_stream, SSLIOStream))
        yield self.client_send_line(b'EHLO mail.example.com\r\n')
        yield self.server_send_line(b'250 mail.example.com welcome\r\n')

    @gen_test
    def test_handshake_fail(self):
        if False:
            i = 10
            return i + 15
        server_future = self.server_start_tls(_server_ssl_options())
        with ExpectLog(gen_log, 'SSL Error'):
            client_future = self.client_start_tls(server_hostname='localhost')
            with self.assertRaises(ssl.SSLError):
                yield client_future
            with self.assertRaises((ssl.SSLError, socket.error)):
                yield server_future

    @gen_test
    def test_check_hostname(self):
        if False:
            i = 10
            return i + 15
        server_future = self.server_start_tls(_server_ssl_options())
        with ExpectLog(gen_log, 'SSL Error'):
            client_future = self.client_start_tls(ssl.create_default_context(), server_hostname='127.0.0.1')
            with self.assertRaises(ssl.SSLError):
                yield client_future
            with self.assertRaises(Exception):
                yield server_future

    @gen_test
    def test_typed_memoryview(self):
        if False:
            while True:
                i = 10
        buf = memoryview(bytes(80)).cast('L')
        assert self.server_stream is not None
        yield self.server_stream.write(buf)
        assert self.client_stream is not None
        recv = (yield self.client_stream.read_bytes(buf.nbytes))
        self.assertEqual(bytes(recv), bytes(buf))

class WaitForHandshakeTest(AsyncTestCase):

    @gen.coroutine
    def connect_to_server(self, server_cls):
        if False:
            print('Hello World!')
        server = client = None
        try:
            (sock, port) = bind_unused_port()
            server = server_cls(ssl_options=_server_ssl_options())
            server.add_socket(sock)
            ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            with ignore_deprecation():
                ssl_ctx.options |= getattr(ssl, 'OP_NO_TLSv1_3', 0)
                client = SSLIOStream(socket.socket(), ssl_options=ssl_ctx)
            yield client.connect(('127.0.0.1', port))
            self.assertIsNotNone(client.socket.cipher())
        finally:
            if server is not None:
                server.stop()
            if client is not None:
                client.close()

    @gen_test
    def test_wait_for_handshake_future(self):
        if False:
            i = 10
            return i + 15
        test = self
        handshake_future = Future()

        class TestServer(TCPServer):

            def handle_stream(self, stream, address):
                if False:
                    i = 10
                    return i + 15
                test.assertIsNone(stream.socket.cipher())
                test.io_loop.spawn_callback(self.handle_connection, stream)

            @gen.coroutine
            def handle_connection(self, stream):
                if False:
                    print('Hello World!')
                yield stream.wait_for_handshake()
                handshake_future.set_result(None)
        yield self.connect_to_server(TestServer)
        yield handshake_future

    @gen_test
    def test_wait_for_handshake_already_waiting_error(self):
        if False:
            while True:
                i = 10
        test = self
        handshake_future = Future()

        class TestServer(TCPServer):

            @gen.coroutine
            def handle_stream(self, stream, address):
                if False:
                    print('Hello World!')
                fut = stream.wait_for_handshake()
                test.assertRaises(RuntimeError, stream.wait_for_handshake)
                yield fut
                handshake_future.set_result(None)
        yield self.connect_to_server(TestServer)
        yield handshake_future

    @gen_test
    def test_wait_for_handshake_already_connected(self):
        if False:
            i = 10
            return i + 15
        handshake_future = Future()

        class TestServer(TCPServer):

            @gen.coroutine
            def handle_stream(self, stream, address):
                if False:
                    while True:
                        i = 10
                yield stream.wait_for_handshake()
                yield stream.wait_for_handshake()
                handshake_future.set_result(None)
        yield self.connect_to_server(TestServer)
        yield handshake_future

class TestIOStreamCheckHostname(AsyncTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        (self.listener, self.port) = bind_unused_port()

        def accept_callback(connection, address):
            if False:
                for i in range(10):
                    print('nop')
            ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_ctx.load_cert_chain(os.path.join(os.path.dirname(__file__), 'test.crt'), os.path.join(os.path.dirname(__file__), 'test.key'))
            connection = ssl_ctx.wrap_socket(connection, server_side=True, do_handshake_on_connect=False)
            SSLIOStream(connection)
        netutil.add_accept_handler(self.listener, accept_callback)
        self.client_ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        self.client_ssl_ctx.load_verify_locations(os.path.join(os.path.dirname(__file__), 'test.crt'))

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.io_loop.remove_handler(self.listener.fileno())
        self.listener.close()
        super().tearDown()

    @gen_test
    async def test_match(self):
        stream = SSLIOStream(socket.socket(), ssl_options=self.client_ssl_ctx)
        await stream.connect(('127.0.0.1', self.port), server_hostname='foo.example.com')
        stream.close()

    @gen_test
    async def test_no_match(self):
        stream = SSLIOStream(socket.socket(), ssl_options=self.client_ssl_ctx)
        with ExpectLog(gen_log, '.*alert bad certificate', level=logging.WARNING, required=platform.system() != 'Windows'):
            with self.assertRaises(ssl.SSLCertVerificationError):
                with ExpectLog(gen_log, '.*(certificate verify failed: Hostname mismatch)', level=logging.WARNING):
                    await stream.connect(('127.0.0.1', self.port), server_hostname='bar.example.com')
            if platform.system() != 'Windows':
                await asyncio.sleep(0.1)

    @gen_test
    async def test_check_disabled(self):
        self.client_ssl_ctx.check_hostname = False
        stream = SSLIOStream(socket.socket(), ssl_options=self.client_ssl_ctx)
        await stream.connect(('127.0.0.1', self.port), server_hostname='bar.example.com')

@skipIfNonUnix
class TestPipeIOStream(TestReadWriteMixin, AsyncTestCase):

    @gen.coroutine
    def make_iostream_pair(self, **kwargs):
        if False:
            return 10
        (r, w) = os.pipe()
        return (PipeIOStream(r, **kwargs), PipeIOStream(w, **kwargs))

    @gen_test
    def test_pipe_iostream(self):
        if False:
            return 10
        (rs, ws) = (yield self.make_iostream_pair())
        ws.write(b'hel')
        ws.write(b'lo world')
        data = (yield rs.read_until(b' '))
        self.assertEqual(data, b'hello ')
        data = (yield rs.read_bytes(3))
        self.assertEqual(data, b'wor')
        ws.close()
        data = (yield rs.read_until_close())
        self.assertEqual(data, b'ld')
        rs.close()

    @gen_test
    def test_pipe_iostream_big_write(self):
        if False:
            while True:
                i = 10
        (rs, ws) = (yield self.make_iostream_pair())
        NUM_BYTES = 1048576
        ws.write(b'1' * NUM_BYTES)
        data = (yield rs.read_bytes(NUM_BYTES))
        self.assertEqual(data, b'1' * NUM_BYTES)
        ws.close()
        rs.close()

class TestStreamBuffer(unittest.TestCase):
    """
    Unit tests for the private _StreamBuffer class.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.random = random.Random(42)

    def to_bytes(self, b):
        if False:
            return 10
        if isinstance(b, (bytes, bytearray)):
            return bytes(b)
        elif isinstance(b, memoryview):
            return b.tobytes()
        else:
            raise TypeError(b)

    def make_streambuffer(self, large_buf_threshold=10):
        if False:
            for i in range(10):
                print('nop')
        buf = _StreamBuffer()
        assert buf._large_buf_threshold
        buf._large_buf_threshold = large_buf_threshold
        return buf

    def check_peek(self, buf, expected):
        if False:
            while True:
                i = 10
        size = 1
        while size < 2 * len(expected):
            got = self.to_bytes(buf.peek(size))
            self.assertTrue(got)
            self.assertLessEqual(len(got), size)
            self.assertTrue(expected.startswith(got), (expected, got))
            size = (size * 3 + 1) // 2

    def check_append_all_then_skip_all(self, buf, objs, input_type):
        if False:
            return 10
        self.assertEqual(len(buf), 0)
        expected = b''
        for o in objs:
            expected += o
            buf.append(input_type(o))
            self.assertEqual(len(buf), len(expected))
            self.check_peek(buf, expected)
        while expected:
            n = self.random.randrange(1, len(expected) + 1)
            expected = expected[n:]
            buf.advance(n)
            self.assertEqual(len(buf), len(expected))
            self.check_peek(buf, expected)
        self.assertEqual(len(buf), 0)

    def test_small(self):
        if False:
            while True:
                i = 10
        objs = [b'12', b'345', b'67', b'89a', b'bcde', b'fgh', b'ijklmn']
        buf = self.make_streambuffer()
        self.check_append_all_then_skip_all(buf, objs, bytes)
        buf = self.make_streambuffer()
        self.check_append_all_then_skip_all(buf, objs, bytearray)
        buf = self.make_streambuffer()
        self.check_append_all_then_skip_all(buf, objs, memoryview)
        buf = self.make_streambuffer(10)
        for i in range(9):
            buf.append(b'x')
        self.assertEqual(len(buf._buffers), 1)
        for i in range(9):
            buf.append(b'x')
        self.assertEqual(len(buf._buffers), 2)
        buf.advance(10)
        self.assertEqual(len(buf._buffers), 1)
        buf.advance(8)
        self.assertEqual(len(buf._buffers), 0)
        self.assertEqual(len(buf), 0)

    def test_large(self):
        if False:
            i = 10
            return i + 15
        objs = [b'12' * 5, b'345' * 2, b'67' * 20, b'89a' * 12, b'bcde' * 1, b'fgh' * 7, b'ijklmn' * 2]
        buf = self.make_streambuffer()
        self.check_append_all_then_skip_all(buf, objs, bytes)
        buf = self.make_streambuffer()
        self.check_append_all_then_skip_all(buf, objs, bytearray)
        buf = self.make_streambuffer()
        self.check_append_all_then_skip_all(buf, objs, memoryview)
        buf = self.make_streambuffer(10)
        for i in range(3):
            buf.append(b'x' * 11)
        self.assertEqual(len(buf._buffers), 3)
        buf.append(b'y')
        self.assertEqual(len(buf._buffers), 4)
        buf.append(b'z')
        self.assertEqual(len(buf._buffers), 4)
        buf.advance(33)
        self.assertEqual(len(buf._buffers), 1)
        buf.advance(2)
        self.assertEqual(len(buf._buffers), 0)
        self.assertEqual(len(buf), 0)