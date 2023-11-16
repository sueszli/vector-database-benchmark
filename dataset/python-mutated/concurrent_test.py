from concurrent import futures
import logging
import re
import socket
import typing
import unittest
from tornado.concurrent import Future, run_on_executor, future_set_result_unless_cancelled
from tornado.escape import utf8, to_unicode
from tornado import gen
from tornado.iostream import IOStream
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncTestCase, bind_unused_port, gen_test

class MiscFutureTest(AsyncTestCase):

    def test_future_set_result_unless_cancelled(self):
        if False:
            print('Hello World!')
        fut = Future()
        future_set_result_unless_cancelled(fut, 42)
        self.assertEqual(fut.result(), 42)
        self.assertFalse(fut.cancelled())
        fut = Future()
        fut.cancel()
        is_cancelled = fut.cancelled()
        future_set_result_unless_cancelled(fut, 42)
        self.assertEqual(fut.cancelled(), is_cancelled)
        if not is_cancelled:
            self.assertEqual(fut.result(), 42)

class CapServer(TCPServer):

    @gen.coroutine
    def handle_stream(self, stream, address):
        if False:
            for i in range(10):
                print('nop')
        data = (yield stream.read_until(b'\n'))
        data = to_unicode(data)
        if data == data.upper():
            stream.write(b'error\talready capitalized\n')
        else:
            stream.write(utf8('ok\t%s' % data.upper()))
        stream.close()

class CapError(Exception):
    pass

class BaseCapClient(object):

    def __init__(self, port):
        if False:
            for i in range(10):
                print('nop')
        self.port = port

    def process_response(self, data):
        if False:
            return 10
        m = re.match('(.*)\t(.*)\n', to_unicode(data))
        if m is None:
            raise Exception('did not match')
        (status, message) = m.groups()
        if status == 'ok':
            return message
        else:
            raise CapError(message)

class GeneratorCapClient(BaseCapClient):

    @gen.coroutine
    def capitalize(self, request_data):
        if False:
            print('Hello World!')
        logging.debug('capitalize')
        stream = IOStream(socket.socket())
        logging.debug('connecting')
        yield stream.connect(('127.0.0.1', self.port))
        stream.write(utf8(request_data + '\n'))
        logging.debug('reading')
        data = (yield stream.read_until(b'\n'))
        logging.debug('returning')
        stream.close()
        raise gen.Return(self.process_response(data))

class ClientTestMixin(object):
    client_class = None

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.server = CapServer()
        (sock, port) = bind_unused_port()
        self.server.add_sockets([sock])
        self.client = self.client_class(port=port)

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.server.stop()
        super().tearDown()

    def test_future(self: typing.Any):
        if False:
            return 10
        future = self.client.capitalize('hello')
        self.io_loop.add_future(future, self.stop)
        self.wait()
        self.assertEqual(future.result(), 'HELLO')

    def test_future_error(self: typing.Any):
        if False:
            return 10
        future = self.client.capitalize('HELLO')
        self.io_loop.add_future(future, self.stop)
        self.wait()
        self.assertRaisesRegex(CapError, 'already capitalized', future.result)

    def test_generator(self: typing.Any):
        if False:
            return 10

        @gen.coroutine
        def f():
            if False:
                for i in range(10):
                    print('nop')
            result = (yield self.client.capitalize('hello'))
            self.assertEqual(result, 'HELLO')
        self.io_loop.run_sync(f)

    def test_generator_error(self: typing.Any):
        if False:
            print('Hello World!')

        @gen.coroutine
        def f():
            if False:
                for i in range(10):
                    print('nop')
            with self.assertRaisesRegex(CapError, 'already capitalized'):
                yield self.client.capitalize('HELLO')
        self.io_loop.run_sync(f)

class GeneratorClientTest(ClientTestMixin, AsyncTestCase):
    client_class = GeneratorCapClient

class RunOnExecutorTest(AsyncTestCase):

    @gen_test
    def test_no_calling(self):
        if False:
            i = 10
            return i + 15

        class Object(object):

            def __init__(self):
                if False:
                    return 10
                self.executor = futures.thread.ThreadPoolExecutor(1)

            @run_on_executor
            def f(self):
                if False:
                    while True:
                        i = 10
                return 42
        o = Object()
        answer = (yield o.f())
        self.assertEqual(answer, 42)

    @gen_test
    def test_call_with_no_args(self):
        if False:
            print('Hello World!')

        class Object(object):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.executor = futures.thread.ThreadPoolExecutor(1)

            @run_on_executor()
            def f(self):
                if False:
                    i = 10
                    return i + 15
                return 42
        o = Object()
        answer = (yield o.f())
        self.assertEqual(answer, 42)

    @gen_test
    def test_call_with_executor(self):
        if False:
            return 10

        class Object(object):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.__executor = futures.thread.ThreadPoolExecutor(1)

            @run_on_executor(executor='_Object__executor')
            def f(self):
                if False:
                    while True:
                        i = 10
                return 42
        o = Object()
        answer = (yield o.f())
        self.assertEqual(answer, 42)

    @gen_test
    def test_async_await(self):
        if False:
            print('Hello World!')

        class Object(object):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.executor = futures.thread.ThreadPoolExecutor(1)

            @run_on_executor()
            def f(self):
                if False:
                    i = 10
                    return i + 15
                return 42
        o = Object()

        async def f():
            answer = await o.f()
            return answer
        result = (yield f())
        self.assertEqual(result, 42)
if __name__ == '__main__':
    unittest.main()