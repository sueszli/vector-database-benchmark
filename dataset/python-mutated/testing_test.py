from tornado import gen, ioloop
from tornado.httpserver import HTTPServer
from tornado.locks import Event
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, bind_unused_port, gen_test
from tornado.web import Application
import asyncio
import contextlib
import inspect
import gc
import os
import platform
import sys
import traceback
import unittest
import warnings

@contextlib.contextmanager
def set_environ(name, value):
    if False:
        while True:
            i = 10
    old_value = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            del os.environ[name]
        else:
            os.environ[name] = old_value

class AsyncTestCaseTest(AsyncTestCase):

    def test_wait_timeout(self):
        if False:
            while True:
                i = 10
        time = self.io_loop.time
        self.io_loop.add_timeout(time() + 0.01, self.stop)
        self.wait()
        self.io_loop.add_timeout(time() + 1, self.stop)
        with self.assertRaises(self.failureException):
            self.wait(timeout=0.01)
        self.io_loop.add_timeout(time() + 1, self.stop)
        with set_environ('ASYNC_TEST_TIMEOUT', '0.01'):
            with self.assertRaises(self.failureException):
                self.wait()

    def test_subsequent_wait_calls(self):
        if False:
            i = 10
            return i + 15
        '\n        This test makes sure that a second call to wait()\n        clears the first timeout.\n        '
        self.io_loop.add_timeout(self.io_loop.time() + 0.0, self.stop)
        self.wait(timeout=0.1)
        self.io_loop.add_timeout(self.io_loop.time() + 0.2, self.stop)
        self.wait(timeout=0.4)

class LeakTest(AsyncTestCase):

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        gc.collect()

    def test_leaked_coroutine(self):
        if False:
            return 10
        event = Event()

        async def callback():
            try:
                await event.wait()
            except asyncio.CancelledError:
                pass
        self.io_loop.add_callback(callback)
        self.io_loop.add_callback(self.stop)
        self.wait()

class AsyncHTTPTestCaseTest(AsyncHTTPTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        (sock, port) = bind_unused_port()
        app = Application()
        server = HTTPServer(app, **self.get_httpserver_options())
        server.add_socket(sock)
        self.second_port = port
        self.second_server = server

    def get_app(self):
        if False:
            for i in range(10):
                print('nop')
        return Application()

    def test_fetch_segment(self):
        if False:
            print('Hello World!')
        path = '/path'
        response = self.fetch(path)
        self.assertEqual(response.request.url, self.get_url(path))

    def test_fetch_full_http_url(self):
        if False:
            while True:
                i = 10
        path = 'http://127.0.0.1:%d/path' % self.second_port
        response = self.fetch(path)
        self.assertEqual(response.request.url, path)

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.second_server.stop()
        super().tearDown()

class AsyncTestCaseWrapperTest(unittest.TestCase):

    def test_undecorated_generator(self):
        if False:
            print('Hello World!')

        class Test(AsyncTestCase):

            def test_gen(self):
                if False:
                    for i in range(10):
                        print('nop')
                yield
        test = Test('test_gen')
        result = unittest.TestResult()
        test.run(result)
        self.assertEqual(len(result.errors), 1)
        self.assertIn('should be decorated', result.errors[0][1])

    @unittest.skipIf(platform.python_implementation() == 'PyPy', 'pypy destructor warnings cannot be silenced')
    @unittest.skipIf(sys.version_info >= (3, 12), 'py312 has its own check for test case returns')
    def test_undecorated_coroutine(self):
        if False:
            i = 10
            return i + 15

        class Test(AsyncTestCase):

            async def test_coro(self):
                pass
        test = Test('test_coro')
        result = unittest.TestResult()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            test.run(result)
        self.assertEqual(len(result.errors), 1)
        self.assertIn('should be decorated', result.errors[0][1])

    def test_undecorated_generator_with_skip(self):
        if False:
            for i in range(10):
                print('nop')

        class Test(AsyncTestCase):

            @unittest.skip("don't run this")
            def test_gen(self):
                if False:
                    return 10
                yield
        test = Test('test_gen')
        result = unittest.TestResult()
        test.run(result)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.skipped), 1)

    def test_other_return(self):
        if False:
            while True:
                i = 10

        class Test(AsyncTestCase):

            def test_other_return(self):
                if False:
                    i = 10
                    return i + 15
                return 42
        test = Test('test_other_return')
        result = unittest.TestResult()
        test.run(result)
        self.assertEqual(len(result.errors), 1)
        self.assertIn('Return value from test method ignored', result.errors[0][1])

    def test_unwrap(self):
        if False:
            for i in range(10):
                print('nop')

        class Test(AsyncTestCase):

            def test_foo(self):
                if False:
                    i = 10
                    return i + 15
                pass
        test = Test('test_foo')
        self.assertIs(inspect.unwrap(test.test_foo), test.test_foo.orig_method)

class SetUpTearDownTest(unittest.TestCase):

    def test_set_up_tear_down(self):
        if False:
            return 10
        '\n        This test makes sure that AsyncTestCase calls super methods for\n        setUp and tearDown.\n\n        InheritBoth is a subclass of both AsyncTestCase and\n        SetUpTearDown, with the ordering so that the super of\n        AsyncTestCase will be SetUpTearDown.\n        '
        events = []
        result = unittest.TestResult()

        class SetUpTearDown(unittest.TestCase):

            def setUp(self):
                if False:
                    return 10
                events.append('setUp')

            def tearDown(self):
                if False:
                    while True:
                        i = 10
                events.append('tearDown')

        class InheritBoth(AsyncTestCase, SetUpTearDown):

            def test(self):
                if False:
                    return 10
                events.append('test')
        InheritBoth('test').run(result)
        expected = ['setUp', 'test', 'tearDown']
        self.assertEqual(expected, events)

class AsyncHTTPTestCaseSetUpTearDownTest(unittest.TestCase):

    def test_tear_down_releases_app_and_http_server(self):
        if False:
            while True:
                i = 10
        result = unittest.TestResult()

        class SetUpTearDown(AsyncHTTPTestCase):

            def get_app(self):
                if False:
                    print('Hello World!')
                return Application()

            def test(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.assertTrue(hasattr(self, '_app'))
                self.assertTrue(hasattr(self, 'http_server'))
        test = SetUpTearDown('test')
        test.run(result)
        self.assertFalse(hasattr(test, '_app'))
        self.assertFalse(hasattr(test, 'http_server'))

class GenTest(AsyncTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.finished = False

    def tearDown(self):
        if False:
            return 10
        self.assertTrue(self.finished)
        super().tearDown()

    @gen_test
    def test_sync(self):
        if False:
            for i in range(10):
                print('nop')
        self.finished = True

    @gen_test
    def test_async(self):
        if False:
            while True:
                i = 10
        yield gen.moment
        self.finished = True

    def test_timeout(self):
        if False:
            print('Hello World!')

        @gen_test(timeout=0.1)
        def test(self):
            if False:
                return 10
            yield gen.sleep(1)
        try:
            test(self)
            self.fail('did not get expected exception')
        except ioloop.TimeoutError:
            self.assertIn('gen.sleep(1)', traceback.format_exc())
        self.finished = True

    def test_no_timeout(self):
        if False:
            for i in range(10):
                print('nop')

        @gen_test(timeout=1)
        def test(self):
            if False:
                for i in range(10):
                    print('nop')
            yield gen.sleep(0.1)
        test(self)
        self.finished = True

    def test_timeout_environment_variable(self):
        if False:
            i = 10
            return i + 15

        @gen_test(timeout=0.5)
        def test_long_timeout(self):
            if False:
                print('Hello World!')
            yield gen.sleep(0.25)
        with set_environ('ASYNC_TEST_TIMEOUT', '0.1'):
            test_long_timeout(self)
        self.finished = True

    def test_no_timeout_environment_variable(self):
        if False:
            print('Hello World!')

        @gen_test(timeout=0.01)
        def test_short_timeout(self):
            if False:
                while True:
                    i = 10
            yield gen.sleep(1)
        with set_environ('ASYNC_TEST_TIMEOUT', '0.1'):
            with self.assertRaises(ioloop.TimeoutError):
                test_short_timeout(self)
        self.finished = True

    def test_with_method_args(self):
        if False:
            for i in range(10):
                print('nop')

        @gen_test
        def test_with_args(self, *args):
            if False:
                print('Hello World!')
            self.assertEqual(args, ('test',))
            yield gen.moment
        test_with_args(self, 'test')
        self.finished = True

    def test_with_method_kwargs(self):
        if False:
            i = 10
            return i + 15

        @gen_test
        def test_with_kwargs(self, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            self.assertDictEqual(kwargs, {'test': 'test'})
            yield gen.moment
        test_with_kwargs(self, test='test')
        self.finished = True

    def test_native_coroutine(self):
        if False:
            i = 10
            return i + 15

        @gen_test
        async def test(self):
            self.finished = True
        test(self)

    def test_native_coroutine_timeout(self):
        if False:
            for i in range(10):
                print('nop')

        @gen_test(timeout=0.1)
        async def test(self):
            await gen.sleep(1)
        try:
            test(self)
            self.fail('did not get expected exception')
        except ioloop.TimeoutError:
            self.finished = True
if __name__ == '__main__':
    unittest.main()