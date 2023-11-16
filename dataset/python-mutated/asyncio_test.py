import asyncio
import threading
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from tornado import gen
from tornado.ioloop import IOLoop
from tornado.platform.asyncio import AsyncIOLoop, to_asyncio_future, AnyThreadEventLoopPolicy, AddThreadSelectorEventLoop
from tornado.testing import AsyncTestCase, gen_test

class AsyncIOLoopTest(AsyncTestCase):

    @property
    def asyncio_loop(self):
        if False:
            for i in range(10):
                print('nop')
        return self.io_loop.asyncio_loop

    def test_asyncio_callback(self):
        if False:
            return 10

        async def add_callback():
            asyncio.get_event_loop().call_soon(self.stop)
        self.asyncio_loop.run_until_complete(add_callback())
        self.wait()

    @gen_test
    def test_asyncio_future(self):
        if False:
            print('Hello World!')
        x = (yield asyncio.ensure_future(asyncio.get_event_loop().run_in_executor(None, lambda : 42)))
        self.assertEqual(x, 42)

    @gen_test
    def test_asyncio_yield_from(self):
        if False:
            for i in range(10):
                print('nop')

        @gen.coroutine
        def f():
            if False:
                for i in range(10):
                    print('nop')
            event_loop = asyncio.get_event_loop()
            x = (yield from event_loop.run_in_executor(None, lambda : 42))
            return x
        result = (yield f())
        self.assertEqual(result, 42)

    def test_asyncio_adapter(self):
        if False:
            print('Hello World!')

        @gen.coroutine
        def tornado_coroutine():
            if False:
                while True:
                    i = 10
            yield gen.moment
            raise gen.Return(42)

        async def native_coroutine_without_adapter():
            return await tornado_coroutine()

        async def native_coroutine_with_adapter():
            return await to_asyncio_future(tornado_coroutine())

        async def native_coroutine_with_adapter2():
            return await to_asyncio_future(native_coroutine_without_adapter())
        self.assertEqual(self.io_loop.run_sync(native_coroutine_without_adapter), 42)
        self.assertEqual(self.io_loop.run_sync(native_coroutine_with_adapter), 42)
        self.assertEqual(self.io_loop.run_sync(native_coroutine_with_adapter2), 42)
        self.assertEqual(self.asyncio_loop.run_until_complete(native_coroutine_without_adapter()), 42)
        self.assertEqual(self.asyncio_loop.run_until_complete(native_coroutine_with_adapter()), 42)
        self.assertEqual(self.asyncio_loop.run_until_complete(native_coroutine_with_adapter2()), 42)

    def test_add_thread_close_idempotent(self):
        if False:
            i = 10
            return i + 15
        loop = AddThreadSelectorEventLoop(asyncio.get_event_loop())
        loop.close()
        loop.close()

class LeakTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        AsyncIOLoop(make_current=False).close()
        self.orig_policy = asyncio.get_event_loop_policy()
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    def tearDown(self):
        if False:
            return 10
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except Exception:
            pass
        else:
            loop.close()
        asyncio.set_event_loop_policy(self.orig_policy)

    def test_ioloop_close_leak(self):
        if False:
            while True:
                i = 10
        orig_count = len(IOLoop._ioloop_for_asyncio)
        for i in range(10):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                loop = AsyncIOLoop()
                loop.close()
        new_count = len(IOLoop._ioloop_for_asyncio) - orig_count
        self.assertEqual(new_count, 0)

    def test_asyncio_close_leak(self):
        if False:
            i = 10
            return i + 15
        orig_count = len(IOLoop._ioloop_for_asyncio)
        for i in range(10):
            loop = asyncio.new_event_loop()
            loop.call_soon(IOLoop.current)
            loop.call_soon(loop.stop)
            loop.run_forever()
            loop.close()
        new_count = len(IOLoop._ioloop_for_asyncio) - orig_count
        self.assertEqual(new_count, 1)

class SelectorThreadLeakTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        asyncio.run(self.dummy_tornado_coroutine())
        self.orig_thread_count = threading.active_count()

    def assert_no_thread_leak(self):
        if False:
            while True:
                i = 10
        deadline = time.time() + 1
        while time.time() < deadline:
            threads = list(threading.enumerate())
            if len(threads) <= self.orig_thread_count:
                break
            time.sleep(0.1)
        self.assertLessEqual(len(threads), self.orig_thread_count, threads)

    async def dummy_tornado_coroutine(self):
        IOLoop.current()

    def test_asyncio_run(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(10):
            asyncio.run(self.dummy_tornado_coroutine())
        self.assert_no_thread_leak()

    def test_asyncio_manual(self):
        if False:
            i = 10
            return i + 15
        for i in range(10):
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.dummy_tornado_coroutine())
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        self.assert_no_thread_leak()

    def test_tornado(self):
        if False:
            print('Hello World!')
        for i in range(10):
            loop = IOLoop(make_current=False)
            loop.run_sync(self.dummy_tornado_coroutine)
            loop.close()
        self.assert_no_thread_leak()

class AnyThreadEventLoopPolicyTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.orig_policy = asyncio.get_event_loop_policy()
        self.executor = ThreadPoolExecutor(1)

    def tearDown(self):
        if False:
            print('Hello World!')
        asyncio.set_event_loop_policy(self.orig_policy)
        self.executor.shutdown()

    def get_event_loop_on_thread(self):
        if False:
            for i in range(10):
                print('nop')

        def get_and_close_event_loop():
            if False:
                while True:
                    i = 10
            "Get the event loop. Close it if one is returned.\n\n            Returns the (closed) event loop. This is a silly thing\n            to do and leaves the thread in a broken state, but it's\n            enough for this test. Closing the loop avoids resource\n            leak warnings.\n            "
            loop = asyncio.get_event_loop()
            loop.close()
            return loop
        future = self.executor.submit(get_and_close_event_loop)
        return future.result()

    def test_asyncio_accessor(self):
        if False:
            for i in range(10):
                print('nop')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            self.assertRaises(RuntimeError, self.executor.submit(asyncio.get_event_loop).result)
            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
            self.assertIsInstance(self.executor.submit(asyncio.get_event_loop).result(), asyncio.AbstractEventLoop)
            self.executor.submit(lambda : asyncio.get_event_loop().close()).result()

    def test_tornado_accessor(self):
        if False:
            return 10
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            self.assertIsInstance(self.executor.submit(IOLoop.current).result(), IOLoop)
            self.executor.submit(lambda : asyncio.get_event_loop().close()).result()
            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
            self.assertIsInstance(self.executor.submit(IOLoop.current).result(), IOLoop)
            self.executor.submit(lambda : asyncio.get_event_loop().close()).result()