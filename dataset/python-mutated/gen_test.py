import asyncio
from concurrent import futures
import gc
import datetime
import platform
import sys
import time
import weakref
import unittest
from tornado.concurrent import Future
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import skipOnTravis, skipNotCPython
from tornado.web import Application, RequestHandler, HTTPError
from tornado import gen
try:
    import contextvars
except ImportError:
    contextvars = None
import typing
if typing.TYPE_CHECKING:
    from typing import List, Optional

class GenBasicTest(AsyncTestCase):

    @gen.coroutine
    def delay(self, iterations, arg):
        if False:
            for i in range(10):
                print('nop')
        'Returns arg after a number of IOLoop iterations.'
        for i in range(iterations):
            yield gen.moment
        raise gen.Return(arg)

    @gen.coroutine
    def async_future(self, result):
        if False:
            i = 10
            return i + 15
        yield gen.moment
        return result

    @gen.coroutine
    def async_exception(self, e):
        if False:
            for i in range(10):
                print('nop')
        yield gen.moment
        raise e

    @gen.coroutine
    def add_one_async(self, x):
        if False:
            return 10
        yield gen.moment
        raise gen.Return(x + 1)

    def test_no_yield(self):
        if False:
            print('Hello World!')

        @gen.coroutine
        def f():
            if False:
                i = 10
                return i + 15
            pass
        self.io_loop.run_sync(f)

    def test_exception_phase1(self):
        if False:
            i = 10
            return i + 15

        @gen.coroutine
        def f():
            if False:
                for i in range(10):
                    print('nop')
            1 / 0
        self.assertRaises(ZeroDivisionError, self.io_loop.run_sync, f)

    def test_exception_phase2(self):
        if False:
            i = 10
            return i + 15

        @gen.coroutine
        def f():
            if False:
                return 10
            yield gen.moment
            1 / 0
        self.assertRaises(ZeroDivisionError, self.io_loop.run_sync, f)

    def test_bogus_yield(self):
        if False:
            for i in range(10):
                print('nop')

        @gen.coroutine
        def f():
            if False:
                i = 10
                return i + 15
            yield 42
        self.assertRaises(gen.BadYieldError, self.io_loop.run_sync, f)

    def test_bogus_yield_tuple(self):
        if False:
            print('Hello World!')

        @gen.coroutine
        def f():
            if False:
                for i in range(10):
                    print('nop')
            yield (1, 2)
        self.assertRaises(gen.BadYieldError, self.io_loop.run_sync, f)

    def test_reuse(self):
        if False:
            for i in range(10):
                print('nop')

        @gen.coroutine
        def f():
            if False:
                return 10
            yield gen.moment
        self.io_loop.run_sync(f)
        self.io_loop.run_sync(f)

    def test_none(self):
        if False:
            i = 10
            return i + 15

        @gen.coroutine
        def f():
            if False:
                for i in range(10):
                    print('nop')
            yield None
        self.io_loop.run_sync(f)

    def test_multi(self):
        if False:
            i = 10
            return i + 15

        @gen.coroutine
        def f():
            if False:
                while True:
                    i = 10
            results = (yield [self.add_one_async(1), self.add_one_async(2)])
            self.assertEqual(results, [2, 3])
        self.io_loop.run_sync(f)

    def test_multi_dict(self):
        if False:
            while True:
                i = 10

        @gen.coroutine
        def f():
            if False:
                return 10
            results = (yield dict(foo=self.add_one_async(1), bar=self.add_one_async(2)))
            self.assertEqual(results, dict(foo=2, bar=3))
        self.io_loop.run_sync(f)

    def test_multi_delayed(self):
        if False:
            i = 10
            return i + 15

        @gen.coroutine
        def f():
            if False:
                i = 10
                return i + 15
            responses = (yield gen.multi_future([self.delay(3, 'v1'), self.delay(1, 'v2')]))
            self.assertEqual(responses, ['v1', 'v2'])
        self.io_loop.run_sync(f)

    def test_multi_dict_delayed(self):
        if False:
            while True:
                i = 10

        @gen.coroutine
        def f():
            if False:
                while True:
                    i = 10
            responses = (yield gen.multi_future(dict(foo=self.delay(3, 'v1'), bar=self.delay(1, 'v2'))))
            self.assertEqual(responses, dict(foo='v1', bar='v2'))
        self.io_loop.run_sync(f)

    @skipOnTravis
    @gen_test
    def test_multi_performance(self):
        if False:
            print('Hello World!')
        start = time.time()
        yield [gen.moment for i in range(2000)]
        end = time.time()
        self.assertLess(end - start, 1.0)

    @gen_test
    def test_multi_empty(self):
        if False:
            for i in range(10):
                print('nop')
        x = (yield [])
        self.assertTrue(isinstance(x, list))
        y = (yield {})
        self.assertTrue(isinstance(y, dict))

    @gen_test
    def test_future(self):
        if False:
            i = 10
            return i + 15
        result = (yield self.async_future(1))
        self.assertEqual(result, 1)

    @gen_test
    def test_multi_future(self):
        if False:
            return 10
        results = (yield [self.async_future(1), self.async_future(2)])
        self.assertEqual(results, [1, 2])

    @gen_test
    def test_multi_future_duplicate(self):
        if False:
            while True:
                i = 10
        f = self.async_future(2)
        results = (yield [self.async_future(1), f, self.async_future(3), f])
        self.assertEqual(results, [1, 2, 3, 2])

    @gen_test
    def test_multi_dict_future(self):
        if False:
            for i in range(10):
                print('nop')
        results = (yield dict(foo=self.async_future(1), bar=self.async_future(2)))
        self.assertEqual(results, dict(foo=1, bar=2))

    @gen_test
    def test_multi_exceptions(self):
        if False:
            return 10
        with ExpectLog(app_log, 'Multiple exceptions in yield list'):
            with self.assertRaises(RuntimeError) as cm:
                yield gen.Multi([self.async_exception(RuntimeError('error 1')), self.async_exception(RuntimeError('error 2'))])
        self.assertEqual(str(cm.exception), 'error 1')
        with self.assertRaises(RuntimeError):
            yield gen.Multi([self.async_exception(RuntimeError('error 1')), self.async_future(2)])
        with self.assertRaises(RuntimeError):
            yield gen.Multi([self.async_exception(RuntimeError('error 1')), self.async_exception(RuntimeError('error 2'))], quiet_exceptions=RuntimeError)

    @gen_test
    def test_multi_future_exceptions(self):
        if False:
            print('Hello World!')
        with ExpectLog(app_log, 'Multiple exceptions in yield list'):
            with self.assertRaises(RuntimeError) as cm:
                yield [self.async_exception(RuntimeError('error 1')), self.async_exception(RuntimeError('error 2'))]
        self.assertEqual(str(cm.exception), 'error 1')
        with self.assertRaises(RuntimeError):
            yield [self.async_exception(RuntimeError('error 1')), self.async_future(2)]
        with self.assertRaises(RuntimeError):
            yield gen.multi_future([self.async_exception(RuntimeError('error 1')), self.async_exception(RuntimeError('error 2'))], quiet_exceptions=RuntimeError)

    def test_sync_raise_return(self):
        if False:
            print('Hello World!')

        @gen.coroutine
        def f():
            if False:
                for i in range(10):
                    print('nop')
            raise gen.Return()
        self.io_loop.run_sync(f)

    def test_async_raise_return(self):
        if False:
            while True:
                i = 10

        @gen.coroutine
        def f():
            if False:
                i = 10
                return i + 15
            yield gen.moment
            raise gen.Return()
        self.io_loop.run_sync(f)

    def test_sync_raise_return_value(self):
        if False:
            for i in range(10):
                print('nop')

        @gen.coroutine
        def f():
            if False:
                for i in range(10):
                    print('nop')
            raise gen.Return(42)
        self.assertEqual(42, self.io_loop.run_sync(f))

    def test_sync_raise_return_value_tuple(self):
        if False:
            print('Hello World!')

        @gen.coroutine
        def f():
            if False:
                i = 10
                return i + 15
            raise gen.Return((1, 2))
        self.assertEqual((1, 2), self.io_loop.run_sync(f))

    def test_async_raise_return_value(self):
        if False:
            for i in range(10):
                print('nop')

        @gen.coroutine
        def f():
            if False:
                i = 10
                return i + 15
            yield gen.moment
            raise gen.Return(42)
        self.assertEqual(42, self.io_loop.run_sync(f))

    def test_async_raise_return_value_tuple(self):
        if False:
            for i in range(10):
                print('nop')

        @gen.coroutine
        def f():
            if False:
                return 10
            yield gen.moment
            raise gen.Return((1, 2))
        self.assertEqual((1, 2), self.io_loop.run_sync(f))

class GenCoroutineTest(AsyncTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.finished = False
        super().setUp()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        assert self.finished

    def test_attributes(self):
        if False:
            print('Hello World!')
        self.finished = True

        def f():
            if False:
                for i in range(10):
                    print('nop')
            yield gen.moment
        coro = gen.coroutine(f)
        self.assertEqual(coro.__name__, f.__name__)
        self.assertEqual(coro.__module__, f.__module__)
        self.assertIs(coro.__wrapped__, f)

    def test_is_coroutine_function(self):
        if False:
            for i in range(10):
                print('nop')
        self.finished = True

        def f():
            if False:
                return 10
            yield gen.moment
        coro = gen.coroutine(f)
        self.assertFalse(gen.is_coroutine_function(f))
        self.assertTrue(gen.is_coroutine_function(coro))
        self.assertFalse(gen.is_coroutine_function(coro()))

    @gen_test
    def test_sync_gen_return(self):
        if False:
            return 10

        @gen.coroutine
        def f():
            if False:
                print('Hello World!')
            raise gen.Return(42)
        result = (yield f())
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_async_gen_return(self):
        if False:
            for i in range(10):
                print('nop')

        @gen.coroutine
        def f():
            if False:
                while True:
                    i = 10
            yield gen.moment
            raise gen.Return(42)
        result = (yield f())
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_sync_return(self):
        if False:
            print('Hello World!')

        @gen.coroutine
        def f():
            if False:
                i = 10
                return i + 15
            return 42
        result = (yield f())
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_async_return(self):
        if False:
            print('Hello World!')

        @gen.coroutine
        def f():
            if False:
                print('Hello World!')
            yield gen.moment
            return 42
        result = (yield f())
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_async_early_return(self):
        if False:
            while True:
                i = 10

        @gen.coroutine
        def f():
            if False:
                print('Hello World!')
            if True:
                return 42
            yield gen.Task(self.io_loop.add_callback)
        result = (yield f())
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_async_await(self):
        if False:
            print('Hello World!')

        @gen.coroutine
        def f1():
            if False:
                i = 10
                return i + 15
            yield gen.moment
            raise gen.Return(42)

        async def f2():
            result = await f1()
            return result
        result = (yield f2())
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_asyncio_sleep_zero(self):
        if False:
            i = 10
            return i + 15

        async def f():
            import asyncio
            await asyncio.sleep(0)
            return 42
        result = (yield f())
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_async_await_mixed_multi_native_future(self):
        if False:
            for i in range(10):
                print('nop')

        @gen.coroutine
        def f1():
            if False:
                return 10
            yield gen.moment

        async def f2():
            await f1()
            return 42

        @gen.coroutine
        def f3():
            if False:
                i = 10
                return i + 15
            yield gen.moment
            raise gen.Return(43)
        results = (yield [f2(), f3()])
        self.assertEqual(results, [42, 43])
        self.finished = True

    @gen_test
    def test_async_with_timeout(self):
        if False:
            while True:
                i = 10

        async def f1():
            return 42
        result = (yield gen.with_timeout(datetime.timedelta(hours=1), f1()))
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_sync_return_no_value(self):
        if False:
            while True:
                i = 10

        @gen.coroutine
        def f():
            if False:
                return 10
            return
        result = (yield f())
        self.assertEqual(result, None)
        self.finished = True

    @gen_test
    def test_async_return_no_value(self):
        if False:
            while True:
                i = 10

        @gen.coroutine
        def f():
            if False:
                for i in range(10):
                    print('nop')
            yield gen.moment
            return
        result = (yield f())
        self.assertEqual(result, None)
        self.finished = True

    @gen_test
    def test_sync_raise(self):
        if False:
            while True:
                i = 10

        @gen.coroutine
        def f():
            if False:
                while True:
                    i = 10
            1 / 0
        future = f()
        with self.assertRaises(ZeroDivisionError):
            yield future
        self.finished = True

    @gen_test
    def test_async_raise(self):
        if False:
            for i in range(10):
                print('nop')

        @gen.coroutine
        def f():
            if False:
                while True:
                    i = 10
            yield gen.moment
            1 / 0
        future = f()
        with self.assertRaises(ZeroDivisionError):
            yield future
        self.finished = True

    @gen_test
    def test_replace_yieldpoint_exception(self):
        if False:
            return 10

        @gen.coroutine
        def f1():
            if False:
                i = 10
                return i + 15
            1 / 0

        @gen.coroutine
        def f2():
            if False:
                while True:
                    i = 10
            try:
                yield f1()
            except ZeroDivisionError:
                raise KeyError()
        future = f2()
        with self.assertRaises(KeyError):
            yield future
        self.finished = True

    @gen_test
    def test_swallow_yieldpoint_exception(self):
        if False:
            i = 10
            return i + 15

        @gen.coroutine
        def f1():
            if False:
                i = 10
                return i + 15
            1 / 0

        @gen.coroutine
        def f2():
            if False:
                return 10
            try:
                yield f1()
            except ZeroDivisionError:
                raise gen.Return(42)
        result = (yield f2())
        self.assertEqual(result, 42)
        self.finished = True

    @gen_test
    def test_moment(self):
        if False:
            i = 10
            return i + 15
        calls = []

        @gen.coroutine
        def f(name, yieldable):
            if False:
                i = 10
                return i + 15
            for i in range(5):
                calls.append(name)
                yield yieldable
        immediate = Future()
        immediate.set_result(None)
        yield [f('a', immediate), f('b', immediate)]
        self.assertEqual(''.join(calls), 'aaaaabbbbb')
        calls = []
        yield [f('a', gen.moment), f('b', gen.moment)]
        self.assertEqual(''.join(calls), 'ababababab')
        self.finished = True
        calls = []
        yield [f('a', gen.moment), f('b', immediate)]
        self.assertEqual(''.join(calls), 'abbbbbaaaa')

    @gen_test
    def test_sleep(self):
        if False:
            print('Hello World!')
        yield gen.sleep(0.01)
        self.finished = True

    @gen_test
    def test_py3_leak_exception_context(self):
        if False:
            print('Hello World!')

        class LeakedException(Exception):
            pass

        @gen.coroutine
        def inner(iteration):
            if False:
                while True:
                    i = 10
            raise LeakedException(iteration)
        try:
            yield inner(1)
        except LeakedException as e:
            self.assertEqual(str(e), '1')
            self.assertIsNone(e.__context__)
        try:
            yield inner(2)
        except LeakedException as e:
            self.assertEqual(str(e), '2')
            self.assertIsNone(e.__context__)
        self.finished = True

    @skipNotCPython
    @unittest.skipIf((3,) < sys.version_info < (3, 6), 'asyncio.Future has reference cycles')
    def test_coroutine_refcounting(self):
        if False:
            for i in range(10):
                print('nop')

        @gen.coroutine
        def inner():
            if False:
                print('Hello World!')

            class Foo(object):
                pass
            local_var = Foo()
            self.local_ref = weakref.ref(local_var)

            def dummy():
                if False:
                    for i in range(10):
                        print('nop')
                pass
            yield gen.coroutine(dummy)()
            raise ValueError('Some error')

        @gen.coroutine
        def inner2():
            if False:
                for i in range(10):
                    print('nop')
            try:
                yield inner()
            except ValueError:
                pass
        self.io_loop.run_sync(inner2, timeout=3)
        self.assertIs(self.local_ref(), None)
        self.finished = True

    def test_asyncio_future_debug_info(self):
        if False:
            i = 10
            return i + 15
        self.finished = True
        asyncio_loop = asyncio.get_event_loop()
        self.addCleanup(asyncio_loop.set_debug, asyncio_loop.get_debug())
        asyncio_loop.set_debug(True)

        def f():
            if False:
                for i in range(10):
                    print('nop')
            yield gen.moment
        coro = gen.coroutine(f)()
        self.assertIsInstance(coro, asyncio.Future)
        expected = 'created at %s:%d' % (__file__, f.__code__.co_firstlineno + 3)
        actual = repr(coro)
        self.assertIn(expected, actual)

    @gen_test
    def test_asyncio_gather(self):
        if False:
            i = 10
            return i + 15

        @gen.coroutine
        def f():
            if False:
                while True:
                    i = 10
            yield gen.moment
            raise gen.Return(1)
        ret = (yield asyncio.gather(f(), f()))
        self.assertEqual(ret, [1, 1])
        self.finished = True

class GenCoroutineSequenceHandler(RequestHandler):

    @gen.coroutine
    def get(self):
        if False:
            i = 10
            return i + 15
        yield gen.moment
        self.write('1')
        yield gen.moment
        self.write('2')
        yield gen.moment
        self.finish('3')

class GenCoroutineUnfinishedSequenceHandler(RequestHandler):

    @gen.coroutine
    def get(self):
        if False:
            while True:
                i = 10
        yield gen.moment
        self.write('1')
        yield gen.moment
        self.write('2')
        yield gen.moment
        self.write('3')

class UndecoratedCoroutinesHandler(RequestHandler):

    @gen.coroutine
    def prepare(self):
        if False:
            while True:
                i = 10
        self.chunks = []
        yield gen.moment
        self.chunks.append('1')

    @gen.coroutine
    def get(self):
        if False:
            while True:
                i = 10
        self.chunks.append('2')
        yield gen.moment
        self.chunks.append('3')
        yield gen.moment
        self.write(''.join(self.chunks))

class AsyncPrepareErrorHandler(RequestHandler):

    @gen.coroutine
    def prepare(self):
        if False:
            while True:
                i = 10
        yield gen.moment
        raise HTTPError(403)

    def get(self):
        if False:
            print('Hello World!')
        self.finish('ok')

class NativeCoroutineHandler(RequestHandler):

    async def get(self):
        await asyncio.sleep(0)
        self.write('ok')

class GenWebTest(AsyncHTTPTestCase):

    def get_app(self):
        if False:
            print('Hello World!')
        return Application([('/coroutine_sequence', GenCoroutineSequenceHandler), ('/coroutine_unfinished_sequence', GenCoroutineUnfinishedSequenceHandler), ('/undecorated_coroutine', UndecoratedCoroutinesHandler), ('/async_prepare_error', AsyncPrepareErrorHandler), ('/native_coroutine', NativeCoroutineHandler)])

    def test_coroutine_sequence_handler(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/coroutine_sequence')
        self.assertEqual(response.body, b'123')

    def test_coroutine_unfinished_sequence_handler(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/coroutine_unfinished_sequence')
        self.assertEqual(response.body, b'123')

    def test_undecorated_coroutines(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/undecorated_coroutine')
        self.assertEqual(response.body, b'123')

    def test_async_prepare_error_handler(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/async_prepare_error')
        self.assertEqual(response.code, 403)

    def test_native_coroutine_handler(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/native_coroutine')
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, b'ok')

class WithTimeoutTest(AsyncTestCase):

    @gen_test
    def test_timeout(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(gen.TimeoutError):
            yield gen.with_timeout(datetime.timedelta(seconds=0.1), Future())

    @gen_test
    def test_completes_before_timeout(self):
        if False:
            return 10
        future = Future()
        self.io_loop.add_timeout(datetime.timedelta(seconds=0.1), lambda : future.set_result('asdf'))
        result = (yield gen.with_timeout(datetime.timedelta(seconds=3600), future))
        self.assertEqual(result, 'asdf')

    @gen_test
    def test_fails_before_timeout(self):
        if False:
            print('Hello World!')
        future = Future()
        self.io_loop.add_timeout(datetime.timedelta(seconds=0.1), lambda : future.set_exception(ZeroDivisionError()))
        with self.assertRaises(ZeroDivisionError):
            yield gen.with_timeout(datetime.timedelta(seconds=3600), future)

    @gen_test
    def test_already_resolved(self):
        if False:
            while True:
                i = 10
        future = Future()
        future.set_result('asdf')
        result = (yield gen.with_timeout(datetime.timedelta(seconds=3600), future))
        self.assertEqual(result, 'asdf')

    @gen_test
    def test_timeout_concurrent_future(self):
        if False:
            print('Hello World!')
        with futures.ThreadPoolExecutor(1) as executor:
            with self.assertRaises(gen.TimeoutError):
                yield gen.with_timeout(self.io_loop.time(), executor.submit(time.sleep, 0.1))

    @gen_test
    def test_completed_concurrent_future(self):
        if False:
            return 10
        with futures.ThreadPoolExecutor(1) as executor:

            def dummy():
                if False:
                    return 10
                pass
            f = executor.submit(dummy)
            f.result()
            yield gen.with_timeout(datetime.timedelta(seconds=3600), f)

    @gen_test
    def test_normal_concurrent_future(self):
        if False:
            while True:
                i = 10
        with futures.ThreadPoolExecutor(1) as executor:
            yield gen.with_timeout(datetime.timedelta(seconds=3600), executor.submit(lambda : time.sleep(0.01)))

class WaitIteratorTest(AsyncTestCase):

    @gen_test
    def test_empty_iterator(self):
        if False:
            while True:
                i = 10
        g = gen.WaitIterator()
        self.assertTrue(g.done(), 'empty generator iterated')
        with self.assertRaises(ValueError):
            g = gen.WaitIterator(Future(), bar=Future())
        self.assertEqual(g.current_index, None, 'bad nil current index')
        self.assertEqual(g.current_future, None, 'bad nil current future')

    @gen_test
    def test_already_done(self):
        if False:
            while True:
                i = 10
        f1 = Future()
        f2 = Future()
        f3 = Future()
        f1.set_result(24)
        f2.set_result(42)
        f3.set_result(84)
        g = gen.WaitIterator(f1, f2, f3)
        i = 0
        while not g.done():
            r = (yield g.next())
            if i == 0:
                self.assertEqual(g.current_index, 0)
                self.assertIs(g.current_future, f1)
                self.assertEqual(r, 24)
            elif i == 1:
                self.assertEqual(g.current_index, 1)
                self.assertIs(g.current_future, f2)
                self.assertEqual(r, 42)
            elif i == 2:
                self.assertEqual(g.current_index, 2)
                self.assertIs(g.current_future, f3)
                self.assertEqual(r, 84)
            i += 1
        self.assertEqual(g.current_index, None, 'bad nil current index')
        self.assertEqual(g.current_future, None, 'bad nil current future')
        dg = gen.WaitIterator(f1=f1, f2=f2)
        while not dg.done():
            dr = (yield dg.next())
            if dg.current_index == 'f1':
                self.assertTrue(dg.current_future == f1 and dr == 24, 'WaitIterator dict status incorrect')
            elif dg.current_index == 'f2':
                self.assertTrue(dg.current_future == f2 and dr == 42, 'WaitIterator dict status incorrect')
            else:
                self.fail('got bad WaitIterator index {}'.format(dg.current_index))
            i += 1
        self.assertEqual(dg.current_index, None, 'bad nil current index')
        self.assertEqual(dg.current_future, None, 'bad nil current future')

    def finish_coroutines(self, iteration, futures):
        if False:
            while True:
                i = 10
        if iteration == 3:
            futures[2].set_result(24)
        elif iteration == 5:
            futures[0].set_exception(ZeroDivisionError())
        elif iteration == 8:
            futures[1].set_result(42)
            futures[3].set_result(84)
        if iteration < 8:
            self.io_loop.add_callback(self.finish_coroutines, iteration + 1, futures)

    @gen_test
    def test_iterator(self):
        if False:
            print('Hello World!')
        futures = [Future(), Future(), Future(), Future()]
        self.finish_coroutines(0, futures)
        g = gen.WaitIterator(*futures)
        i = 0
        while not g.done():
            try:
                r = (yield g.next())
            except ZeroDivisionError:
                self.assertIs(g.current_future, futures[0], 'exception future invalid')
            else:
                if i == 0:
                    self.assertEqual(r, 24, 'iterator value incorrect')
                    self.assertEqual(g.current_index, 2, 'wrong index')
                elif i == 2:
                    self.assertEqual(r, 42, 'iterator value incorrect')
                    self.assertEqual(g.current_index, 1, 'wrong index')
                elif i == 3:
                    self.assertEqual(r, 84, 'iterator value incorrect')
                    self.assertEqual(g.current_index, 3, 'wrong index')
            i += 1

    @gen_test
    def test_iterator_async_await(self):
        if False:
            return 10
        futures = [Future(), Future(), Future(), Future()]
        self.finish_coroutines(0, futures)
        self.finished = False

        async def f():
            i = 0
            g = gen.WaitIterator(*futures)
            try:
                async for r in g:
                    if i == 0:
                        self.assertEqual(r, 24, 'iterator value incorrect')
                        self.assertEqual(g.current_index, 2, 'wrong index')
                    else:
                        raise Exception('expected exception on iteration 1')
                    i += 1
            except ZeroDivisionError:
                i += 1
            async for r in g:
                if i == 2:
                    self.assertEqual(r, 42, 'iterator value incorrect')
                    self.assertEqual(g.current_index, 1, 'wrong index')
                elif i == 3:
                    self.assertEqual(r, 84, 'iterator value incorrect')
                    self.assertEqual(g.current_index, 3, 'wrong index')
                else:
                    raise Exception("didn't expect iteration %d" % i)
                i += 1
            self.finished = True
        yield f()
        self.assertTrue(self.finished)

    @gen_test
    def test_no_ref(self):
        if False:
            for i in range(10):
                print('nop')
        yield gen.with_timeout(datetime.timedelta(seconds=0.1), gen.WaitIterator(gen.sleep(0)).next())

class RunnerGCTest(AsyncTestCase):

    def is_pypy3(self):
        if False:
            print('Hello World!')
        return platform.python_implementation() == 'PyPy' and sys.version_info > (3,)

    @gen_test
    def test_gc(self):
        if False:
            while True:
                i = 10
        weakref_scope = [None]

        def callback():
            if False:
                return 10
            gc.collect(2)
            weakref_scope[0]().set_result(123)

        @gen.coroutine
        def tester():
            if False:
                print('Hello World!')
            fut = Future()
            weakref_scope[0] = weakref.ref(fut)
            self.io_loop.add_callback(callback)
            yield fut
        yield gen.with_timeout(datetime.timedelta(seconds=0.2), tester())

    def test_gc_infinite_coro(self):
        if False:
            for i in range(10):
                print('nop')
        loop = self.get_new_ioloop()
        result = []
        wfut = []

        @gen.coroutine
        def infinite_coro():
            if False:
                i = 10
                return i + 15
            try:
                while True:
                    yield gen.sleep(0.001)
                    result.append(True)
            finally:
                result.append(None)

        @gen.coroutine
        def do_something():
            if False:
                print('Hello World!')
            fut = infinite_coro()
            fut._refcycle = fut
            wfut.append(weakref.ref(fut))
            yield gen.sleep(0.2)
        loop.run_sync(do_something)
        loop.close()
        gc.collect()
        self.assertIs(wfut[0](), None)
        self.assertGreaterEqual(len(result), 2)
        if not self.is_pypy3():
            self.assertIs(result[-1], None)

    def test_gc_infinite_async_await(self):
        if False:
            for i in range(10):
                print('nop')
        import asyncio

        async def infinite_coro(result):
            try:
                while True:
                    await gen.sleep(0.001)
                    result.append(True)
            finally:
                result.append(None)
        loop = self.get_new_ioloop()
        result = []
        wfut = []

        @gen.coroutine
        def do_something():
            if False:
                for i in range(10):
                    print('nop')
            fut = asyncio.get_event_loop().create_task(infinite_coro(result))
            fut._refcycle = fut
            wfut.append(weakref.ref(fut))
            yield gen.sleep(0.2)
        loop.run_sync(do_something)
        with ExpectLog('asyncio', 'Task was destroyed but it is pending'):
            loop.close()
            gc.collect()
        self.assertIs(wfut[0](), None)
        self.assertGreaterEqual(len(result), 2)
        if not self.is_pypy3():
            self.assertIs(result[-1], None)

    def test_multi_moment(self):
        if False:
            return 10

        @gen.coroutine
        def wait_a_moment():
            if False:
                return 10
            result = (yield gen.multi([gen.moment, gen.moment]))
            raise gen.Return(result)
        loop = self.get_new_ioloop()
        result = loop.run_sync(wait_a_moment)
        self.assertEqual(result, [None, None])
if contextvars is not None:
    ctx_var = contextvars.ContextVar('ctx_var')

@unittest.skipIf(contextvars is None, 'contextvars module not present')
class ContextVarsTest(AsyncTestCase):

    async def native_root(self, x):
        ctx_var.set(x)
        await self.inner(x)

    @gen.coroutine
    def gen_root(self, x):
        if False:
            for i in range(10):
                print('nop')
        ctx_var.set(x)
        yield
        yield self.inner(x)

    async def inner(self, x):
        self.assertEqual(ctx_var.get(), x)
        await self.gen_inner(x)
        self.assertEqual(ctx_var.get(), x)
        ctx = contextvars.copy_context()
        await self.io_loop.run_in_executor(None, lambda : ctx.run(self.thread_inner, x))
        self.assertEqual(ctx_var.get(), x)
        await asyncio.get_event_loop().run_in_executor(None, lambda : ctx.run(self.thread_inner, x))
        self.assertEqual(ctx_var.get(), x)

    @gen.coroutine
    def gen_inner(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(ctx_var.get(), x)
        yield
        self.assertEqual(ctx_var.get(), x)

    def thread_inner(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(ctx_var.get(), x)

    @gen_test
    def test_propagate(self):
        if False:
            print('Hello World!')
        yield [self.native_root(1), self.native_root(2), self.gen_root(3), self.gen_root(4)]

    @gen_test
    def test_reset(self):
        if False:
            for i in range(10):
                print('nop')
        token = ctx_var.set(1)
        yield
        ctx_var.reset(token)

    @gen_test
    def test_propagate_to_first_yield_with_native_async_function(self):
        if False:
            i = 10
            return i + 15
        x = 10

        async def native_async_function():
            self.assertEqual(ctx_var.get(), x)
        ctx_var.set(x)
        yield native_async_function()
if __name__ == '__main__':
    unittest.main()