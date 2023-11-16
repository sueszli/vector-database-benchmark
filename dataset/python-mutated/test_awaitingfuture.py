import asyncio
from _asyncio import _AwaitingFuture
from functools import wraps
import unittest
from test import cinder_support
from test.test_asyncio import utils as test_utils
if cinder_support.hasCinderX():
    from test.cinder_support import get_await_stack

def tearDownModule():
    if False:
        for i in range(10):
            print('nop')
    asyncio.set_event_loop_policy(None)

class _AwaitingFutureTests(test_utils.TestCase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self.awaited = {}
        self.loop = self.new_test_loop()
        self.addCleanup(self.loop.close)

    def _new_future(self):
        if False:
            return 10
        awaited = self.loop.create_future()
        fut = _AwaitingFuture(awaited, loop=self.loop)
        self.awaited[fut] = awaited
        return fut

    def _release_fut(self, fut):
        if False:
            return 10
        self.awaited[fut].set_result(42)
        test_utils.run_briefly(self.loop)

    def test_isfuture(self):
        if False:
            return 10
        f = self._new_future()
        self.assertTrue(asyncio.isfuture(f))
        self.assertFalse(asyncio.isfuture(type(f)))
        f.cancel()

    def test_initial_state(self):
        if False:
            for i in range(10):
                print('nop')
        f = self._new_future()
        self.assertFalse(f.cancelled())
        self.assertFalse(f.done())
        f.cancel()
        self.assertTrue(f.cancelled())

    def test_cancel(self):
        if False:
            return 10
        f = self._new_future()
        self.assertTrue(f.cancel())
        self.assertTrue(f.cancelled())
        self.assertTrue(f.done())
        self.assertRaises(asyncio.CancelledError, f.result)
        self.assertRaises(asyncio.CancelledError, f.exception)
        self.assertFalse(f.cancel())

    def test_set_exception(self):
        if False:
            print('Hello World!')
        f = self._new_future()
        with self.assertRaisesRegex(RuntimeError, 'does not support set_exception'):
            f.set_exception(Exception('testing 123'))

    def test_set_result(self):
        if False:
            i = 10
            return i + 15
        f = self._new_future()
        with self.assertRaisesRegex(RuntimeError, 'does not support set_result'):
            f.set_result(123)

    def test_result(self):
        if False:
            for i in range(10):
                print('nop')
        f = self._new_future()
        self.assertRaises(asyncio.InvalidStateError, f.result)
        self.assertFalse(f.done())
        self._release_fut(f)
        self.assertFalse(f.cancelled())
        self.assertTrue(f.done())
        self.assertEqual(f.result(), None)
        self.assertEqual(f.exception(), None)
        self.assertFalse(f.cancel())

    def test_complete_immediately(self):
        if False:
            return 10
        done = self.loop.create_future()
        done.set_result(42)
        fut = _AwaitingFuture(done, loop=self.loop)
        self.assertFalse(fut.done())
        test_utils.run_briefly(self.loop)
        self.assertTrue(fut.done())

    @unittest.skipUnless(cinder_support.hasCinderX(), 'Tests CinderX features')
    def test_propagates_awaiter(self):
        if False:
            while True:
                i = 10
        coro = None

        async def inner():
            return get_await_stack(coro)

        async def outer(coro):
            return await coro

        async def waiter(coro, loop):
            t = loop.create_task(outer(coro))
            f = _AwaitingFuture(t, loop=loop)
            await f
            return t.result()
        coro = inner()
        task = self.loop.create_task(waiter(coro, self.loop))
        self.loop.run_until_complete(task)
        stack = task.result()
        self.assertEqual(len(stack), 2)
        self.assertEqual(stack[0].cr_code, outer.__code__)
        self.assertEqual(stack[1].cr_code, waiter.__code__)

    def test_pass_nonfuture(self):
        if False:
            return 10

        async def foo():
            pass
        try:
            coro = foo()
            with self.assertRaisesRegex(TypeError, 'First argument must be a future.'):
                _AwaitingFuture(foo)
        finally:
            coro.close()

    def test_subclass(self):
        if False:
            return 10
        with self.assertRaisesRegex(TypeError, "'_asyncio._AwaitingFuture' is not an acceptable base type"):

            class MyAwaitingFuture(_AwaitingFuture):
                pass