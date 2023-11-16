import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
from collections.abc import Generator
import contextlib
import datetime
import functools
import socket
import subprocess
import sys
import threading
import time
import types
from unittest import mock
import unittest
from tornado.escape import native_str
from tornado import gen
from tornado.ioloop import IOLoop, TimeoutError, PeriodicCallback
from tornado.log import app_log
from tornado.testing import AsyncTestCase, bind_unused_port, ExpectLog, gen_test, setup_with_context_manager
from tornado.test.util import ignore_deprecation, skipIfNonUnix, skipOnTravis
from tornado.concurrent import Future
import typing
if typing.TYPE_CHECKING:
    from typing import List

class TestIOLoop(AsyncTestCase):

    def test_add_callback_return_sequence(self):
        if False:
            i = 10
            return i + 15
        self.calls = 0
        loop = self.io_loop
        test = self
        old_add_callback = loop.add_callback

        def add_callback(self, callback, *args, **kwargs):
            if False:
                while True:
                    i = 10
            test.calls += 1
            old_add_callback(callback, *args, **kwargs)
        loop.add_callback = types.MethodType(add_callback, loop)
        loop.add_callback(lambda : {})
        loop.add_callback(lambda : [])
        loop.add_timeout(datetime.timedelta(milliseconds=50), loop.stop)
        loop.start()
        self.assertLess(self.calls, 10)

    @skipOnTravis
    def test_add_callback_wakeup(self):
        if False:
            for i in range(10):
                print('nop')

        def callback():
            if False:
                while True:
                    i = 10
            self.called = True
            self.stop()

        def schedule_callback():
            if False:
                i = 10
                return i + 15
            self.called = False
            self.io_loop.add_callback(callback)
            self.start_time = time.time()
        self.io_loop.add_timeout(self.io_loop.time(), schedule_callback)
        self.wait()
        self.assertAlmostEqual(time.time(), self.start_time, places=2)
        self.assertTrue(self.called)

    @skipOnTravis
    def test_add_callback_wakeup_other_thread(self):
        if False:
            for i in range(10):
                print('nop')

        def target():
            if False:
                print('Hello World!')
            time.sleep(0.01)
            self.stop_time = time.time()
            self.io_loop.add_callback(self.stop)
        thread = threading.Thread(target=target)
        self.io_loop.add_callback(thread.start)
        self.wait()
        delta = time.time() - self.stop_time
        self.assertLess(delta, 0.1)
        thread.join()

    def test_add_timeout_timedelta(self):
        if False:
            while True:
                i = 10
        self.io_loop.add_timeout(datetime.timedelta(microseconds=1), self.stop)
        self.wait()

    def test_multiple_add(self):
        if False:
            i = 10
            return i + 15
        (sock, port) = bind_unused_port()
        try:
            self.io_loop.add_handler(sock.fileno(), lambda fd, events: None, IOLoop.READ)
            self.assertRaises(Exception, self.io_loop.add_handler, sock.fileno(), lambda fd, events: None, IOLoop.READ)
        finally:
            self.io_loop.remove_handler(sock.fileno())
            sock.close()

    def test_remove_without_add(self):
        if False:
            while True:
                i = 10
        (sock, port) = bind_unused_port()
        try:
            self.io_loop.remove_handler(sock.fileno())
        finally:
            sock.close()

    def test_add_callback_from_signal(self):
        if False:
            for i in range(10):
                print('nop')
        with ignore_deprecation():
            self.io_loop.add_callback_from_signal(self.stop)
        self.wait()

    def test_add_callback_from_signal_other_thread(self):
        if False:
            return 10
        other_ioloop = IOLoop()
        thread = threading.Thread(target=other_ioloop.start)
        thread.start()
        with ignore_deprecation():
            other_ioloop.add_callback_from_signal(other_ioloop.stop)
        thread.join()
        other_ioloop.close()

    def test_add_callback_while_closing(self):
        if False:
            while True:
                i = 10
        closing = threading.Event()

        def target():
            if False:
                for i in range(10):
                    print('nop')
            other_ioloop.add_callback(other_ioloop.stop)
            other_ioloop.start()
            closing.set()
            other_ioloop.close(all_fds=True)
        other_ioloop = IOLoop()
        thread = threading.Thread(target=target)
        thread.start()
        closing.wait()
        for i in range(1000):
            other_ioloop.add_callback(lambda : None)

    @skipIfNonUnix
    def test_read_while_writeable(self):
        if False:
            return 10
        (client, server) = socket.socketpair()
        try:

            def handler(fd, events):
                if False:
                    for i in range(10):
                        print('nop')
                self.assertEqual(events, IOLoop.READ)
                self.stop()
            self.io_loop.add_handler(client.fileno(), handler, IOLoop.READ)
            self.io_loop.add_timeout(self.io_loop.time() + 0.01, functools.partial(server.send, b'asdf'))
            self.wait()
            self.io_loop.remove_handler(client.fileno())
        finally:
            client.close()
            server.close()

    def test_remove_timeout_after_fire(self):
        if False:
            print('Hello World!')
        handle = self.io_loop.add_timeout(self.io_loop.time(), self.stop)
        self.wait()
        self.io_loop.remove_timeout(handle)

    def test_remove_timeout_cleanup(self):
        if False:
            while True:
                i = 10
        for i in range(2000):
            timeout = self.io_loop.add_timeout(self.io_loop.time() + 3600, lambda : None)
            self.io_loop.remove_timeout(timeout)
        self.io_loop.add_callback(lambda : self.io_loop.add_callback(self.stop))
        self.wait()

    def test_remove_timeout_from_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        calls = [False, False]
        now = self.io_loop.time()

        def t1():
            if False:
                for i in range(10):
                    print('nop')
            calls[0] = True
            self.io_loop.remove_timeout(t2_handle)
        self.io_loop.add_timeout(now + 0.01, t1)

        def t2():
            if False:
                for i in range(10):
                    print('nop')
            calls[1] = True
        t2_handle = self.io_loop.add_timeout(now + 0.02, t2)
        self.io_loop.add_timeout(now + 0.03, self.stop)
        time.sleep(0.03)
        self.wait()
        self.assertEqual(calls, [True, False])

    def test_timeout_with_arguments(self):
        if False:
            print('Hello World!')
        results = []
        self.io_loop.add_timeout(self.io_loop.time(), results.append, 1)
        self.io_loop.add_timeout(datetime.timedelta(seconds=0), results.append, 2)
        self.io_loop.call_at(self.io_loop.time(), results.append, 3)
        self.io_loop.call_later(0, results.append, 4)
        self.io_loop.call_later(0, self.stop)
        self.wait()
        self.assertEqual(sorted(results), [1, 2, 3, 4])

    def test_add_timeout_return(self):
        if False:
            i = 10
            return i + 15
        handle = self.io_loop.add_timeout(self.io_loop.time(), lambda : None)
        self.assertFalse(handle is None)
        self.io_loop.remove_timeout(handle)

    def test_call_at_return(self):
        if False:
            for i in range(10):
                print('nop')
        handle = self.io_loop.call_at(self.io_loop.time(), lambda : None)
        self.assertFalse(handle is None)
        self.io_loop.remove_timeout(handle)

    def test_call_later_return(self):
        if False:
            print('Hello World!')
        handle = self.io_loop.call_later(0, lambda : None)
        self.assertFalse(handle is None)
        self.io_loop.remove_timeout(handle)

    def test_close_file_object(self):
        if False:
            i = 10
            return i + 15
        'When a file object is used instead of a numeric file descriptor,\n        the object should be closed (by IOLoop.close(all_fds=True),\n        not just the fd.\n        '

        class SocketWrapper(object):

            def __init__(self, sockobj):
                if False:
                    while True:
                        i = 10
                self.sockobj = sockobj
                self.closed = False

            def fileno(self):
                if False:
                    while True:
                        i = 10
                return self.sockobj.fileno()

            def close(self):
                if False:
                    while True:
                        i = 10
                self.closed = True
                self.sockobj.close()
        (sockobj, port) = bind_unused_port()
        socket_wrapper = SocketWrapper(sockobj)
        io_loop = IOLoop()
        io_loop.add_handler(socket_wrapper, lambda fd, events: None, IOLoop.READ)
        io_loop.close(all_fds=True)
        self.assertTrue(socket_wrapper.closed)

    def test_handler_callback_file_object(self):
        if False:
            while True:
                i = 10
        'The handler callback receives the same fd object it passed in.'
        (server_sock, port) = bind_unused_port()
        fds = []

        def handle_connection(fd, events):
            if False:
                return 10
            fds.append(fd)
            (conn, addr) = server_sock.accept()
            conn.close()
            self.stop()
        self.io_loop.add_handler(server_sock, handle_connection, IOLoop.READ)
        with contextlib.closing(socket.socket()) as client_sock:
            client_sock.connect(('127.0.0.1', port))
            self.wait()
        self.io_loop.remove_handler(server_sock)
        self.io_loop.add_handler(server_sock.fileno(), handle_connection, IOLoop.READ)
        with contextlib.closing(socket.socket()) as client_sock:
            client_sock.connect(('127.0.0.1', port))
            self.wait()
        self.assertIs(fds[0], server_sock)
        self.assertEqual(fds[1], server_sock.fileno())
        self.io_loop.remove_handler(server_sock.fileno())
        server_sock.close()

    def test_mixed_fd_fileobj(self):
        if False:
            i = 10
            return i + 15
        (server_sock, port) = bind_unused_port()

        def f(fd, events):
            if False:
                return 10
            pass
        self.io_loop.add_handler(server_sock, f, IOLoop.READ)
        with self.assertRaises(Exception):
            self.io_loop.add_handler(server_sock.fileno(), f, IOLoop.READ)
        self.io_loop.remove_handler(server_sock.fileno())
        server_sock.close()

    def test_reentrant(self):
        if False:
            print('Hello World!')
        'Calling start() twice should raise an error, not deadlock.'
        returned_from_start = [False]
        got_exception = [False]

        def callback():
            if False:
                return 10
            try:
                self.io_loop.start()
                returned_from_start[0] = True
            except Exception:
                got_exception[0] = True
            self.stop()
        self.io_loop.add_callback(callback)
        self.wait()
        self.assertTrue(got_exception[0])
        self.assertFalse(returned_from_start[0])

    def test_exception_logging(self):
        if False:
            while True:
                i = 10
        'Uncaught exceptions get logged by the IOLoop.'
        self.io_loop.add_callback(lambda : 1 / 0)
        self.io_loop.add_callback(self.stop)
        with ExpectLog(app_log, 'Exception in callback'):
            self.wait()

    def test_exception_logging_future(self):
        if False:
            while True:
                i = 10
        'The IOLoop examines exceptions from Futures and logs them.'

        @gen.coroutine
        def callback():
            if False:
                for i in range(10):
                    print('nop')
            self.io_loop.add_callback(self.stop)
            1 / 0
        self.io_loop.add_callback(callback)
        with ExpectLog(app_log, 'Exception in callback'):
            self.wait()

    def test_exception_logging_native_coro(self):
        if False:
            for i in range(10):
                print('nop')
        'The IOLoop examines exceptions from awaitables and logs them.'

        async def callback():
            self.io_loop.add_callback(self.io_loop.add_callback, self.stop)
            1 / 0
        self.io_loop.add_callback(callback)
        with ExpectLog(app_log, 'Exception in callback'):
            self.wait()

    def test_spawn_callback(self):
        if False:
            while True:
                i = 10
        self.io_loop.add_callback(lambda : 1 / 0)
        self.io_loop.add_callback(self.stop)
        with ExpectLog(app_log, 'Exception in callback'):
            self.wait()
        self.io_loop.spawn_callback(lambda : 1 / 0)
        self.io_loop.add_callback(self.stop)
        with ExpectLog(app_log, 'Exception in callback'):
            self.wait()

    @skipIfNonUnix
    def test_remove_handler_from_handler(self):
        if False:
            print('Hello World!')
        (client, server) = socket.socketpair()
        try:
            client.send(b'abc')
            server.send(b'abc')
            chunks = []

            def handle_read(fd, events):
                if False:
                    i = 10
                    return i + 15
                chunks.append(fd.recv(1024))
                if fd is client:
                    self.io_loop.remove_handler(server)
                else:
                    self.io_loop.remove_handler(client)
            self.io_loop.add_handler(client, handle_read, self.io_loop.READ)
            self.io_loop.add_handler(server, handle_read, self.io_loop.READ)
            self.io_loop.call_later(0.1, self.stop)
            self.wait()
            self.assertEqual(chunks, [b'abc'])
        finally:
            client.close()
            server.close()

    @skipIfNonUnix
    @gen_test
    def test_init_close_race(self):
        if False:
            return 10

        def f():
            if False:
                return 10
            for i in range(10):
                loop = IOLoop(make_current=False)
                loop.close()
        yield gen.multi([self.io_loop.run_in_executor(None, f) for i in range(2)])

    def test_explicit_asyncio_loop(self):
        if False:
            return 10
        asyncio_loop = asyncio.new_event_loop()
        loop = IOLoop(asyncio_loop=asyncio_loop, make_current=False)
        assert loop.asyncio_loop is asyncio_loop
        with self.assertRaises(RuntimeError):
            IOLoop(asyncio_loop=asyncio_loop, make_current=False)
        loop.close()

class TestIOLoopCurrent(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        setup_with_context_manager(self, ignore_deprecation())
        self.io_loop = None
        IOLoop.clear_current()

    def tearDown(self):
        if False:
            while True:
                i = 10
        if self.io_loop is not None:
            self.io_loop.close()

    def test_non_current(self):
        if False:
            print('Hello World!')
        self.io_loop = IOLoop(make_current=False)
        self.assertIsNone(IOLoop.current(instance=False))
        for i in range(3):

            def f():
                if False:
                    while True:
                        i = 10
                self.current_io_loop = IOLoop.current()
                assert self.io_loop is not None
                self.io_loop.stop()
            self.io_loop.add_callback(f)
            self.io_loop.start()
            self.assertIs(self.current_io_loop, self.io_loop)
            self.assertIsNone(IOLoop.current(instance=False))

    def test_force_current(self):
        if False:
            while True:
                i = 10
        self.io_loop = IOLoop(make_current=True)
        self.assertIs(self.io_loop, IOLoop.current())

class TestIOLoopCurrentAsync(AsyncTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        setup_with_context_manager(self, ignore_deprecation())

    @gen_test
    def test_clear_without_current(self):
        if False:
            for i in range(10):
                print('nop')
        with ThreadPoolExecutor(1) as e:
            yield e.submit(IOLoop.clear_current)

class TestIOLoopFutures(AsyncTestCase):

    def test_add_future_threads(self):
        if False:
            while True:
                i = 10
        with futures.ThreadPoolExecutor(1) as pool:

            def dummy():
                if False:
                    for i in range(10):
                        print('nop')
                pass
            self.io_loop.add_future(pool.submit(dummy), lambda future: self.stop(future))
            future = self.wait()
            self.assertTrue(future.done())
            self.assertTrue(future.result() is None)

    @gen_test
    def test_run_in_executor_gen(self):
        if False:
            print('Hello World!')
        event1 = threading.Event()
        event2 = threading.Event()

        def sync_func(self_event, other_event):
            if False:
                for i in range(10):
                    print('nop')
            self_event.set()
            other_event.wait()
            return self_event
        res = (yield [IOLoop.current().run_in_executor(None, sync_func, event1, event2), IOLoop.current().run_in_executor(None, sync_func, event2, event1)])
        self.assertEqual([event1, event2], res)

    @gen_test
    def test_run_in_executor_native(self):
        if False:
            for i in range(10):
                print('nop')
        event1 = threading.Event()
        event2 = threading.Event()

        def sync_func(self_event, other_event):
            if False:
                return 10
            self_event.set()
            other_event.wait()
            return self_event

        async def async_wrapper(self_event, other_event):
            return await IOLoop.current().run_in_executor(None, sync_func, self_event, other_event)
        res = (yield [async_wrapper(event1, event2), async_wrapper(event2, event1)])
        self.assertEqual([event1, event2], res)

    @gen_test
    def test_set_default_executor(self):
        if False:
            print('Hello World!')
        count = [0]

        class MyExecutor(futures.ThreadPoolExecutor):

            def submit(self, func, *args):
                if False:
                    i = 10
                    return i + 15
                count[0] += 1
                return super().submit(func, *args)
        event = threading.Event()

        def sync_func():
            if False:
                print('Hello World!')
            event.set()
        executor = MyExecutor(1)
        loop = IOLoop.current()
        loop.set_default_executor(executor)
        yield loop.run_in_executor(None, sync_func)
        self.assertEqual(1, count[0])
        self.assertTrue(event.is_set())

class TestIOLoopRunSync(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.io_loop = IOLoop(make_current=False)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.io_loop.close()

    def test_sync_result(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(gen.BadYieldError):
            self.io_loop.run_sync(lambda : 42)

    def test_sync_exception(self):
        if False:
            return 10
        with self.assertRaises(ZeroDivisionError):
            self.io_loop.run_sync(lambda : 1 / 0)

    def test_async_result(self):
        if False:
            print('Hello World!')

        @gen.coroutine
        def f():
            if False:
                print('Hello World!')
            yield gen.moment
            raise gen.Return(42)
        self.assertEqual(self.io_loop.run_sync(f), 42)

    def test_async_exception(self):
        if False:
            print('Hello World!')

        @gen.coroutine
        def f():
            if False:
                return 10
            yield gen.moment
            1 / 0
        with self.assertRaises(ZeroDivisionError):
            self.io_loop.run_sync(f)

    def test_current(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                for i in range(10):
                    print('nop')
            self.assertIs(IOLoop.current(), self.io_loop)
        self.io_loop.run_sync(f)

    def test_timeout(self):
        if False:
            while True:
                i = 10

        @gen.coroutine
        def f():
            if False:
                for i in range(10):
                    print('nop')
            yield gen.sleep(1)
        self.assertRaises(TimeoutError, self.io_loop.run_sync, f, timeout=0.01)

    def test_native_coroutine(self):
        if False:
            while True:
                i = 10

        @gen.coroutine
        def f1():
            if False:
                return 10
            yield gen.moment

        async def f2():
            await f1()
        self.io_loop.run_sync(f2)

class TestPeriodicCallbackMath(unittest.TestCase):

    def simulate_calls(self, pc, durations):
        if False:
            print('Hello World!')
        'Simulate a series of calls to the PeriodicCallback.\n\n        Pass a list of call durations in seconds (negative values\n        work to simulate clock adjustments during the call, or more or\n        less equivalently, between calls). This method returns the\n        times at which each call would be made.\n        '
        calls = []
        now = 1000
        pc._next_timeout = now
        for d in durations:
            pc._update_next(now)
            calls.append(pc._next_timeout)
            now = pc._next_timeout + d
        return calls

    def dummy(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_basic(self):
        if False:
            while True:
                i = 10
        pc = PeriodicCallback(self.dummy, 10000)
        self.assertEqual(self.simulate_calls(pc, [0] * 5), [1010, 1020, 1030, 1040, 1050])

    def test_overrun(self):
        if False:
            i = 10
            return i + 15
        call_durations = [9, 9, 10, 11, 20, 20, 35, 35, 0, 0, 0]
        expected = [1010, 1020, 1030, 1050, 1070, 1100, 1130, 1170, 1210, 1220, 1230]
        pc = PeriodicCallback(self.dummy, 10000)
        self.assertEqual(self.simulate_calls(pc, call_durations), expected)

    def test_clock_backwards(self):
        if False:
            return 10
        pc = PeriodicCallback(self.dummy, 10000)
        self.assertEqual(self.simulate_calls(pc, [-2, -1, -3, -2, 0]), [1010, 1020, 1030, 1040, 1050])
        self.assertEqual(self.simulate_calls(pc, [-100, 0, 0]), [1010, 1020, 1030])

    def test_jitter(self):
        if False:
            i = 10
            return i + 15
        random_times = [0.5, 1, 0, 0.75]
        expected = [1010, 1022.5, 1030, 1041.25]
        call_durations = [0] * len(random_times)
        pc = PeriodicCallback(self.dummy, 10000, jitter=0.5)

        def mock_random():
            if False:
                print('Hello World!')
            return random_times.pop(0)
        with mock.patch('random.random', mock_random):
            self.assertEqual(self.simulate_calls(pc, call_durations), expected)

    def test_timedelta(self):
        if False:
            for i in range(10):
                print('nop')
        pc = PeriodicCallback(lambda : None, datetime.timedelta(minutes=1, seconds=23))
        expected_callback_time = 83000
        self.assertEqual(pc.callback_time, expected_callback_time)

class TestPeriodicCallbackAsync(AsyncTestCase):

    def test_periodic_plain(self):
        if False:
            for i in range(10):
                print('nop')
        count = 0

        def callback() -> None:
            if False:
                for i in range(10):
                    print('nop')
            nonlocal count
            count += 1
            if count == 3:
                self.stop()
        pc = PeriodicCallback(callback, 10)
        pc.start()
        self.wait()
        pc.stop()
        self.assertEqual(count, 3)

    def test_periodic_coro(self) -> None:
        if False:
            while True:
                i = 10
        counts = [0, 0]

        @gen.coroutine
        def callback() -> 'Generator[Future[None], object, None]':
            if False:
                while True:
                    i = 10
            counts[0] += 1
            yield gen.sleep(0.025)
            counts[1] += 1
            if counts[1] == 3:
                pc.stop()
                self.io_loop.add_callback(self.stop)
        pc = PeriodicCallback(callback, 10)
        pc.start()
        self.wait()
        self.assertEqual(counts[0], 3)
        self.assertEqual(counts[1], 3)

    def test_periodic_async(self) -> None:
        if False:
            return 10
        counts = [0, 0]

        async def callback() -> None:
            counts[0] += 1
            await gen.sleep(0.025)
            counts[1] += 1
            if counts[1] == 3:
                pc.stop()
                self.io_loop.add_callback(self.stop)
        pc = PeriodicCallback(callback, 10)
        pc.start()
        self.wait()
        self.assertEqual(counts[0], 3)
        self.assertEqual(counts[1], 3)

class TestIOLoopConfiguration(unittest.TestCase):

    def run_python(self, *statements):
        if False:
            while True:
                i = 10
        stmt_list = ['from tornado.ioloop import IOLoop', 'classname = lambda x: x.__class__.__name__'] + list(statements)
        args = [sys.executable, '-c', '; '.join(stmt_list)]
        return native_str(subprocess.check_output(args)).strip()

    def test_default(self):
        if False:
            for i in range(10):
                print('nop')
        cls = self.run_python('print(classname(IOLoop.current()))')
        self.assertEqual(cls, 'AsyncIOMainLoop')
        cls = self.run_python('print(classname(IOLoop()))')
        self.assertEqual(cls, 'AsyncIOLoop')

    def test_asyncio(self):
        if False:
            while True:
                i = 10
        cls = self.run_python('IOLoop.configure("tornado.platform.asyncio.AsyncIOLoop")', 'print(classname(IOLoop.current()))')
        self.assertEqual(cls, 'AsyncIOMainLoop')

    def test_asyncio_main(self):
        if False:
            print('Hello World!')
        cls = self.run_python('from tornado.platform.asyncio import AsyncIOMainLoop', 'AsyncIOMainLoop().install()', 'print(classname(IOLoop.current()))')
        self.assertEqual(cls, 'AsyncIOMainLoop')
if __name__ == '__main__':
    unittest.main()