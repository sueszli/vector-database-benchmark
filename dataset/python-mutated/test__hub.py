import re
import time
import unittest
import gevent.testing as greentest
import gevent.testing.timing
import gevent
from gevent import socket
from gevent.hub import Waiter, get_hub
from gevent._compat import NativeStrIO
from gevent._compat import get_this_psutil_process
DELAY = 0.1

class TestCloseSocketWhilePolling(greentest.TestCase):

    def test(self):
        if False:
            i = 10
            return i + 15
        sock = socket.socket()
        self._close_on_teardown(sock)
        t = get_hub().loop.timer(0)
        t.start(sock.close)
        with self.assertRaises(socket.error):
            try:
                sock.connect(('python.org', 81))
            finally:
                t.close()
        gevent.sleep(0)

class TestExceptionInMainloop(greentest.TestCase):

    def test_sleep(self):
        if False:
            return 10
        start = time.time()
        gevent.sleep(DELAY)
        delay = time.time() - start
        delay_range = DELAY * 0.9
        self.assertTimeWithinRange(delay, DELAY - delay_range, DELAY + delay_range)
        error = greentest.ExpectedException('TestExceptionInMainloop.test_sleep/fail')

        def fail():
            if False:
                print('Hello World!')
            raise error
        with get_hub().loop.timer(0.001) as t:
            t.start(fail)
            self.expect_one_error()
            start = time.time()
            gevent.sleep(DELAY)
            delay = time.time() - start
            self.assert_error(value=error)
            self.assertTimeWithinRange(delay, DELAY - delay_range, DELAY + delay_range)

class TestSleep(gevent.testing.timing.AbstractGenericWaitTestCase):

    def wait(self, timeout):
        if False:
            for i in range(10):
                print('nop')
        gevent.sleep(timeout)

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        gevent.sleep(0)

class TestWaiterGet(gevent.testing.timing.AbstractGenericWaitTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestWaiterGet, self).setUp()
        self.waiter = Waiter()

    def wait(self, timeout):
        if False:
            for i in range(10):
                print('nop')
        with get_hub().loop.timer(timeout) as evt:
            evt.start(self.waiter.switch, None)
            return self.waiter.get()

class TestWaiter(greentest.TestCase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        waiter = Waiter()
        self.assertEqual(str(waiter), '<Waiter greenlet=None>')
        waiter.switch(25)
        self.assertEqual(str(waiter), '<Waiter greenlet=None value=25>')
        self.assertEqual(waiter.get(), 25)
        waiter = Waiter()
        waiter.throw(ZeroDivisionError)
        assert re.match('^<Waiter greenlet=None exc_info=.*ZeroDivisionError.*$', str(waiter)), str(waiter)
        self.assertRaises(ZeroDivisionError, waiter.get)
        waiter = Waiter()
        g = gevent.spawn(waiter.get)
        g.name = 'AName'
        gevent.sleep(0)
        str_waiter = str(waiter)
        self.assertTrue(str_waiter.startswith('<Waiter greenlet=<Greenlet "AName'), str_waiter)
        g.kill()

@greentest.skipOnCI('Racy on CI')
class TestPeriodicMonitoringThread(greentest.TestCase):

    def _reset_hub(self):
        if False:
            print('Hello World!')
        hub = get_hub()
        try:
            del hub.exception_stream
        except AttributeError:
            pass
        if hub._threadpool is not None:
            hub.threadpool.join()
            hub.threadpool.kill()
            del hub.threadpool

    def setUp(self):
        if False:
            return 10
        super(TestPeriodicMonitoringThread, self).setUp()
        self.monitor_thread = gevent.config.monitor_thread
        gevent.config.monitor_thread = True
        from gevent.monkey import get_original
        self.lock = get_original('threading', 'Lock')()
        self.monitor_fired = 0
        self.monitored_hubs = set()
        self._reset_hub()

    def tearDown(self):
        if False:
            while True:
                i = 10
        hub = get_hub()
        if not self.monitor_thread and hub.periodic_monitoring_thread:
            hub.periodic_monitoring_thread.kill()
            hub.periodic_monitoring_thread = None
        gevent.config.monitor_thread = self.monitor_thread
        self.monitored_hubs = None
        self._reset_hub()
        super(TestPeriodicMonitoringThread, self).tearDown()

    def _monitor(self, hub):
        if False:
            while True:
                i = 10
        with self.lock:
            self.monitor_fired += 1
            if self.monitored_hubs is not None:
                self.monitored_hubs.add(hub)

    def test_config(self):
        if False:
            return 10
        self.assertEqual(0.1, gevent.config.max_blocking_time)

    def _run_monitoring_threads(self, monitor, kill=True):
        if False:
            i = 10
            return i + 15
        self.assertTrue(monitor.should_run)
        from threading import Condition
        cond = Condition()
        cond.acquire()

        def monitor_cond(_hub):
            if False:
                print('Hello World!')
            cond.acquire()
            cond.notify_all()
            cond.release()
            if kill:
                monitor.kill()
        monitor.add_monitoring_function(monitor_cond, 0.01)
        cond.wait()
        cond.release()
        monitor.add_monitoring_function(monitor_cond, None)

    @greentest.ignores_leakcheck
    def test_kill_removes_trace(self):
        if False:
            while True:
                i = 10
        from greenlet import gettrace
        hub = get_hub()
        hub.start_periodic_monitoring_thread()
        self.assertIsNotNone(gettrace())
        hub.periodic_monitoring_thread.kill()
        self.assertIsNone(gettrace())

    @greentest.ignores_leakcheck
    def test_blocking_this_thread(self):
        if False:
            return 10
        hub = get_hub()
        stream = hub.exception_stream = NativeStrIO()
        monitor = hub.start_periodic_monitoring_thread()
        self.assertIsNotNone(monitor)
        basic_monitor_func_count = 1
        if get_this_psutil_process() is not None:
            basic_monitor_func_count += 1
        self.assertEqual(basic_monitor_func_count, len(monitor.monitoring_functions()))
        monitor.add_monitoring_function(self._monitor, 0.1)
        self.assertEqual(basic_monitor_func_count + 1, len(monitor.monitoring_functions()))
        self.assertEqual(self._monitor, monitor.monitoring_functions()[-1].function)
        self.assertEqual(0.1, monitor.monitoring_functions()[-1].period)
        gevent.sleep(hub.loop.approx_timer_resolution)
        assert hub.exception_stream is stream
        try:
            time.sleep(0.3)
            self._run_monitoring_threads(monitor)
        finally:
            monitor.add_monitoring_function(self._monitor, None)
            self.assertEqual(basic_monitor_func_count, len(monitor._monitoring_functions))
            assert hub.exception_stream is stream
            monitor.kill()
            del hub.exception_stream
        self.assertGreaterEqual(self.monitor_fired, 1)
        data = stream.getvalue()
        self.assertIn('appears to be blocked', data)
        self.assertIn('PeriodicMonitoringThread', data)

    def _prep_worker_thread(self):
        if False:
            i = 10
            return i + 15
        hub = get_hub()
        threadpool = hub.threadpool
        worker_hub = threadpool.apply(get_hub)
        assert hub is not worker_hub
        stream = NativeStrIO()
        self.assertIsNone(worker_hub.periodic_monitoring_thread)

        def task():
            if False:
                while True:
                    i = 10
            get_hub().exception_stream = stream
            gevent.sleep(0.01)
            mon = get_hub().periodic_monitoring_thread
            mon.add_monitoring_function(self._monitor, 0.1)
            return mon
        worker_monitor = threadpool.apply(task)
        self.assertIsNotNone(worker_monitor)
        return (worker_hub, stream, worker_monitor)

    @greentest.ignores_leakcheck
    def test_blocking_threadpool_thread_task_queue(self):
        if False:
            while True:
                i = 10
        (worker_hub, stream, worker_monitor) = self._prep_worker_thread()
        self._run_monitoring_threads(worker_monitor)
        worker_monitor.kill()
        with self.lock:
            self.assertIn(worker_hub, self.monitored_hubs)
            self.assertEqual(stream.getvalue(), '')

    @greentest.ignores_leakcheck
    def test_blocking_threadpool_thread_one_greenlet(self):
        if False:
            return 10
        hub = get_hub()
        threadpool = hub.threadpool
        (worker_hub, stream, worker_monitor) = self._prep_worker_thread()
        task = threadpool.spawn(time.sleep, 0.3)
        self._run_monitoring_threads(worker_monitor)
        task.get()
        worker_monitor.kill()
        with self.lock:
            self.assertIn(worker_hub, self.monitored_hubs)
            self.assertEqual(stream.getvalue(), '')

    @greentest.ignores_leakcheck
    def test_blocking_threadpool_thread_multi_greenlet(self):
        if False:
            print('Hello World!')
        hub = get_hub()
        threadpool = hub.threadpool
        (worker_hub, stream, worker_monitor) = self._prep_worker_thread()

        def task():
            if False:
                print('Hello World!')
            g = gevent.spawn(time.sleep, 0.7)
            g.join()
        task = threadpool.spawn(task)
        self._run_monitoring_threads(worker_monitor, kill=False)
        task.get()
        worker_monitor.kill()
        self.assertIn(worker_hub, self.monitored_hubs)
        data = stream.getvalue()
        self.assertIn('appears to be blocked', data)
        self.assertIn('PeriodicMonitoringThread', data)

class TestLoopInterface(unittest.TestCase):

    def test_implemensts_ILoop(self):
        if False:
            i = 10
            return i + 15
        from gevent.testing import verify
        from gevent._interfaces import ILoop
        loop = get_hub().loop
        verify.verifyObject(ILoop, loop)

    def test_callback_implements_ICallback(self):
        if False:
            i = 10
            return i + 15
        from gevent.testing import verify
        from gevent._interfaces import ICallback
        loop = get_hub().loop
        cb = loop.run_callback(lambda : None)
        verify.verifyObject(ICallback, cb)

    def test_callback_ts_implements_ICallback(self):
        if False:
            i = 10
            return i + 15
        from gevent.testing import verify
        from gevent._interfaces import ICallback
        loop = get_hub().loop
        cb = loop.run_callback_threadsafe(lambda : None)
        verify.verifyObject(ICallback, cb)

class TestHandleError(unittest.TestCase):

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        try:
            del get_hub().handle_error
        except AttributeError:
            pass

    def test_exception_in_custom_handle_error_does_not_crash(self):
        if False:
            while True:
                i = 10

        def bad_handle_error(*args):
            if False:
                for i in range(10):
                    print('nop')
            raise AttributeError
        get_hub().handle_error = bad_handle_error

        class MyException(Exception):
            pass

        def raises():
            if False:
                for i in range(10):
                    print('nop')
            raise MyException
        with self.assertRaises(MyException):
            gevent.spawn(raises).get()
if __name__ == '__main__':
    greentest.main()