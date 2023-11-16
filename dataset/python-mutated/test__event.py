from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import weakref
import gevent
from gevent.event import Event, AsyncResult
import gevent.testing as greentest
from gevent.testing.six import xrange
from gevent.testing.timing import AbstractGenericGetTestCase
from gevent.testing.timing import AbstractGenericWaitTestCase
from gevent.testing.timing import SMALL_TICK
from gevent.testing.timing import SMALL_TICK_MAX_ADJ
DELAY = SMALL_TICK + SMALL_TICK_MAX_ADJ

class TestEventWait(AbstractGenericWaitTestCase):

    def wait(self, timeout):
        if False:
            i = 10
            return i + 15
        Event().wait(timeout=timeout)

    def test_cover(self):
        if False:
            while True:
                i = 10
        str(Event())

class TestGeventWaitOnEvent(AbstractGenericWaitTestCase):

    def wait(self, timeout):
        if False:
            i = 10
            return i + 15
        gevent.wait([Event()], timeout=timeout)

    def test_set_during_wait(self):
        if False:
            print('Hello World!')
        event = Event()

        def setter():
            if False:
                i = 10
                return i + 15
            event.set()

        def waiter():
            if False:
                i = 10
                return i + 15
            s = gevent.spawn(setter)
            res = event.wait()
            self.assertTrue(res)
            self.assertTrue(event.ready())
            s.join()
            event.clear()
            self.assertFalse(event.ready())
            o = gevent.wait((event,), timeout=0.01)
            self.assertFalse(event.ready())
            self.assertNotIn(event, o)
        gevent.spawn(waiter).join()

class TestAsyncResultWait(AbstractGenericWaitTestCase):

    def wait(self, timeout):
        if False:
            return 10
        AsyncResult().wait(timeout=timeout)

class TestWaitAsyncResult(AbstractGenericWaitTestCase):

    def wait(self, timeout):
        if False:
            for i in range(10):
                print('nop')
        gevent.wait([AsyncResult()], timeout=timeout)

class TestAsyncResultGet(AbstractGenericGetTestCase):

    def wait(self, timeout):
        if False:
            while True:
                i = 10
        AsyncResult().get(timeout=timeout)

class MyException(Exception):
    pass

class TestAsyncResult(greentest.TestCase):

    def test_link(self):
        if False:
            return 10
        ar = AsyncResult()
        self.assertRaises(TypeError, ar.rawlink, None)
        ar.unlink(None)
        ar.unlink(None)
        str(ar)

    def test_set_exc(self):
        if False:
            while True:
                i = 10
        log = []
        e = AsyncResult()
        self.assertEqual(e.exc_info, ())
        self.assertEqual(e.exception, None)

        def waiter():
            if False:
                for i in range(10):
                    print('nop')
            with self.assertRaises(MyException) as exc:
                e.get()
            log.append(('caught', exc.exception))
        gevent.spawn(waiter)
        obj = MyException()
        e.set_exception(obj)
        gevent.sleep(0)
        self.assertEqual(log, [('caught', obj)])

    def test_set(self):
        if False:
            while True:
                i = 10
        event1 = AsyncResult()
        timer_exc = MyException('interrupted')
        g = gevent.spawn_later(DELAY, event1.set, 'hello event1')
        self._close_on_teardown(g.kill)
        with gevent.Timeout.start_new(0, timer_exc):
            with self.assertRaises(MyException) as exc:
                event1.get()
            self.assertIs(timer_exc, exc.exception)

    def test_set_with_timeout(self):
        if False:
            print('Hello World!')
        event2 = AsyncResult()
        X = object()
        result = gevent.with_timeout(DELAY, event2.get, timeout_value=X)
        self.assertIs(result, X, 'Nobody sent anything to event2 yet it received %r' % (result,))

    def test_nonblocking_get(self):
        if False:
            i = 10
            return i + 15
        ar = AsyncResult()
        self.assertRaises(gevent.Timeout, ar.get, block=False)
        self.assertRaises(gevent.Timeout, ar.get_nowait)

class TestAsyncResultCrossThread(greentest.TestCase):

    def _makeOne(self):
        if False:
            i = 10
            return i + 15
        return AsyncResult()

    def _setOne(self, one):
        if False:
            while True:
                i = 10
        one.set('from main')
    BG_WAIT_DELAY = 60

    def _check_pypy_switch(self):
        if False:
            print('Hello World!')
        if greentest.PYPY:
            import sys
            if sys.pypy_version_info[:3] <= (7, 3, 3):
                self.skipTest('PyPy bug: https://foss.heptapod.net/pypy/pypy/-/issues/3381')

    @greentest.ignores_leakcheck
    def test_cross_thread_use(self, timed_wait=False, wait_in_bg=False):
        if False:
            return 10
        self.assertNotMonkeyPatched()
        from threading import Thread as NativeThread
        from threading import Event as NativeEvent
        if not wait_in_bg:
            self._check_pypy_switch()
        test = self

        class Thread(NativeThread):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                NativeThread.__init__(self)
                self.daemon = True
                self.running_event = NativeEvent()
                self.finished_event = NativeEvent()
                self.async_result = test._makeOne()
                self.result = '<never set>'

            def run(self):
                if False:
                    i = 10
                    return i + 15
                g_event = Event()

                def spin():
                    if False:
                        for i in range(10):
                            print('nop')
                    while not g_event.is_set():
                        g_event.wait(DELAY * 2)
                glet = gevent.spawn(spin)

                def work():
                    if False:
                        while True:
                            i = 10
                    self.running_event.set()
                    if timed_wait:
                        self.result = self.async_result.wait(test.BG_WAIT_DELAY)
                    else:
                        self.result = self.async_result.wait()
                if wait_in_bg:
                    worker = gevent.spawn(work)
                    worker.join()
                    del worker
                else:
                    work()
                g_event.set()
                glet.join()
                del glet
                self.finished_event.set()
                gevent.get_hub().destroy(destroy_loop=True)
        thread = Thread()
        thread.start()
        try:
            thread.running_event.wait()
            self._setOne(thread.async_result)
            thread.finished_event.wait(DELAY * 5)
        finally:
            thread.join(DELAY * 15)
        self._check_result(thread.result)

    def _check_result(self, result):
        if False:
            return 10
        self.assertEqual(result, 'from main')

    def test_cross_thread_use_bg(self):
        if False:
            while True:
                i = 10
        self.test_cross_thread_use(timed_wait=False, wait_in_bg=True)

    def test_cross_thread_use_timed(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_cross_thread_use(timed_wait=True, wait_in_bg=False)

    def test_cross_thread_use_timed_bg(self):
        if False:
            print('Hello World!')
        self.test_cross_thread_use(timed_wait=True, wait_in_bg=True)

    @greentest.ignores_leakcheck
    def test_cross_thread_use_set_in_bg(self):
        if False:
            return 10
        self.assertNotMonkeyPatched()
        from threading import Thread as NativeThread
        from threading import Event as NativeEvent
        self._check_pypy_switch()
        test = self

        class Thread(NativeThread):

            def __init__(self):
                if False:
                    print('Hello World!')
                NativeThread.__init__(self)
                self.daemon = True
                self.running_event = NativeEvent()
                self.finished_event = NativeEvent()
                self.async_result = test._makeOne()
                self.result = '<never set>'

            def run(self):
                if False:
                    return 10
                self.running_event.set()
                test._setOne(self.async_result)
                self.finished_event.set()
                gevent.get_hub().destroy(destroy_loop=True)
        thread = Thread()
        glet = None
        try:
            glet = gevent.spawn(thread.start)
            result = thread.async_result.wait(self.BG_WAIT_DELAY)
        finally:
            thread.join(DELAY * 15)
            if glet is not None:
                glet.join(DELAY)
        self._check_result(result)

    @greentest.ignores_leakcheck
    def test_cross_thread_use_set_in_bg2(self):
        if False:
            return 10
        self.test_cross_thread_use_set_in_bg()

class TestEventCrossThread(TestAsyncResultCrossThread):

    def _makeOne(self):
        if False:
            i = 10
            return i + 15
        return Event()

    def _setOne(self, one):
        if False:
            i = 10
            return i + 15
        one.set()

    def _check_result(self, result):
        if False:
            return 10
        self.assertTrue(result)

class TestAsyncResultAsLinkTarget(greentest.TestCase):
    error_fatal = False

    def test_set(self):
        if False:
            print('Hello World!')
        g = gevent.spawn(lambda : 1)
        (s1, s2, s3) = (AsyncResult(), AsyncResult(), AsyncResult())
        g.link(s1)
        g.link_value(s2)
        g.link_exception(s3)
        self.assertEqual(s1.get(), 1)
        self.assertEqual(s2.get(), 1)
        X = object()
        result = gevent.with_timeout(DELAY, s3.get, timeout_value=X)
        self.assertIs(result, X)

    def test_set_exception(self):
        if False:
            i = 10
            return i + 15

        def func():
            if False:
                while True:
                    i = 10
            raise greentest.ExpectedException('TestAsyncResultAsLinkTarget.test_set_exception')
        g = gevent.spawn(func)
        (s1, s2, s3) = (AsyncResult(), AsyncResult(), AsyncResult())
        g.link(s1)
        g.link_value(s2)
        g.link_exception(s3)
        self.assertRaises(greentest.ExpectedException, s1.get)
        X = object()
        result = gevent.with_timeout(DELAY, s2.get, timeout_value=X)
        self.assertIs(result, X)
        self.assertRaises(greentest.ExpectedException, s3.get)

class TestEvent_SetThenClear(greentest.TestCase):
    N = 1

    def test(self):
        if False:
            while True:
                i = 10
        e = Event()
        waiters = [gevent.spawn(e.wait) for i in range(self.N)]
        gevent.sleep(0.001)
        e.set()
        e.clear()
        for greenlet in waiters:
            greenlet.join()

class TestEvent_SetThenClear100(TestEvent_SetThenClear):
    N = 100

class TestEvent_SetThenClear1000(TestEvent_SetThenClear):
    N = 1000

class TestWait(greentest.TestCase):
    N = 5
    count = None
    timeout = 1
    period = timeout / 100.0

    def _sender(self, events, asyncs):
        if False:
            i = 10
            return i + 15
        while events or asyncs:
            gevent.sleep(self.period)
            if events:
                events.pop().set()
            gevent.sleep(self.period)
            if asyncs:
                asyncs.pop().set()

    @greentest.skipOnAppVeyor('Not all results have arrived sometimes due to timer issues')
    def test(self):
        if False:
            return 10
        events = [Event() for _ in xrange(self.N)]
        asyncs = [AsyncResult() for _ in xrange(self.N)]
        max_len = len(events) + len(asyncs)
        sender = gevent.spawn(self._sender, events, asyncs)
        results = gevent.wait(events + asyncs, count=self.count, timeout=self.timeout)
        if self.timeout is None:
            expected_len = max_len
        else:
            expected_len = min(max_len, self.timeout / self.period)
        if self.count is None:
            self.assertTrue(sender.ready(), sender)
        else:
            expected_len = min(self.count, expected_len)
            self.assertFalse(sender.ready(), sender)
            sender.kill()
        self.assertEqual(expected_len, len(results), (expected_len, len(results), results))

class TestWait_notimeout(TestWait):
    timeout = None

class TestWait_count1(TestWait):
    count = 1

class TestWait_count2(TestWait):
    count = 2

class TestEventBasics(greentest.TestCase):

    def test_weakref(self):
        if False:
            print('Hello World!')
        e = Event()
        r = weakref.ref(e)
        self.assertIs(e, r())
        del e
        del r

    def test_wait_while_notifying(self):
        if False:
            for i in range(10):
                print('nop')
        event = Event()
        results = []

        def wait_then_append(arg):
            if False:
                for i in range(10):
                    print('nop')
            event.wait()
            results.append(arg)
        gevent.spawn(wait_then_append, 1)
        gevent.spawn(wait_then_append, 2)
        gevent.idle()
        self.assertEqual(2, event.linkcount())
        check = gevent.get_hub().loop.check()
        check.start(results.append, 4)
        event.set()
        wait_then_append(3)
        self.assertEqual(results, [1, 2, 3])
        check.stop()
        check.close()

    def test_gevent_wait_twice_when_already_set(self):
        if False:
            print('Hello World!')
        event = Event()
        event.set()
        result = gevent.wait([event])
        self.assertEqual(result, [event])
        result = gevent.wait([event])
        self.assertEqual(result, [event])
del AbstractGenericGetTestCase
del AbstractGenericWaitTestCase
if __name__ == '__main__':
    greentest.main()