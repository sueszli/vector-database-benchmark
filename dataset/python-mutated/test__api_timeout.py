import sys
import gevent.testing as greentest
import weakref
import time
import gc
from gevent import sleep
from gevent import Timeout
from gevent import get_hub
from gevent.testing.timing import SMALL_TICK as DELAY
from gevent.testing import flaky

class Error(Exception):
    pass

class _UpdateNowProxy(object):
    update_now_calls = 0

    def __init__(self, loop):
        if False:
            for i in range(10):
                print('nop')
        self.loop = loop

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self.loop, name)

    def update_now(self):
        if False:
            i = 10
            return i + 15
        self.update_now_calls += 1
        self.loop.update_now()

class _UpdateNowWithTimerProxy(_UpdateNowProxy):

    def timer(self, *_args, **_kwargs):
        if False:
            return 10
        return _Timer(self)

class _Timer(object):
    pending = False
    active = False

    def __init__(self, loop):
        if False:
            return 10
        self.loop = loop

    def start(self, *_args, **kwargs):
        if False:
            return 10
        if kwargs.get('update'):
            self.loop.update_now()
        self.pending = self.active = True

    def stop(self):
        if False:
            print('Hello World!')
        self.active = self.pending = False

    def close(self):
        if False:
            while True:
                i = 10
        'Does nothing'

class Test(greentest.TestCase):

    def test_timeout_calls_update_now(self):
        if False:
            for i in range(10):
                print('nop')
        hub = get_hub()
        loop = hub.loop
        proxy = _UpdateNowWithTimerProxy(loop)
        hub.loop = proxy
        try:
            with Timeout(DELAY * 2) as t:
                self.assertTrue(t.pending)
        finally:
            hub.loop = loop
        self.assertEqual(1, proxy.update_now_calls)

    def test_sleep_calls_update_now(self):
        if False:
            print('Hello World!')
        hub = get_hub()
        loop = hub.loop
        proxy = _UpdateNowProxy(loop)
        hub.loop = proxy
        try:
            sleep(0.01)
        finally:
            hub.loop = loop
        self.assertEqual(1, proxy.update_now_calls)

    @greentest.skipOnAppVeyor('Timing is flaky, especially under Py 3.4/64-bit')
    @greentest.skipOnPyPy3OnCI('Timing is flaky, especially under Py 3.4/64-bit')
    @greentest.reraises_flaky_timeout((Timeout, AssertionError))
    def test_api(self):
        if False:
            i = 10
            return i + 15
        t = Timeout(DELAY * 2)
        self.assertFalse(t.pending, t)
        with t:
            self.assertTrue(t.pending, t)
            sleep(DELAY)
        self.assertFalse(t.pending, t)
        sleep(DELAY * 2)
        with self.assertRaises(Timeout) as exc:
            with Timeout(DELAY) as t:
                sleep(DELAY * 10)
        self.assertIs(exc.exception, t)
        with self.assertRaises(IOError):
            with Timeout(DELAY, IOError('Operation takes way too long')):
                sleep(DELAY * 10)
        with self.assertRaises(ValueError):
            with Timeout(DELAY, ValueError):
                sleep(DELAY * 10)
        try:
            1 / 0
        except ZeroDivisionError:
            with self.assertRaises(ZeroDivisionError):
                with Timeout(DELAY, sys.exc_info()[0]):
                    sleep(DELAY * 10)
                    raise AssertionError('should not get there')
                raise AssertionError('should not get there')
        else:
            raise AssertionError('should not get there')
        with Timeout(DELAY) as timer:
            timer.cancel()
            sleep(DELAY * 2)
        XDELAY = 0.1
        start = time.time()
        with Timeout(XDELAY, False):
            sleep(XDELAY * 2)
        delta = time.time() - start
        self.assertTimeWithinRange(delta, 0, XDELAY * 2)
        with Timeout(None):
            sleep(DELAY)
        sleep(DELAY)

    def test_ref(self):
        if False:
            for i in range(10):
                print('nop')
        err = Error()
        err_ref = weakref.ref(err)
        with Timeout(DELAY * 2, err):
            sleep(DELAY)
        del err
        gc.collect()
        self.assertFalse(err_ref(), err_ref)

    @flaky.reraises_flaky_race_condition()
    def test_nested_timeout(self):
        if False:
            print('Hello World!')
        with Timeout(DELAY, False):
            with Timeout(DELAY * 10, False):
                sleep(DELAY * 3 * 20)
            raise AssertionError('should not get there')
        with Timeout(DELAY) as t1:
            with Timeout(DELAY * 20) as t2:
                with self.assertRaises(Timeout) as exc:
                    sleep(DELAY * 30)
                self.assertIs(exc.exception, t1)
                self.assertFalse(t1.pending, t1)
                self.assertTrue(t2.pending, t2)
            self.assertFalse(t2.pending)
        with Timeout(DELAY * 20) as t1:
            with Timeout(DELAY) as t2:
                with self.assertRaises(Timeout) as exc:
                    sleep(DELAY * 30)
                self.assertIs(exc.exception, t2)
                self.assertTrue(t1.pending, t1)
                self.assertFalse(t2.pending, t2)
        self.assertFalse(t1.pending)
if __name__ == '__main__':
    greentest.main()