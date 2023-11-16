from __future__ import print_function
from gevent import config
import gevent.testing as greentest
from gevent.testing import TestCase
from gevent.testing import LARGE_TIMEOUT
from gevent.testing.sysinfo import CFFI_BACKEND
from gevent.testing.flaky import reraises_flaky_timeout

class Test(TestCase):
    __timeout__ = LARGE_TIMEOUT
    repeat = 0
    timer_duration = 0.001

    def setUp(self):
        if False:
            return 10
        super(Test, self).setUp()
        self.called = []
        self.loop = config.loop(default=False)
        self.timer = self.loop.timer(self.timer_duration, repeat=self.repeat)
        assert not self.loop.default

    def cleanup(self):
        if False:
            return 10
        self.timer.close()
        self.loop.run()
        self.loop.destroy()
        self.loop = None
        self.timer = None

    def f(self, x=None):
        if False:
            print('Hello World!')
        self.called.append(1)
        if x is not None:
            x.stop()

    def assertTimerInKeepalive(self):
        if False:
            return 10
        if CFFI_BACKEND:
            self.assertIn(self.timer, self.loop._keepaliveset)

    def assertTimerNotInKeepalive(self):
        if False:
            return 10
        if CFFI_BACKEND:
            self.assertNotIn(self.timer, self.loop._keepaliveset)

    def test_main(self):
        if False:
            i = 10
            return i + 15
        loop = self.loop
        x = self.timer
        x.start(self.f)
        self.assertTimerInKeepalive()
        self.assertTrue(x.active, x)
        with self.assertRaises((AttributeError, ValueError)):
            x.priority = 1
        loop.run()
        self.assertEqual(x.pending, 0)
        self.assertEqual(self.called, [1])
        self.assertIsNone(x.callback)
        self.assertIsNone(x.args)
        if x.priority is not None:
            self.assertEqual(x.priority, 0)
            x.priority = 1
            self.assertEqual(x.priority, 1)
        x.stop()
        self.assertTimerNotInKeepalive()

class TestAgain(Test):
    repeat = 1

    def test_main(self):
        if False:
            while True:
                i = 10
        x = self.timer
        x.again(self.f, x)
        self.assertTimerInKeepalive()
        self.assertEqual(x.args, (x,))
        self.loop.run()
        self.assertEqual(self.called, [1])
        x.stop()
        self.assertTimerNotInKeepalive()

class TestTimerResolution(Test):

    @reraises_flaky_timeout(AssertionError)
    def test_resolution(self):
        if False:
            for i in range(10):
                print('nop')
        from gevent._compat import perf_counter
        import socket
        s = socket.socket()
        self._close_on_teardown(s)
        fd = s.fileno()
        ran_at_least_once = False
        fired_at = []

        def timer_counter():
            if False:
                i = 10
                return i + 15
            fired_at.append(perf_counter())
        loop = self.loop
        timer_multiplier = 11
        max_time = self.timer_duration * timer_multiplier
        assert max_time < 0.3
        for _ in range(150):
            io = loop.io(fd, 1)
            io.start(lambda events=None: None)
            now = perf_counter()
            del fired_at[:]
            timer = self.timer
            timer.start(timer_counter)
            loop.run(once=True)
            io.stop()
            io.close()
            timer.stop()
            if fired_at:
                ran_at_least_once = True
                self.assertEqual(1, len(fired_at))
                self.assertTimeWithinRange(fired_at[0] - now, 0, max_time)
        if not greentest.RUNNING_ON_CI:
            self.assertTrue(ran_at_least_once)
if __name__ == '__main__':
    greentest.main()