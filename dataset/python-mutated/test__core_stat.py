from __future__ import print_function
import os
import tempfile
import time
import gevent
import gevent.core
import gevent.testing as greentest
import gevent.testing.flaky
DELAY = 0.5
WIN = greentest.WIN
LIBUV = greentest.LIBUV

class TestCoreStat(greentest.TestCase):
    __timeout__ = greentest.LARGE_TIMEOUT

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestCoreStat, self).setUp()
        (fd, path) = tempfile.mkstemp(suffix='.gevent_test_core_stat')
        os.close(fd)
        self.temp_path = path
        self.hub = gevent.get_hub()
        self.watcher = self.hub.loop.stat(self.temp_path, interval=-1)

    def tearDown(self):
        if False:
            return 10
        self.watcher.close()
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)
        super(TestCoreStat, self).tearDown()

    def _write(self):
        if False:
            i = 10
            return i + 15
        with open(self.temp_path, 'wb', buffering=0) as f:
            f.write(b'x')

    def _check_attr(self, name, none):
        if False:
            i = 10
            return i + 15
        try:
            x = getattr(self.watcher, name)
        except ImportError:
            if WIN:
                pass
            else:
                raise
        else:
            if WIN and (not LIBUV):
                self.assertIsNone(x, 'Only None is supported on Windows')
            if none:
                self.assertIsNone(x, name)
            else:
                self.assertIsNotNone(x, name)

    def _wait_on_greenlet(self, func, *greenlet_args):
        if False:
            return 10
        start = time.time()
        self.hub.loop.update_now()
        greenlet = gevent.spawn_later(DELAY, func, *greenlet_args)
        with gevent.Timeout(5 + DELAY + 0.5):
            self.hub.wait(self.watcher)
        now = time.time()
        self.assertGreaterEqual(now, start, 'Time must move forward')
        wait_duration = now - start
        reaction = wait_duration - DELAY
        if reaction <= 0.0:
            raise gevent.testing.flaky.FlakyTestRaceCondition('Bad timer resolution (on Windows?), test is useless. Start %s, now %s' % (start, now))
        self.assertGreaterEqual(reaction, 0.0, 'Watcher %s reacted too early: %.3fs' % (self.watcher, reaction))
        greenlet.join()

    def test_watcher_basics(self):
        if False:
            for i in range(10):
                print('nop')
        watcher = self.watcher
        filename = self.temp_path
        self.assertEqual(watcher.path, filename)
        filenames = filename if isinstance(filename, bytes) else filename.encode('ascii')
        self.assertEqual(watcher._paths, filenames)
        self.assertEqual(watcher.interval, -1)

    def test_write(self):
        if False:
            i = 10
            return i + 15
        self._wait_on_greenlet(self._write)
        self._check_attr('attr', False)
        self._check_attr('prev', False)
        self.assertNotEqual(self.watcher.interval, -1)

    def test_unlink(self):
        if False:
            i = 10
            return i + 15
        self._wait_on_greenlet(os.unlink, self.temp_path)
        self._check_attr('attr', True)
        self._check_attr('prev', False)
if __name__ == '__main__':
    greentest.main()