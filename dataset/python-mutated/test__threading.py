"""
Tests specifically for the monkey-patched threading module.
"""
from gevent import monkey
monkey.patch_all()
import gevent.hub
assert gevent.hub._get_hub() is None, 'monkey.patch_all() should not init hub'
import gevent
import gevent.testing as greentest
import threading

def helper():
    if False:
        i = 10
        return i + 15
    threading.current_thread()
    gevent.sleep(0.2)

class TestCleanup(greentest.TestCase):

    def _do_test(self, spawn):
        if False:
            for i in range(10):
                print('nop')
        before = len(threading._active)
        g = spawn(helper)
        gevent.sleep(0.1)
        self.assertEqual(len(threading._active), before + 1)
        try:
            g.join()
        except AttributeError:
            while not g.dead:
                gevent.sleep()
            del g
        self.assertEqual(len(threading._active), before)

    def test_cleanup_gevent(self):
        if False:
            while True:
                i = 10
        self._do_test(gevent.spawn)

    @greentest.skipOnPyPy('weakref is not cleaned up in a timely fashion')
    def test_cleanup_raw(self):
        if False:
            for i in range(10):
                print('nop')
        self._do_test(gevent.spawn_raw)

class TestLockThread(greentest.TestCase):

    def _spawn(self, func):
        if False:
            print('Hello World!')
        t = threading.Thread(target=func)
        t.start()
        return t

    def test_spin_lock_switches(self):
        if False:
            while True:
                i = 10
        lock = threading.Lock()
        lock.acquire()
        spawned = []

        def background():
            if False:
                return 10
            spawned.append(True)
            while not lock.acquire(False):
                pass
        thread = threading.Thread(target=background)
        thread.start()
        self.assertEqual(spawned, [True])
        thread.join(0)
        lock.release()
        thread.join()

class TestLockGreenlet(TestLockThread):

    def _spawn(self, func):
        if False:
            return 10
        return gevent.spawn(func)
if __name__ == '__main__':
    greentest.main()