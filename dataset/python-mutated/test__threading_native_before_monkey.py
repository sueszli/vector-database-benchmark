import threading
from time import sleep as time_sleep
import gevent.testing as greentest

class NativeThread(threading.Thread):
    do_run = True

    def run(self):
        if False:
            i = 10
            return i + 15
        while self.do_run:
            time_sleep(0.1)

    def stop(self, timeout=None):
        if False:
            while True:
                i = 10
        self.do_run = False
        self.join(timeout=timeout)
native_thread = None

class Test(greentest.TestCase):

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        global native_thread
        if native_thread is not None:
            native_thread.stop(1)
            native_thread = None

    def test_main_thread(self):
        if False:
            return 10
        current = threading.current_thread()
        self.assertNotIsInstance(current, threading._DummyThread)
        self.assertIsInstance(current, monkey.get_original('threading', 'Thread'))
        repr(current)
        if hasattr(threading, 'main_thread'):
            self.assertEqual(threading.current_thread(), threading.main_thread())

    @greentest.ignores_leakcheck
    def test_join_native_thread(self):
        if False:
            return 10
        if native_thread is None or not native_thread.do_run:
            self.skipTest('native_thread already closed')
        self.assertTrue(native_thread.is_alive())
        native_thread.stop(timeout=1)
        self.assertFalse(native_thread.is_alive())
        native_thread.stop()
        self.assertFalse(native_thread.is_alive())
if __name__ == '__main__':
    native_thread = NativeThread()
    native_thread.daemon = True
    native_thread.start()
    from gevent import monkey
    monkey.patch_all()
    greentest.main()