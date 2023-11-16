from __future__ import print_function
from __future__ import absolute_import
import weakref
import gevent
import gevent.exceptions
from gevent.lock import Semaphore
from gevent.lock import BoundedSemaphore
import gevent.testing as greentest
from gevent.testing import timing

class TestSemaphore(greentest.TestCase):

    def test_acquire_returns_false_after_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        s = Semaphore(value=0)
        result = s.acquire(timeout=0.01)
        assert result is False, repr(result)

    def test_release_twice(self):
        if False:
            return 10
        s = Semaphore()
        result = []
        s.rawlink(lambda s: result.append('a'))
        s.release()
        s.rawlink(lambda s: result.append('b'))
        s.release()
        gevent.sleep(0.001)
        self.assertEqual(sorted(result), ['a', 'b'])

    def test_semaphore_weakref(self):
        if False:
            for i in range(10):
                print('nop')
        s = Semaphore()
        r = weakref.ref(s)
        self.assertEqual(s, r())

    @greentest.ignores_leakcheck
    def test_semaphore_in_class_with_del(self):
        if False:
            while True:
                i = 10

        class X(object):

            def __init__(self):
                if False:
                    return 10
                self.s = Semaphore()

            def __del__(self):
                if False:
                    print('Hello World!')
                self.s.acquire()
        X()
        import gc
        gc.collect()
        gc.collect()

    def test_rawlink_on_unacquired_runs_notifiers(self):
        if False:
            for i in range(10):
                print('nop')
        s = Semaphore()
        gevent.wait([s])

class TestSemaphoreMultiThread(greentest.TestCase):

    def _getTargetClass(self):
        if False:
            i = 10
            return i + 15
        return Semaphore

    def _makeOne(self):
        if False:
            return 10
        return self._getTargetClass()(1)

    def _makeThreadMain(self, thread_running, thread_acquired, sem, acquired, exc_info, **thread_acquire_kwargs):
        if False:
            print('Hello World!')
        from gevent._hub_local import get_hub_if_exists
        import sys

        def thread_main():
            if False:
                for i in range(10):
                    print('nop')
            thread_running.set()
            try:
                acquired.append(sem.acquire(**thread_acquire_kwargs))
            except:
                exc_info[:] = sys.exc_info()
                raise
            finally:
                hub = get_hub_if_exists()
                if hub is not None:
                    hub.join()
                    hub.destroy(destroy_loop=True)
                thread_acquired.set()
        return thread_main
    IDLE_ITERATIONS = 5

    def _do_test_acquire_in_one_then_another(self, release=True, require_thread_acquired_to_finish=False, **thread_acquire_kwargs):
        if False:
            i = 10
            return i + 15
        from gevent import monkey
        self.assertFalse(monkey.is_module_patched('threading'))
        import threading
        thread_running = threading.Event()
        thread_acquired = threading.Event()
        sem = self._makeOne()
        sem.acquire()
        exc_info = []
        acquired = []
        t = threading.Thread(target=self._makeThreadMain(thread_running, thread_acquired, sem, acquired, exc_info, **thread_acquire_kwargs))
        t.daemon = True
        t.start()
        thread_running.wait(10)
        if release:
            sem.release()
            for _ in range(self.IDLE_ITERATIONS):
                gevent.idle()
                if thread_acquired.wait(timing.LARGE_TICK):
                    break
            self.assertEqual(acquired, [True])
        if not release and thread_acquire_kwargs.get('timeout'):
            for _ in range(self.IDLE_ITERATIONS):
                gevent.idle()
                if thread_acquired.wait(timing.LARGE_TICK):
                    break
        thread_acquired.wait(timing.LARGE_TICK * 5)
        if require_thread_acquired_to_finish:
            self.assertTrue(thread_acquired.is_set())
        try:
            self.assertEqual(exc_info, [])
        finally:
            exc_info = None
        return (sem, acquired)

    def test_acquire_in_one_then_another(self):
        if False:
            for i in range(10):
                print('nop')
        self._do_test_acquire_in_one_then_another(release=True)

    def test_acquire_in_one_then_another_timed(self):
        if False:
            print('Hello World!')
        (sem, acquired_in_thread) = self._do_test_acquire_in_one_then_another(release=False, require_thread_acquired_to_finish=True, timeout=timing.SMALLEST_RELIABLE_DELAY)
        self.assertEqual([False], acquired_in_thread)
        sem.release()
        notifier = getattr(sem, '_notifier', None)
        self.assertIsNone(notifier)

    def test_acquire_in_one_wait_greenlet_wait_thread_gives_up(self):
        if False:
            return 10
        from gevent import monkey
        self.assertFalse(monkey.is_module_patched('threading'))
        import threading
        sem = self._makeOne()
        sem.acquire()

        def greenlet_one():
            if False:
                i = 10
                return i + 15
            ack = sem.acquire()
            thread.start()
            gevent.sleep(timing.LARGE_TICK)
            return ack
        exc_info = []
        acquired = []
        glet = gevent.spawn(greenlet_one)
        thread = threading.Thread(target=self._makeThreadMain(threading.Event(), threading.Event(), sem, acquired, exc_info, timeout=timing.LARGE_TICK))
        thread.daemon = True
        gevent.idle()
        sem.release()
        glet.join()
        for _ in range(3):
            gevent.idle()
            thread.join(timing.LARGE_TICK)
        self.assertEqual(glet.value, True)
        self.assertEqual([], exc_info)
        self.assertEqual([False], acquired)
        self.assertTrue(glet.dead, glet)
        glet = None

    def assertOneHasNoHub(self, sem):
        if False:
            while True:
                i = 10
        self.assertIsNone(sem.hub, sem)

    @greentest.skipOnPyPyOnWindows("Flaky there; can't reproduce elsewhere")
    def test_dueling_threads(self, acquire_args=(), create_hub=None):
        if False:
            return 10
        from gevent import monkey
        from gevent._hub_local import get_hub_if_exists
        self.assertFalse(monkey.is_module_patched('threading'))
        import threading
        from time import sleep as native_sleep
        sem = self._makeOne()
        self.assertOneHasNoHub(sem)
        count = 10000
        results = [-1, -1]
        run = True

        def do_it(ix):
            if False:
                i = 10
                return i + 15
            if create_hub:
                gevent.get_hub()
            try:
                for i in range(count):
                    if not run:
                        break
                    acquired = sem.acquire(*acquire_args)
                    assert acquire_args or acquired
                    if acquired:
                        sem.release()
                    results[ix] = i
                    if not create_hub:
                        self.assertIsNone(get_hub_if_exists(), (get_hub_if_exists(), ix, i))
                    if create_hub and i % 10 == 0:
                        gevent.sleep(timing.SMALLEST_RELIABLE_DELAY)
                    elif i % 100 == 0:
                        native_sleep(timing.SMALLEST_RELIABLE_DELAY)
            except Exception as ex:
                import traceback
                traceback.print_exc()
                results[ix] = str(ex)
                ex = None
            finally:
                hub = get_hub_if_exists()
                if hub is not None:
                    hub.join()
                    hub.destroy(destroy_loop=True)
        t1 = threading.Thread(target=do_it, args=(0,))
        t1.daemon = True
        t2 = threading.Thread(target=do_it, args=(1,))
        t2.daemon = True
        t1.start()
        t2.start()
        t1.join(1)
        t2.join(1)
        while t1.is_alive() or t2.is_alive():
            cur = list(results)
            t1.join(7)
            t2.join(7)
            if cur == results:
                run = False
                break
        self.assertEqual(results, [count - 1, count - 1])

    def test_dueling_threads_timeout(self):
        if False:
            i = 10
            return i + 15
        self.test_dueling_threads((True, 4))

    def test_dueling_threads_with_hub(self):
        if False:
            i = 10
            return i + 15
        self.test_dueling_threads(create_hub=True)

class TestBoundedSemaphoreMultiThread(TestSemaphoreMultiThread):

    def _getTargetClass(self):
        if False:
            for i in range(10):
                print('nop')
        return BoundedSemaphore

@greentest.skipOnPurePython('Needs C extension')
class TestCExt(greentest.TestCase):

    def test_c_extension(self):
        if False:
            return 10
        self.assertEqual(Semaphore.__module__, 'gevent._gevent_c_semaphore')

class SwitchWithFixedHash(object):

    def __init__(self, greenlet, hashcode):
        if False:
            return 10
        self.switch = greenlet.switch
        self.hashcode = hashcode

    def __hash__(self):
        if False:
            return 10
        raise AssertionError

    def __eq__(self, other):
        if False:
            return 10
        raise AssertionError

    def __call__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.switch(*args, **kwargs)

    def __repr__(self):
        if False:
            print('Hello World!')
        return repr(self.switch)

class FirstG(gevent.Greenlet):
    hashcode = 10

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        gevent.Greenlet.__init__(self, *args, **kwargs)
        self.switch = SwitchWithFixedHash(self, self.hashcode)

class LastG(FirstG):
    hashcode = 12

def acquire_then_exit(sem, should_quit):
    if False:
        return 10
    sem.acquire()
    should_quit.append(True)

def acquire_then_spawn(sem, should_quit):
    if False:
        print('Hello World!')
    if should_quit:
        return
    sem.acquire()
    g = FirstG.spawn(release_then_spawn, sem, should_quit)
    g.join()

def release_then_spawn(sem, should_quit):
    if False:
        print('Hello World!')
    sem.release()
    if should_quit:
        return
    g = FirstG.spawn(acquire_then_spawn, sem, should_quit)
    g.join()

class TestSemaphoreFair(greentest.TestCase):

    def test_fair_or_hangs(self):
        if False:
            while True:
                i = 10
        sem = Semaphore()
        should_quit = []
        keep_going1 = FirstG.spawn(acquire_then_spawn, sem, should_quit)
        keep_going2 = FirstG.spawn(acquire_then_spawn, sem, should_quit)
        exiting = LastG.spawn(acquire_then_exit, sem, should_quit)
        with self.assertRaises(gevent.exceptions.LoopExit):
            gevent.joinall([keep_going1, keep_going2, exiting])
        self.assertTrue(exiting.dead, exiting)
        self.assertTrue(keep_going2.dead, keep_going2)
        self.assertFalse(keep_going1.dead, keep_going1)
        sem.release()
        keep_going1.kill()
        keep_going2.kill()
        exiting.kill()
        gevent.idle()
if __name__ == '__main__':
    greentest.main()