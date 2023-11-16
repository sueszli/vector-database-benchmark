"""Tests for Coordinator."""
import sys
import threading
import time
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator

def StopOnEvent(coord, wait_for_stop, set_when_stopped):
    if False:
        for i in range(10):
            print('nop')
    wait_for_stop.wait()
    coord.request_stop()
    set_when_stopped.set()

def RaiseOnEvent(coord, wait_for_stop, set_when_stopped, ex, report_exception):
    if False:
        while True:
            i = 10
    try:
        wait_for_stop.wait()
        raise ex
    except RuntimeError as e:
        if report_exception:
            coord.request_stop(e)
        else:
            coord.request_stop(sys.exc_info())
    finally:
        if set_when_stopped:
            set_when_stopped.set()

def RaiseOnEventUsingContextHandler(coord, wait_for_stop, set_when_stopped, ex):
    if False:
        for i in range(10):
            print('nop')
    with coord.stop_on_exception():
        wait_for_stop.wait()
        raise ex
    if set_when_stopped:
        set_when_stopped.set()

def SleepABit(n_secs, coord=None):
    if False:
        i = 10
        return i + 15
    if coord:
        coord.register_thread(threading.current_thread())
    time.sleep(n_secs)

def WaitForThreadsToRegister(coord, num_threads):
    if False:
        for i in range(10):
            print('nop')
    while True:
        with coord._lock:
            if len(coord._registered_threads) == num_threads:
                break
        time.sleep(0.001)

class CoordinatorTest(test.TestCase):

    def testStopAPI(self):
        if False:
            return 10
        coord = coordinator.Coordinator()
        self.assertFalse(coord.should_stop())
        self.assertFalse(coord.wait_for_stop(0.01))
        coord.request_stop()
        self.assertTrue(coord.should_stop())
        self.assertTrue(coord.wait_for_stop(0.01))

    def testStopAsync(self):
        if False:
            while True:
                i = 10
        coord = coordinator.Coordinator()
        self.assertFalse(coord.should_stop())
        self.assertFalse(coord.wait_for_stop(0.1))
        wait_for_stop_ev = threading.Event()
        has_stopped_ev = threading.Event()
        t = threading.Thread(target=StopOnEvent, args=(coord, wait_for_stop_ev, has_stopped_ev))
        t.start()
        self.assertFalse(coord.should_stop())
        self.assertFalse(coord.wait_for_stop(0.01))
        wait_for_stop_ev.set()
        has_stopped_ev.wait()
        self.assertTrue(coord.wait_for_stop(0.05))
        self.assertTrue(coord.should_stop())

    def testJoin(self):
        if False:
            while True:
                i = 10
        coord = coordinator.Coordinator()
        threads = [threading.Thread(target=SleepABit, args=(0.01,)), threading.Thread(target=SleepABit, args=(0.02,)), threading.Thread(target=SleepABit, args=(0.01,))]
        for t in threads:
            t.start()
        coord.join(threads)
        for t in threads:
            self.assertFalse(t.is_alive())

    def testJoinAllRegistered(self):
        if False:
            print('Hello World!')
        coord = coordinator.Coordinator()
        threads = [threading.Thread(target=SleepABit, args=(0.01, coord)), threading.Thread(target=SleepABit, args=(0.02, coord)), threading.Thread(target=SleepABit, args=(0.01, coord))]
        for t in threads:
            t.start()
        WaitForThreadsToRegister(coord, 3)
        coord.join()
        for t in threads:
            self.assertFalse(t.is_alive())

    def testJoinSomeRegistered(self):
        if False:
            print('Hello World!')
        coord = coordinator.Coordinator()
        threads = [threading.Thread(target=SleepABit, args=(0.01, coord)), threading.Thread(target=SleepABit, args=(0.02,)), threading.Thread(target=SleepABit, args=(0.01, coord))]
        for t in threads:
            t.start()
        WaitForThreadsToRegister(coord, 2)
        coord.join([threads[1]])
        for t in threads:
            self.assertFalse(t.is_alive())

    def testJoinGraceExpires(self):
        if False:
            i = 10
            return i + 15

        def TestWithGracePeriod(stop_grace_period):
            if False:
                for i in range(10):
                    print('nop')
            coord = coordinator.Coordinator()
            wait_for_stop_ev = threading.Event()
            has_stopped_ev = threading.Event()
            threads = [threading.Thread(target=StopOnEvent, args=(coord, wait_for_stop_ev, has_stopped_ev)), threading.Thread(target=SleepABit, args=(10.0,))]
            for t in threads:
                t.daemon = True
                t.start()
            wait_for_stop_ev.set()
            has_stopped_ev.wait()
            with self.assertRaisesRegex(RuntimeError, 'threads still running'):
                coord.join(threads, stop_grace_period_secs=stop_grace_period)
        TestWithGracePeriod(1e-10)
        TestWithGracePeriod(0.002)
        TestWithGracePeriod(1.0)

    def testJoinWithoutGraceExpires(self):
        if False:
            return 10
        coord = coordinator.Coordinator()
        wait_for_stop_ev = threading.Event()
        has_stopped_ev = threading.Event()
        threads = [threading.Thread(target=StopOnEvent, args=(coord, wait_for_stop_ev, has_stopped_ev)), threading.Thread(target=SleepABit, args=(10.0,))]
        for t in threads:
            t.daemon = True
            t.start()
        wait_for_stop_ev.set()
        has_stopped_ev.wait()
        coord.join(threads, stop_grace_period_secs=1.0, ignore_live_threads=True)

    def testJoinRaiseReportExcInfo(self):
        if False:
            for i in range(10):
                print('nop')
        coord = coordinator.Coordinator()
        ev_1 = threading.Event()
        ev_2 = threading.Event()
        threads = [threading.Thread(target=RaiseOnEvent, args=(coord, ev_1, ev_2, RuntimeError('First'), False)), threading.Thread(target=RaiseOnEvent, args=(coord, ev_2, None, RuntimeError('Too late'), False))]
        for t in threads:
            t.start()
        ev_1.set()
        with self.assertRaisesRegex(RuntimeError, 'First'):
            coord.join(threads)

    def testJoinRaiseReportException(self):
        if False:
            return 10
        coord = coordinator.Coordinator()
        ev_1 = threading.Event()
        ev_2 = threading.Event()
        threads = [threading.Thread(target=RaiseOnEvent, args=(coord, ev_1, ev_2, RuntimeError('First'), True)), threading.Thread(target=RaiseOnEvent, args=(coord, ev_2, None, RuntimeError('Too late'), True))]
        for t in threads:
            t.start()
        ev_1.set()
        with self.assertRaisesRegex(RuntimeError, 'First'):
            coord.join(threads)

    def testJoinIgnoresOutOfRange(self):
        if False:
            i = 10
            return i + 15
        coord = coordinator.Coordinator()
        ev_1 = threading.Event()
        threads = [threading.Thread(target=RaiseOnEvent, args=(coord, ev_1, None, errors_impl.OutOfRangeError(None, None, 'First'), True))]
        for t in threads:
            t.start()
        ev_1.set()
        coord.join(threads)

    def testJoinIgnoresMyExceptionType(self):
        if False:
            while True:
                i = 10
        coord = coordinator.Coordinator(clean_stop_exception_types=(ValueError,))
        ev_1 = threading.Event()
        threads = [threading.Thread(target=RaiseOnEvent, args=(coord, ev_1, None, ValueError('Clean stop'), True))]
        for t in threads:
            t.start()
        ev_1.set()
        coord.join(threads)

    def testJoinRaiseReportExceptionUsingHandler(self):
        if False:
            print('Hello World!')
        coord = coordinator.Coordinator()
        ev_1 = threading.Event()
        ev_2 = threading.Event()
        threads = [threading.Thread(target=RaiseOnEventUsingContextHandler, args=(coord, ev_1, ev_2, RuntimeError('First'))), threading.Thread(target=RaiseOnEventUsingContextHandler, args=(coord, ev_2, None, RuntimeError('Too late')))]
        for t in threads:
            t.start()
        ev_1.set()
        with self.assertRaisesRegex(RuntimeError, 'First'):
            coord.join(threads)

    def testClearStopClearsExceptionToo(self):
        if False:
            return 10
        coord = coordinator.Coordinator()
        ev_1 = threading.Event()
        threads = [threading.Thread(target=RaiseOnEvent, args=(coord, ev_1, None, RuntimeError('First'), True))]
        for t in threads:
            t.start()
        with self.assertRaisesRegex(RuntimeError, 'First'):
            ev_1.set()
            coord.join(threads)
        coord.clear_stop()
        threads = [threading.Thread(target=RaiseOnEvent, args=(coord, ev_1, None, RuntimeError('Second'), True))]
        for t in threads:
            t.start()
        with self.assertRaisesRegex(RuntimeError, 'Second'):
            ev_1.set()
            coord.join(threads)

    def testRequestStopRaisesIfJoined(self):
        if False:
            i = 10
            return i + 15
        coord = coordinator.Coordinator()
        coord.join([])
        reported = False
        with self.assertRaisesRegex(RuntimeError, 'Too late'):
            try:
                raise RuntimeError('Too late')
            except RuntimeError as e:
                reported = True
                coord.request_stop(e)
        self.assertTrue(reported)
        coord.clear_stop()
        try:
            raise RuntimeError('After clear')
        except RuntimeError as e:
            coord.request_stop(e)
        with self.assertRaisesRegex(RuntimeError, 'After clear'):
            coord.join([])

    def testRequestStopRaisesIfJoined_ExcInfo(self):
        if False:
            print('Hello World!')
        coord = coordinator.Coordinator()
        coord.join([])
        reported = False
        with self.assertRaisesRegex(RuntimeError, 'Too late'):
            try:
                raise RuntimeError('Too late')
            except RuntimeError:
                reported = True
                coord.request_stop(sys.exc_info())
        self.assertTrue(reported)
        coord.clear_stop()
        try:
            raise RuntimeError('After clear')
        except RuntimeError:
            coord.request_stop(sys.exc_info())
        with self.assertRaisesRegex(RuntimeError, 'After clear'):
            coord.join([])

def _StopAt0(coord, n):
    if False:
        i = 10
        return i + 15
    if n[0] == 0:
        coord.request_stop()
    else:
        n[0] -= 1

class LooperTest(test.TestCase):

    def testTargetArgs(self):
        if False:
            i = 10
            return i + 15
        n = [3]
        coord = coordinator.Coordinator()
        thread = coordinator.LooperThread.loop(coord, 0, target=_StopAt0, args=(coord, n))
        coord.join([thread])
        self.assertEqual(0, n[0])

    def testTargetKwargs(self):
        if False:
            for i in range(10):
                print('nop')
        n = [3]
        coord = coordinator.Coordinator()
        thread = coordinator.LooperThread.loop(coord, 0, target=_StopAt0, kwargs={'coord': coord, 'n': n})
        coord.join([thread])
        self.assertEqual(0, n[0])

    def testTargetMixedArgs(self):
        if False:
            print('Hello World!')
        n = [3]
        coord = coordinator.Coordinator()
        thread = coordinator.LooperThread.loop(coord, 0, target=_StopAt0, args=(coord,), kwargs={'n': n})
        coord.join([thread])
        self.assertEqual(0, n[0])
if __name__ == '__main__':
    test.main()