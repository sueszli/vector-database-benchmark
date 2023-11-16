"""PyUnit testing that threads honor our signal semantics"""
import unittest
import signal
import os
import sys
from test.support import threading_helper
import _thread as thread
import time
if sys.platform[:3] == 'win':
    raise unittest.SkipTest("Can't test signal on %s" % sys.platform)
process_pid = os.getpid()
signalled_all = thread.allocate_lock()
USING_PTHREAD_COND = sys.thread_info.name == 'pthread' and sys.thread_info.lock == 'mutex+cond'

def registerSignals(for_usr1, for_usr2, for_alrm):
    if False:
        i = 10
        return i + 15
    usr1 = signal.signal(signal.SIGUSR1, for_usr1)
    usr2 = signal.signal(signal.SIGUSR2, for_usr2)
    alrm = signal.signal(signal.SIGALRM, for_alrm)
    return (usr1, usr2, alrm)

def handle_signals(sig, frame):
    if False:
        print('Hello World!')
    signal_blackboard[sig]['tripped'] += 1
    signal_blackboard[sig]['tripped_by'] = thread.get_ident()

def send_signals():
    if False:
        return 10
    os.kill(process_pid, signal.SIGUSR1)
    os.kill(process_pid, signal.SIGUSR2)
    signalled_all.release()

class ThreadSignals(unittest.TestCase):

    def test_signals(self):
        if False:
            return 10
        with threading_helper.wait_threads_exit():
            signalled_all.acquire()
            self.spawnSignallingThread()
            signalled_all.acquire()
        if signal_blackboard[signal.SIGUSR1]['tripped'] == 0 or signal_blackboard[signal.SIGUSR2]['tripped'] == 0:
            try:
                signal.alarm(1)
                signal.pause()
            finally:
                signal.alarm(0)
        self.assertEqual(signal_blackboard[signal.SIGUSR1]['tripped'], 1)
        self.assertEqual(signal_blackboard[signal.SIGUSR1]['tripped_by'], thread.get_ident())
        self.assertEqual(signal_blackboard[signal.SIGUSR2]['tripped'], 1)
        self.assertEqual(signal_blackboard[signal.SIGUSR2]['tripped_by'], thread.get_ident())
        signalled_all.release()

    def spawnSignallingThread(self):
        if False:
            while True:
                i = 10
        thread.start_new_thread(send_signals, ())

    def alarm_interrupt(self, sig, frame):
        if False:
            i = 10
            return i + 15
        raise KeyboardInterrupt

    @unittest.skipIf(USING_PTHREAD_COND, 'POSIX condition variables cannot be interrupted')
    @unittest.skipIf(sys.platform.startswith('linux') and (not sys.thread_info.version), 'Issue 34004: musl does not allow interruption of locks by signals.')
    @unittest.skipIf(sys.platform.startswith('openbsd'), 'lock cannot be interrupted on OpenBSD')
    def test_lock_acquire_interruption(self):
        if False:
            i = 10
            return i + 15
        oldalrm = signal.signal(signal.SIGALRM, self.alarm_interrupt)
        try:
            lock = thread.allocate_lock()
            lock.acquire()
            signal.alarm(1)
            t1 = time.monotonic()
            self.assertRaises(KeyboardInterrupt, lock.acquire, timeout=5)
            dt = time.monotonic() - t1
            self.assertLess(dt, 3.0)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, oldalrm)

    @unittest.skipIf(USING_PTHREAD_COND, 'POSIX condition variables cannot be interrupted')
    @unittest.skipIf(sys.platform.startswith('linux') and (not sys.thread_info.version), 'Issue 34004: musl does not allow interruption of locks by signals.')
    @unittest.skipIf(sys.platform.startswith('openbsd'), 'lock cannot be interrupted on OpenBSD')
    def test_rlock_acquire_interruption(self):
        if False:
            return 10
        oldalrm = signal.signal(signal.SIGALRM, self.alarm_interrupt)
        try:
            rlock = thread.RLock()

            def other_thread():
                if False:
                    return 10
                rlock.acquire()
            with threading_helper.wait_threads_exit():
                thread.start_new_thread(other_thread, ())
                while rlock.acquire(blocking=False):
                    rlock.release()
                    time.sleep(0.01)
                signal.alarm(1)
                t1 = time.monotonic()
                self.assertRaises(KeyboardInterrupt, rlock.acquire, timeout=5)
                dt = time.monotonic() - t1
                self.assertLess(dt, 3.0)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, oldalrm)

    def acquire_retries_on_intr(self, lock):
        if False:
            i = 10
            return i + 15
        self.sig_recvd = False

        def my_handler(signal, frame):
            if False:
                while True:
                    i = 10
            self.sig_recvd = True
        old_handler = signal.signal(signal.SIGUSR1, my_handler)
        try:

            def other_thread():
                if False:
                    return 10
                lock.acquire()
                time.sleep(0.5)
                os.kill(process_pid, signal.SIGUSR1)
                time.sleep(0.5)
                lock.release()
            with threading_helper.wait_threads_exit():
                thread.start_new_thread(other_thread, ())
                while lock.acquire(blocking=False):
                    lock.release()
                    time.sleep(0.01)
                result = lock.acquire()
                self.assertTrue(self.sig_recvd)
                self.assertTrue(result)
        finally:
            signal.signal(signal.SIGUSR1, old_handler)

    def test_lock_acquire_retries_on_intr(self):
        if False:
            while True:
                i = 10
        self.acquire_retries_on_intr(thread.allocate_lock())

    def test_rlock_acquire_retries_on_intr(self):
        if False:
            for i in range(10):
                print('nop')
        self.acquire_retries_on_intr(thread.RLock())

    def test_interrupted_timed_acquire(self):
        if False:
            print('Hello World!')
        self.start = None
        self.end = None
        self.sigs_recvd = 0
        done = thread.allocate_lock()
        done.acquire()
        lock = thread.allocate_lock()
        lock.acquire()

        def my_handler(signum, frame):
            if False:
                for i in range(10):
                    print('nop')
            self.sigs_recvd += 1
        old_handler = signal.signal(signal.SIGUSR1, my_handler)
        try:

            def timed_acquire():
                if False:
                    while True:
                        i = 10
                self.start = time.monotonic()
                lock.acquire(timeout=0.5)
                self.end = time.monotonic()

            def send_signals():
                if False:
                    print('Hello World!')
                for _ in range(40):
                    time.sleep(0.02)
                    os.kill(process_pid, signal.SIGUSR1)
                done.release()
            with threading_helper.wait_threads_exit():
                thread.start_new_thread(send_signals, ())
                timed_acquire()
                done.acquire()
                self.assertLess(self.end - self.start, 2.0)
                self.assertGreater(self.end - self.start, 0.3)
                self.assertGreater(self.sigs_recvd, 0)
        finally:
            signal.signal(signal.SIGUSR1, old_handler)

def setUpModule():
    if False:
        print('Hello World!')
    global signal_blackboard
    signal_blackboard = {signal.SIGUSR1: {'tripped': 0, 'tripped_by': 0}, signal.SIGUSR2: {'tripped': 0, 'tripped_by': 0}, signal.SIGALRM: {'tripped': 0, 'tripped_by': 0}}
    oldsigs = registerSignals(handle_signals, handle_signals, handle_signals)
    unittest.addModuleCleanup(registerSignals, *oldsigs)
if __name__ == '__main__':
    unittest.main()