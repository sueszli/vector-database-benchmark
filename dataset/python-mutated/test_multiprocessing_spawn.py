import os
import pickle
import random
import signal
import sys
import time
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, IS_WINDOWS, NO_MULTIPROCESSING_SPAWN
import torch.multiprocessing as mp

def _test_success_func(i):
    if False:
        i = 10
        return i + 15
    pass

def _test_success_single_arg_func(i, arg):
    if False:
        return 10
    if arg:
        arg.put(i)

def _test_exception_single_func(i, arg):
    if False:
        return 10
    if i == arg:
        raise ValueError('legitimate exception from process %d' % i)
    time.sleep(1.0)

def _test_exception_all_func(i):
    if False:
        print('Hello World!')
    time.sleep(random.random() / 10)
    raise ValueError('legitimate exception from process %d' % i)

def _test_terminate_signal_func(i):
    if False:
        return 10
    if i == 0:
        os.kill(os.getpid(), signal.SIGABRT)
    time.sleep(1.0)

def _test_terminate_exit_func(i, arg):
    if False:
        while True:
            i = 10
    if i == 0:
        sys.exit(arg)
    time.sleep(1.0)

def _test_success_first_then_exception_func(i, arg):
    if False:
        print('Hello World!')
    if i == 0:
        return
    time.sleep(0.1)
    raise ValueError('legitimate exception')

def _test_nested_child_body(i, ready_queue, nested_child_sleep):
    if False:
        return 10
    ready_queue.put(None)
    time.sleep(nested_child_sleep)

def _test_infinite_task(i):
    if False:
        while True:
            i = 10
    while True:
        time.sleep(1)

def _test_process_exit(idx):
    if False:
        i = 10
        return i + 15
    sys.exit(12)

def _test_nested(i, pids_queue, nested_child_sleep, start_method):
    if False:
        i = 10
        return i + 15
    context = mp.get_context(start_method)
    nested_child_ready_queue = context.Queue()
    nprocs = 2
    mp_context = mp.start_processes(fn=_test_nested_child_body, args=(nested_child_ready_queue, nested_child_sleep), nprocs=nprocs, join=False, daemon=False, start_method=start_method)
    pids_queue.put(mp_context.pids())
    for _ in range(nprocs):
        nested_child_ready_queue.get()
    os.kill(os.getpid(), signal.SIGTERM)

class _TestMultiProcessing:
    start_method = None

    def test_success(self):
        if False:
            print('Hello World!')
        mp.start_processes(_test_success_func, nprocs=2, start_method=self.start_method)

    def test_success_non_blocking(self):
        if False:
            while True:
                i = 10
        mp_context = mp.start_processes(_test_success_func, nprocs=2, join=False, start_method=self.start_method)
        mp_context.join(timeout=None)
        mp_context.join(timeout=None)
        self.assertTrue(mp_context.join(timeout=None))

    def test_first_argument_index(self):
        if False:
            for i in range(10):
                print('nop')
        context = mp.get_context(self.start_method)
        queue = context.SimpleQueue()
        mp.start_processes(_test_success_single_arg_func, args=(queue,), nprocs=2, start_method=self.start_method)
        self.assertEqual([0, 1], sorted([queue.get(), queue.get()]))

    def test_exception_single(self):
        if False:
            i = 10
            return i + 15
        nprocs = 2
        for i in range(nprocs):
            with self.assertRaisesRegex(Exception, '\nValueError: legitimate exception from process %d$' % i):
                mp.start_processes(_test_exception_single_func, args=(i,), nprocs=nprocs, start_method=self.start_method)

    def test_exception_all(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(Exception, '\nValueError: legitimate exception from process (0|1)$'):
            mp.start_processes(_test_exception_all_func, nprocs=2, start_method=self.start_method)

    def test_terminate_signal(self):
        if False:
            return 10
        message = 'process 0 terminated with signal (SIGABRT|SIGIOT)'
        if IS_WINDOWS:
            message = 'process 0 terminated with exit code 22'
        with self.assertRaisesRegex(Exception, message):
            mp.start_processes(_test_terminate_signal_func, nprocs=2, start_method=self.start_method)

    def test_terminate_exit(self):
        if False:
            for i in range(10):
                print('nop')
        exitcode = 123
        with self.assertRaisesRegex(Exception, 'process 0 terminated with exit code %d' % exitcode):
            mp.start_processes(_test_terminate_exit_func, args=(exitcode,), nprocs=2, start_method=self.start_method)

    def test_success_first_then_exception(self):
        if False:
            return 10
        exitcode = 123
        with self.assertRaisesRegex(Exception, 'ValueError: legitimate exception'):
            mp.start_processes(_test_success_first_then_exception_func, args=(exitcode,), nprocs=2, start_method=self.start_method)

    @unittest.skipIf(sys.platform != 'linux', 'Only runs on Linux; requires prctl(2)')
    def _test_nested(self):
        if False:
            return 10
        context = mp.get_context(self.start_method)
        pids_queue = context.Queue()
        nested_child_sleep = 20.0
        mp_context = mp.start_processes(fn=_test_nested, args=(pids_queue, nested_child_sleep, self.start_method), nprocs=1, join=False, daemon=False, start_method=self.start_method)
        pids = pids_queue.get()
        start = time.time()
        while len(pids) > 0:
            for pid in pids:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pids.remove(pid)
                    break
            self.assertLess(time.time() - start, nested_child_sleep / 2)
            time.sleep(0.1)

@unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that don't support the spawn start method")
class SpawnTest(TestCase, _TestMultiProcessing):
    start_method = 'spawn'

    def test_exception_raises(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(mp.ProcessRaisedException):
            mp.spawn(_test_success_first_then_exception_func, args=(), nprocs=1)

    def test_signal_raises(self):
        if False:
            print('Hello World!')
        context = mp.spawn(_test_infinite_task, args=(), nprocs=1, join=False)
        for pid in context.pids():
            os.kill(pid, signal.SIGTERM)
        with self.assertRaises(mp.ProcessExitedException):
            context.join()

    def _test_process_exited(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(mp.ProcessExitedException) as e:
            mp.spawn(_test_process_exit, args=(), nprocs=1)
            self.assertEqual(12, e.exit_code)

@unittest.skipIf(IS_WINDOWS, 'Fork is only available on Unix')
class ForkTest(TestCase, _TestMultiProcessing):
    start_method = 'fork'

class ErrorTest(TestCase):

    def test_errors_pickleable(self):
        if False:
            i = 10
            return i + 15
        for error in (mp.ProcessRaisedException('Oh no!', 1, 1), mp.ProcessExitedException('Oh no!', 1, 1, 1)):
            pickle.loads(pickle.dumps(error))
if __name__ == '__main__':
    run_tests()