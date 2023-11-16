import multiprocessing
import os
import signal
import sys
import time
import unittest
from paddle.base import core

def set_child_signal_handler(self, child_pid):
    if False:
        return 10
    core._set_process_pids(id(self), (child_pid,))
    current_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(current_handler):
        current_handler = None

    def __handler__(signum, frame):
        if False:
            return 10
        core._throw_error_if_process_failed()
        if current_handler is not None:
            current_handler(signum, frame)
    signal.signal(signal.SIGCHLD, __handler__)

class DygraphDataLoaderSingalHandler(unittest.TestCase):

    def test_child_process_exit_with_error(self):
        if False:
            print('Hello World!')

        def __test_process__():
            if False:
                return 10
            core._set_process_signal_handler()
            sys.exit(1)

        def try_except_exit():
            if False:
                while True:
                    i = 10
            exception = None
            try:
                test_process = multiprocessing.Process(target=__test_process__)
                test_process.start()
                set_child_signal_handler(id(self), test_process.pid)
                time.sleep(5)
            except SystemError as ex:
                self.assertIn('Fatal', str(ex))
                exception = ex
            return exception
        try_time = 10
        exception = None
        for i in range(try_time):
            exception = try_except_exit()
            if exception is not None:
                break
        self.assertIsNotNone(exception)

    def test_child_process_killed_by_sigsegv(self):
        if False:
            return 10

        def __test_process__():
            if False:
                for i in range(10):
                    print('nop')
            core._set_process_signal_handler()
            os.kill(os.getpid(), signal.SIGSEGV)

        def try_except_exit():
            if False:
                return 10
            exception = None
            try:
                test_process = multiprocessing.Process(target=__test_process__)
                test_process.start()
                set_child_signal_handler(id(self), test_process.pid)
                time.sleep(5)
            except SystemError as ex:
                self.assertIn('Segmentation fault', str(ex))
                exception = ex
            return exception
        try_time = 10
        exception = None
        for i in range(try_time):
            exception = try_except_exit()
            if exception is not None:
                break
        self.assertIsNotNone(exception)

    def test_child_process_killed_by_sigbus(self):
        if False:
            i = 10
            return i + 15

        def __test_process__():
            if False:
                while True:
                    i = 10
            core._set_process_signal_handler()
            os.kill(os.getpid(), signal.SIGBUS)

        def try_except_exit():
            if False:
                print('Hello World!')
            exception = None
            try:
                test_process = multiprocessing.Process(target=__test_process__)
                test_process.start()
                set_child_signal_handler(id(self), test_process.pid)
                time.sleep(5)
            except SystemError as ex:
                self.assertIn('Bus error', str(ex))
                exception = ex
            return exception
        try_time = 10
        exception = None
        for i in range(try_time):
            exception = try_except_exit()
            if exception is not None:
                break
        self.assertIsNotNone(exception)

    def test_child_process_killed_by_sigterm(self):
        if False:
            for i in range(10):
                print('nop')

        def __test_process__():
            if False:
                return 10
            core._set_process_signal_handler()
            time.sleep(10)
        test_process = multiprocessing.Process(target=__test_process__)
        test_process.daemon = True
        test_process.start()
        set_child_signal_handler(id(self), test_process.pid)
        time.sleep(1)
if __name__ == '__main__':
    unittest.main()