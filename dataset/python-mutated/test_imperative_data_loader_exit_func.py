import multiprocessing
import queue
import signal
import time
import unittest
from paddle.base.reader import CleanupFuncRegistrar, _cleanup, multiprocess_queue_set

class TestDygraphDataLoaderCleanUpFunc(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.capacity = 10

    def test_clear_queue_set(self):
        if False:
            while True:
                i = 10
        test_queue = queue.Queue(self.capacity)
        multiprocess_queue_set.add(test_queue)
        for i in range(0, self.capacity):
            test_queue.put(i)
        _cleanup()

class TestRegisterExitFunc(unittest.TestCase):

    def none_func(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_not_callable_func(self):
        if False:
            for i in range(10):
                print('nop')
        exception = None
        try:
            CleanupFuncRegistrar.register(5)
        except TypeError as ex:
            self.assertIn('is not callable', str(ex))
            exception = ex
        self.assertIsNotNone(exception)

    def test_old_handler_for_sigint(self):
        if False:
            for i in range(10):
                print('nop')
        CleanupFuncRegistrar.register(function=self.none_func, signals=[signal.SIGINT])

    def test_signal_wrapper_by_sigchld(self):
        if False:
            return 10

        def __test_process__():
            if False:
                return 10
            pass
        CleanupFuncRegistrar.register(function=self.none_func, signals=[signal.SIGCHLD])
        exception = None
        try:
            test_process = multiprocessing.Process(target=__test_process__)
            test_process.start()
            time.sleep(3)
        except SystemExit as ex:
            exception = ex
        self.assertIsNotNone(exception)
if __name__ == '__main__':
    unittest.main()