from typing import Iterable
from unittest import TestCase
from unittest.mock import patch
from golem.vm.memorychecker import MemoryChecker, MemoryCheckerThread

class TestMemoryChecker(TestCase):

    @patch('golem.vm.memorychecker.MemoryCheckerThread')
    def test_memory_checker(self, mc_thread):
        if False:
            print('Hello World!')
        with MemoryChecker() as memory_checker:
            mc_thread().start.assert_called_once()
            self.assertEqual(memory_checker.estm_mem, mc_thread().estm_mem)
            mc_thread.stop.assert_not_called()
        mc_thread().stop.assert_called_once()

class TestMemoryCheckerThread(TestCase):

    @patch('golem.vm.memorychecker.psutil.virtual_memory')
    def test_not_started(self, virtual_memory):
        if False:
            return 10
        virtual_memory().used = 2137
        mc_thread = MemoryCheckerThread()
        self.assertEqual(mc_thread.estm_mem, 0)

    @patch('golem.vm.memorychecker.time.sleep')
    @patch('golem.vm.memorychecker.psutil.virtual_memory')
    def _generic_test(self, virtual_memory, sleep, initial_mem_usage: int, mem_usage: Iterable[int], exp_estimation: Iterable[int]) -> None:
        if False:
            i = 10
            return i + 15
        virtual_memory().used = initial_mem_usage
        mc_thread = MemoryCheckerThread()

        def _advance():
            if False:
                while True:
                    i = 10
            for (used, expected) in zip(mem_usage, exp_estimation):
                virtual_memory().used = used
                yield
                self.assertEqual(mc_thread.estm_mem, expected)
            mc_thread.stop()
            yield
        advance = _advance()
        sleep.side_effect = lambda _: next(advance)
        mc_thread.run()

    def test_memory_usage_constant(self):
        if False:
            print('Hello World!')
        self._generic_test(initial_mem_usage=1000, mem_usage=(1000, 1000, 1000), exp_estimation=(0, 0, 0))

    def test_memory_usage_rising(self):
        if False:
            return 10
        self._generic_test(initial_mem_usage=1000, mem_usage=(2000, 3000, 4000), exp_estimation=(1000, 2000, 3000))

    def test_memory_usage_sinking(self):
        if False:
            print('Hello World!')
        self._generic_test(initial_mem_usage=4000, mem_usage=(4000, 3000, 2000), exp_estimation=(0, 1000, 2000))

    def test_memory_usage_rising_then_sinking(self):
        if False:
            return 10
        self._generic_test(initial_mem_usage=2000, mem_usage=(2000, 3000, 2000, 1000), exp_estimation=(0, 1000, 1000, 1000))

    def test_memory_usage_sinking_then_rising(self):
        if False:
            for i in range(10):
                print('nop')
        self._generic_test(initial_mem_usage=3000, mem_usage=(2000, 3000, 4000, 5000), exp_estimation=(1000, 1000, 1000, 2000))