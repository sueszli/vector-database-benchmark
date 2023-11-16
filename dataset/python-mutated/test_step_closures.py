from threading import Event
from time import sleep
import torch._lazy
import torch._lazy.ts_backend
from torch.testing._internal.common_utils import run_tests, TestCase
torch._lazy.ts_backend.init()

class ClosuresTest(TestCase):

    def test_synchronous(self):
        if False:
            return 10
        flag = Event()
        assert not flag.is_set()

        def closure():
            if False:
                while True:
                    i = 10
            sleep(1)
            assert not flag.is_set()
            flag.set()
        torch._lazy.add_step_closure(closure)
        torch._lazy.mark_step()
        assert flag.is_set()

    def test_asynchronous(self):
        if False:
            while True:
                i = 10
        flag = Event()
        assert not flag.is_set()

        def closure():
            if False:
                return 10
            sleep(1)
            assert flag.is_set()
        torch._lazy.add_step_closure(closure, run_async=True)
        torch._lazy.mark_step()
        assert not flag.is_set()
        flag.set()

    def test_synchronous_exception(self):
        if False:
            print('Hello World!')
        flag = Event()
        assert not flag.is_set()
        try:

            def closure():
                if False:
                    return 10
                flag.set()
                raise RuntimeError('Simulating exception in closure')
            torch._lazy.add_step_closure(closure)
            torch._lazy.mark_step()
            raise AssertionError()
        except RuntimeError as e:
            assert flag.is_set(), 'Should have caught exception from closure'

    def test_asynchronous_exception(self):
        if False:
            while True:
                i = 10
        flag = Event()
        assert not flag.is_set()

        def closure1():
            if False:
                print('Hello World!')
            flag.set()
            raise RuntimeError('Simulating exception in closure1')
        torch._lazy.add_step_closure(closure1, run_async=True)
        torch._lazy.mark_step()
        flag.wait(timeout=5)
        try:

            def closure2():
                if False:
                    return 10
                flag.clear()
            torch._lazy.add_step_closure(closure2, run_async=True)
            torch._lazy.mark_step()
            raise AssertionError()
        except RuntimeError as e:
            pass
        assert flag.is_set()
if __name__ == '__main__':
    run_tests()