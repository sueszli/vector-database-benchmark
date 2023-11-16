import logging
import re
import shutil
import unittest
import torch
import torch._dynamo.test_case
import torch._dynamo.testing
try:
    import dill
except ImportError:
    dill = None
requires_dill = unittest.skipIf(dill is None, 'requires dill')

class ReplayRecordTests(torch._dynamo.test_case.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super().setUpClass()
        cls._exit_stack.enter_context(unittest.mock.patch.object(torch._dynamo.config, 'replay_record_enabled', True))
        torch._logging.set_logs(graph_breaks=True, dynamo=logging.ERROR)
        cls._exit_stack.enter_context(unittest.mock.patch.object(torch._dynamo.config, 'suppress_errors', True))
        cls._exit_stack.enter_context(unittest.mock.patch.object(torch._dynamo.config, 'debug_dir_root', '/tmp/_torchdynamo_debug_/'))

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree(torch._dynamo.config.debug_dir_root, ignore_errors=True)
        torch._logging.set_logs()
        cls._exit_stack.close()

    def check_replay(self, fn, *args, exp_exc_name=None):
        if False:
            return 10
        fn_opt = torch._dynamo.optimize('eager')(fn)
        with self.assertLogs(logger='torch._dynamo', level=logging.ERROR) as log_orig:
            try:
                fn_opt(*args)
            except Exception:
                pass
        with self.assertLogs(logger='torch._dynamo', level=logging.ERROR) as log_replayed:
            file_name_match = re.search("torch._dynamo\\.replay\\('(.*)'\\)", log_orig.output[-1])
            self.assertTrue(file_name_match is not None, 'No record file name found in generated logs.')
            torch._dynamo.replay(file_name_match.groups()[0])

        def get_error_name(log):
            if False:
                for i in range(10):
                    print('nop')
            error_name = re.search('\\w+Error', log.output[-1])
            self.assertIsNotNone(error_name, 'No error name found in logs.')
            return error_name[0]
        orig_error = get_error_name(log_orig)
        replayed_error = get_error_name(log_replayed)
        if exp_exc_name is not None:
            self.assertEqual(orig_error, exp_exc_name)
        self.assertEqual(orig_error, replayed_error, 'Error logs for recorded execution and replayed execution should match.')

    @requires_dill
    def test_unsuccessful_inline(self):
        if False:
            i = 10
            return i + 15

        def level2():
            if False:
                for i in range(10):
                    print('nop')
            z = torch.ones(2, 2)
            a = {z: 10}
            return a[z] * torch.ones(1)

        def level1():
            if False:
                for i in range(10):
                    print('nop')
            y = torch.ones(1, 1)
            return level2() + y

        def level0():
            if False:
                print('Hello World!')
            x = torch.ones(1, 1)
            return level1() + x
        self.check_replay(level0, exp_exc_name='AssertionError')

    @requires_dill
    def test_successful_inline(self):
        if False:
            print('Hello World!')

        def test_fn():
            if False:
                i = 10
                return i + 15
            x = torch.ones(2, 2)

            def level1(a):
                if False:
                    return 10
                return a + torch.ones(2, 2)
            y = level1(x)
            return y + torch.ones(3, 3)
        self.check_replay(test_fn, exp_exc_name='RuntimeError')

    @requires_dill
    def test_nonlocal_fn_call(self):
        if False:
            return 10

        def nonlocal_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + torch.ones(2, 2)

        def test_fn():
            if False:
                while True:
                    i = 10
            z = torch.ones(2, 2)
            x = nonlocal_fn(z)
            return x + torch.ones(3, 3)
        self.check_replay(test_fn, exp_exc_name='RuntimeError')

    @requires_dill
    def test_nonlocal_module_fn_call(self):
        if False:
            while True:
                i = 10
        try:
            from . import mock_modules
        except ImportError:
            import mock_modules

        def test_fn():
            if False:
                for i in range(10):
                    print('nop')
            z = mock_modules.mock_module2.method1([], 2)
            z = torch.ones(2, 2) + z[0]
            return z + torch.zeros(3, 3)
        self.check_replay(test_fn, exp_exc_name='RuntimeError')

    @requires_dill
    def test_nonlocal_module_class(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            from .mock_modules import mock_module2
        except ImportError:
            from mock_modules import mock_module2

        def test_fn():
            if False:
                while True:
                    i = 10
            z = mock_module2.Class1(1, 2)
            y = z.method2(torch.ones(3, 3))
            return y + torch.zeros(3, 5)
        self.check_replay(test_fn, exp_exc_name='TypeError')

    @requires_dill
    def test_local_module(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            from .mock_modules import mock_module3 as _

            def test_fn(x):
                if False:
                    print('Hello World!')
                from .mock_modules import mock_module3
                z = mock_module3.method1([], torch.ones(5, 1))
                return torch.ones(2, 2) + x + z[0]
        except ImportError:

            def test_fn(x):
                if False:
                    while True:
                        i = 10
                from mock_modules import mock_module3
                z = mock_module3.method1([], torch.ones(5, 1))
                return torch.ones(2, 2) + x + z[0]
        self.check_replay(test_fn, torch.ones(1, 1), exp_exc_name='RuntimeError')

    @requires_dill
    def test_fn_call_args(self):
        if False:
            return 10

        def test_fn(x, y):
            if False:
                print('Hello World!')
            return x + y + torch.zeros(2, 2)
        self.check_replay(test_fn, torch.ones(3, 3), torch.ones(2, 2), exp_exc_name='RuntimeError')
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    run_tests()