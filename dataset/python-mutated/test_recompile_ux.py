import unittest
import weakref
import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
import torch._dynamo.testing

class RecompileUxTests(torch._dynamo.test_case.TestCase):
    cache_limit = 1

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super().setUpClass()
        cls._exit_stack.enter_context(torch._dynamo.config.patch('cache_size_limit', cls.cache_limit))

    def test_drop_cache_on_skip(self):
        if False:
            print('Hello World!')

        def model(x, i):
            if False:
                for i in range(10):
                    print('nop')
            return x + i
        attached = False
        triggered = False

        def trigger():
            if False:
                for i in range(10):
                    print('nop')
            nonlocal triggered
            triggered = True

        def compiler(gm, input):
            if False:
                print('Hello World!')
            nonlocal attached
            f = gm.forward
            assert not attached
            weakref.finalize(f, trigger)
            attached = True
            return f
        x = torch.randn(2)
        for i in range(2):
            opt_model = torch._dynamo.optimize(compiler)(model)
            opt_model(x, i)
        self.assertTrue(triggered)

    def test_loop_torture(self):
        if False:
            for i in range(10):
                print('nop')

        def loop_torture(input, iters):
            if False:
                i = 10
                return i + 15
            out = input
            for _ in range(iters):
                out += input
            return out
        compile_counter = torch._dynamo.testing.CompileCounter()
        for _ in range(10):
            x = torch.randn(3)
            iters = torch.randint(low=0, high=1000, size=())
            opt_loop_torture = torch._dynamo.optimize(compile_counter)(loop_torture)
            opt_loop_torture(x, iters)
        self.assertEqual(compile_counter.frame_count, self.cache_limit)

    @torch._dynamo.config.patch('automatic_dynamic_shapes', False)
    def test_dynamic_input(self):
        if False:
            while True:
                i = 10

        def model(input):
            if False:
                print('Hello World!')
            return input + input
        expected_recompiles = 2
        compile_counter = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch('cache_size_limit', expected_recompiles):
            with self.assertLogs(logger='torch._dynamo', level='WARNING') as logs:
                for _ in range(10):
                    bsz = torch.randint(low=0, high=1000, size=())
                    x = torch.randn((bsz, 3, 4))
                    opt_model = torch._dynamo.optimize(compile_counter)(model)
                    opt_model(x)
        self.assertEqual(compile_counter.frame_count, expected_recompiles)
        self.assertEqual(len(logs.records), 1)
        print(logs.records[0])
        self.assertTrue(logs.records[0].getMessage().startswith('torch._dynamo hit config.cache_size_limit'))

    @unittest.skipIf(not torch.cuda.is_available(), 'requires cuda')
    def test_nvfuser_guards(self):
        if False:
            for i in range(10):
                print('nop')

        def func(a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            return a + b * c
        a = torch.rand(3, 4, 5, device='cuda')
        b = torch.rand(3, 4, 5, device='cuda')
        b_v = torch.rand(3, 5, 4, device='cuda').view(3, 4, 5)
        b_p = torch.rand(3, 5, 4, device='cuda').permute(0, 2, 1)
        c = torch.rand(3, 4, 5, device='cuda')
        compile_counter = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch('cache_size_limit', 2):
            opt_func = torch._dynamo.optimize(compile_counter)(func)
            opt_func(a, b, c)
            self.assertEqual(compile_counter.frame_count, 1)
            opt_func(a, b, c)
            self.assertEqual(compile_counter.frame_count, 1)
            opt_func(a, b_v, c)
            self.assertEqual(compile_counter.frame_count, 1)
            opt_func(a, b_p, c)
            self.assertEqual(compile_counter.frame_count, 2)

    def assert_single_log_contains(self, logs, contains_str):
        if False:
            while True:
                i = 10
        self.assertEqual(len(logs.records), 1)
        self.assertTrue(logs.records[0].getMessage().find(contains_str) > 0, msg=f'Expected to find "{contains_str}" in log "{logs.records[0].getMessage()}"')

    def test_verbose_tensor_check(self):
        if False:
            for i in range(10):
                print('nop')

        def func(a):
            if False:
                while True:
                    i = 10
            return torch.add(a, 4)

        def cache_fail_test(cached_input, missed_input, expected_failure):
            if False:
                for i in range(10):
                    print('nop')
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            opt_func = torch._dynamo.optimize('eager')(func)
            opt_func(cached_input)
            with self.assertLogs(logger='torch._dynamo', level='WARNING') as logs:
                opt_func = torch._dynamo.optimize('eager')(func)
                opt_func(missed_input)
            self.assert_single_log_contains(logs, expected_failure)
        a = torch.rand(3, 4, 5)
        cache_fail_test(a, a[0:2, :, :], "tensor 'L['a']' size mismatch at index 0. expected 3, actual 2")
        cache_fail_test(a, a.clone().as_strided((3, 4, 5), stride=(1, 3, 12)), "tensor 'L['a']' stride mismatch at index 0. expected 20, actual 1")
        cache_fail_test(a, a[0, :, :], "tensor 'L['a']' rank mismatch. expected 3, actual 2")
        cache_fail_test(a, a.to('meta'), "tensor 'L['a']' dispatch key set mismatch.")
        cache_fail_test(a, a.to(torch.float16), "tensor 'L['a']' dtype mismatch. expected Float, actual Half")
        a_grad = a.clone()
        a_grad.requires_grad = True
        cache_fail_test(a, a_grad, "tensor 'L['a']' requires_grad mismatch. expected requires_grad=0")

    def test_mismatched_type(self):
        if False:
            for i in range(10):
                print('nop')
        a = torch.rand(3, 4, 5)
        b = torch.rand(3, 4, 5)

        def func(a, b):
            if False:
                print('Hello World!')
            return a + b
        opt_func = torch._dynamo.optimize('eager')(func)
        opt_func(a, b)
        with self.assertLogs(logger='torch._dynamo', level='WARNING') as logs:
            opt_func = torch._dynamo.optimize('eager')(func)
            opt_func(a, 1)
        self.assert_single_log_contains(logs, "expected type of 'L['b']' to be a tensor type, ' but found <class 'int'>")

    @torch._dynamo.config.patch('cache_size_limit', 32)
    def test_multiple_guard_fails(self):
        if False:
            for i in range(10):
                print('nop')
        failure_reasons = []

        def guard_fail_fn(failure):
            if False:
                while True:
                    i = 10
            failure_reasons.append(failure[0])

        def f(x):
            if False:
                i = 10
                return i + 15
            return torch.relu(x)
        opt_f = torch._dynamo.optimize(backend='eager', guard_fail_fn=guard_fail_fn, dynamic=False)(f)
        for i in range(5):
            failure_reasons.clear()
            opt_f(torch.randn(8 + i))
        failure_str = '\n'.join(failure_reasons)
        self.assertExpectedInline(failure_str, "tensor 'L['x']' size mismatch at index 0. expected 11, actual 12\ntensor 'L['x']' size mismatch at index 0. expected 10, actual 12\ntensor 'L['x']' size mismatch at index 0. expected 9, actual 12\ntensor 'L['x']' size mismatch at index 0. expected 8, actual 12")