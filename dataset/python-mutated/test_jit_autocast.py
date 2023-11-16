import torch
from torch.cuda.amp import autocast
from typing import Optional, Tuple
import unittest
from test_jit import JitTestCase
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo
from torch.testing import FileCheck
from jit.test_models import MnistNet
TEST_BFLOAT16 = TEST_CUDA and torch.cuda.is_bf16_supported()

@skipIfTorchDynamo('Not a TorchDynamo suitable test')
class TestAutocast(JitTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        if TEST_CUDA:
            self.a_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
            self.b_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
            self.c_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
            self.d_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
            self.a_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
            self.b_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
            self.c_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
            self.d_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
        self.old_value = torch._C._jit_set_autocast_mode(True)
        super().setUp()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        torch._C._jit_set_autocast_mode(self.old_value)
        super().tearDown()

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_minimal(self):
        if False:
            return 10

        @torch.jit.script
        def fn(a, b):
            if False:
                return 10
            with autocast():
                x = torch.mm(a, b)
                y = torch.sum(x)
                return (x, y)
        (x, y) = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(x.dtype, torch.float16)
        self.assertEqual(y.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA or not TEST_BFLOAT16, 'No cuda bfloat16 support')
    def test_linear_bf16(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def fn(a, b):
            if False:
                print('Hello World!')
            with autocast(dtype=torch.bfloat16):
                x = torch.mm(a, b)
                y = torch.sum(x)
                return (x, y)
        (x, y) = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(x.dtype, torch.bfloat16)
        self.assertEqual(y.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_minimal_cpu(self):
        if False:
            return 10

        @torch.jit.script
        def fn(a, b):
            if False:
                print('Hello World!')
            with autocast():
                return torch.mm(a, b)
        result = fn(self.a_fp32.to('cpu'), self.b_fp32.to('cpu'))
        self.assertEqual(result.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_minimal_off(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def fn(a, b):
            if False:
                while True:
                    i = 10
            with autocast(enabled=False):
                return torch.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_runtime_autocast_state(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def fn(a, b, use_amp: bool):
            if False:
                for i in range(10):
                    print('nop')
            with autocast(enabled=use_amp):
                return torch.mm(a, b)
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32, True)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_runtime_autocast_state_expr(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def fn(a, b):
            if False:
                print('Hello World!')
            with autocast(enabled=True if a[0][0] > 0.5 else False):
                return torch.mm(a, b)
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_explicit_casts(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def fn(a, b, c, d):
            if False:
                print('Hello World!')
            with autocast():
                e = torch.mm(a.double(), b.double()).float()
                f = torch.mm(c, d).double()
            g = torch.mm(c.double(), f)
            return (e, f, g)
        (e, f, g) = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float32)
        self.assertEqual(f.dtype, torch.float64)
        self.assertEqual(g.dtype, torch.float64)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_duplicate_inputs(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def fn(a, b):
            if False:
                print('Hello World!')
            with autocast():
                e = torch.mm(a, a)
                f = torch.mm(e, e)
            return (e, f)
        (e, f) = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_fp32_policy(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def fn(a):
            if False:
                return 10
            with autocast(enabled=True):
                return torch.log(a)
        result = fn(self.a_fp16)
        self.assertEqual(result.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_fp32_policy_with_fp64(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def fn(a):
            if False:
                print('Hello World!')
            with autocast(enabled=True):
                return torch.log(a)
        result = fn(self.a_fp32.double())
        self.assertEqual(result.dtype, torch.float64)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_promote_policy(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def fn(a, b, c, d):
            if False:
                i = 10
                return i + 15
            with autocast():
                e = torch.mm(a, b)
                f = torch.addcmul(e, c, d, value=0.1)
            return (e, f)
        (e, f) = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_promote_policy_fp64(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def fn(a, b):
            if False:
                i = 10
                return i + 15
            with autocast(enabled=True):
                return torch.addcmul(a, a, b, value=0.1)
        result = fn(self.a_fp32.double(), self.b_fp32.double())
        self.assertEqual(result.dtype, torch.float64)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_fp32_set_opt_dtype_policy(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def fn(a, b, c, d, dtype: Optional[int]):
            if False:
                while True:
                    i = 10
            with autocast(enabled=True):
                x = torch.softmax(a, 0)
                y = torch.softmax(b, 0, None)
                z = torch.softmax(c, 0, torch.float64)
                w = torch.softmax(d, 0, dtype)
            return (x, y, z, w)
        (x, y, z, w) = fn(self.a_fp16, self.b_fp16, self.c_fp16, self.d_fp16, None)
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(z.dtype, torch.float64)
        self.assertEqual(w.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_fp32_set_opt_dtype_policy_fp64(self):
        if False:
            return 10

        @torch.jit.script
        def fn(a, b, c, d, dtype: Optional[int]):
            if False:
                print('Hello World!')
            with autocast(enabled=True):
                x = torch.softmax(a, 0)
                y = torch.softmax(b, 0, None)
                z = torch.softmax(c, 0, torch.float64)
                w = torch.softmax(d, 0, dtype)
            return (x, y, z, w)
        (x, y, z, w) = fn(self.a_fp32.double(), self.b_fp32.double(), self.c_fp32.double(), self.d_fp32.double(), None)
        self.assertEqual(x.dtype, torch.float64)
        self.assertEqual(y.dtype, torch.float64)
        self.assertEqual(z.dtype, torch.float64)
        self.assertEqual(w.dtype, torch.float64)

    @unittest.skipIf(True, 'broken due to lack of type propagation')
    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_control_flow(self):
        if False:
            return 10

        @torch.jit.script
        def fn(a, b, c, d):
            if False:
                for i in range(10):
                    print('nop')
            with autocast():
                if a[0][0] > 0.5:
                    e = torch.mm(a, b)
                    x = 1
                else:
                    e = torch.mm(c, d)
                    x = 2
                f = torch.mm(d, e) * x
            return (e, f)
        (e, f) = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_divergent_types(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def fn(a, b, c, d):
            if False:
                return 10
            with autocast():
                if a[0][0] > 0.5:
                    e = torch.mm(a, b)
                    f = torch.mm(a, b).float()
                else:
                    e = torch.mm(c, d).float()
                    f = torch.mm(a, b)
            return torch.mm(e.float(), f.float())
        result = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(result.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_divergent_autocast(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def fn(a, b, c, d):
            if False:
                return 10
            autocast_on = autocast(enabled=True)
            autocast_off = autocast(enabled=False)
            if a[0][0] > 0.5:
                with autocast_on:
                    e = torch.mm(a, b)
            else:
                with autocast_off:
                    e = torch.mm(c, d)
            return torch.mm(e, e)
        fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_conditional_autocast(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def fn(a, b):
            if False:
                print('Hello World!')
            autocast_on = autocast(enabled=True)
            autocast_off = autocast(enabled=False)
            with autocast_on if a[0][0] > 0.5 else autocast_off:
                return torch.mm(a, b)
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_nested_autocast(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def fn(a, b, c, d):
            if False:
                i = 10
                return i + 15
            with autocast(enabled=False):
                e = torch.mm(a, b)
                with autocast(enabled=True):
                    f = torch.mm(e, c)
                    with autocast(enabled=False):
                        g = torch.mm(e, d)
            return (e, f, g)
        (e, f, g) = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float32)
        self.assertEqual(f.dtype, torch.float16)
        self.assertEqual(g.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_implicitly_nested_autocast(self):
        if False:
            return 10

        @torch.jit.script
        def fn(a, b):
            if False:
                return 10
            with autocast(enabled=False), autocast(enabled=True):
                return torch.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_reused_autocast(self):
        if False:
            return 10

        @torch.jit.script
        def fn(a, b, c, d):
            if False:
                while True:
                    i = 10
            autocast_instance = autocast(enabled=True)
            with autocast_instance:
                e = torch.mm(a, b)
                with autocast_instance:
                    e = torch.mm(c, d)
                    f = torch.mm(d, e)
            g = torch.mm(e, f)
            return (e, f, g)
        (e, f, g) = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)
        self.assertEqual(g.dtype, torch.float16)

    @unittest.skipIf(True, 'unsuported autocast syntax')
    def test_reused_autocast_expr(self):
        if False:
            return 10

        @torch.jit.script
        def fn(a, b, c, d):
            if False:
                print('Hello World!')
            with autocast(enabled=True) as autocast_instance:
                e = torch.mm(a, b)
                with autocast_instance:
                    e = torch.mm(c, d)
                    f = torch.mm(d, e)
            g = torch.mm(e, f)
            return (e, f, g)
        (e, f, g) = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)
        self.assertEqual(g.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_callees(self):
        if False:
            print('Hello World!')

        def helper(a, b):
            if False:
                i = 10
                return i + 15
            return torch.mm(a, b)

        @torch.jit.script
        def fn(a, b):
            if False:
                while True:
                    i = 10
            with autocast(enabled=True):
                tmp = helper(a, b)
                tmp = helper(tmp, tmp)
                tmp = helper(tmp, tmp)
                tmp = helper(tmp, tmp)
                return helper(tmp, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_callees_with_autocast_on(self):
        if False:
            while True:
                i = 10

        def helper(a, b):
            if False:
                print('Hello World!')
            with autocast(enabled=True):
                return torch.mm(a, b)

        @torch.jit.script
        def fn(a, b):
            if False:
                while True:
                    i = 10
            with autocast(enabled=False):
                return helper(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_callees_with_autocast_off(self):
        if False:
            print('Hello World!')

        def helper(a, b):
            if False:
                return 10
            with autocast(enabled=False):
                return torch.mm(a, b)

        @torch.jit.script
        def fn(a, b):
            if False:
                return 10
            with autocast(enabled=True):
                return helper(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_eager_and_script(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def fn(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return torch.mm(a, b)
        for i in range(8):
            use_autocast = i % 2 == 0
            expected_dtype = torch.float16 if use_autocast else torch.float32
            with autocast(enabled=use_autocast):
                result = fn(self.a_fp32, self.b_fp32)
            self.assertEqual(result.dtype, expected_dtype)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_script_and_tracing(self):
        if False:
            for i in range(10):
                print('nop')

        def helper(a, b):
            if False:
                print('Hello World!')
            return torch.mm(a, b)
        traced = torch.jit.trace(helper, (self.a_fp32, self.a_fp32))

        @torch.jit.script
        def fn(a, b):
            if False:
                i = 10
                return i + 15
            with autocast(enabled=True):
                return traced(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(True, 'autocast(False) is ignored inside traced functions')
    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_script_and_tracing_with_autocast(self):
        if False:
            i = 10
            return i + 15

        def helper(a, b):
            if False:
                return 10
            with autocast(enabled=False):
                return torch.mm(a, b) * 2.0
        traced = torch.jit.trace(helper, (self.a_fp32, self.a_fp32))

        @torch.jit.script
        def fn(a, b):
            if False:
                return 10
            with autocast(enabled=True):
                return traced(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_tracing_and_script(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def fn(a, b):
            if False:
                i = 10
                return i + 15
            with autocast():
                return torch.mm(a, b)

        def traced(a, b):
            if False:
                return 10
            return fn(a, b)
        traced = torch.jit.trace(traced, (self.a_fp32, self.b_fp32))
        result = traced(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(True, 'scripted called from traced TorchScript is not yet working')
    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_tracing_with_autocast_and_script(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def fn(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return torch.mm(a, b)

        def traced(a, b):
            if False:
                print('Hello World!')
            with autocast(enabled=True):
                return fn(a, b)
        traced = torch.jit.trace(traced, (self.a_fp32, self.b_fp32))
        result = traced(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_script_module(self):
        if False:
            i = 10
            return i + 15

        class TestModule(torch.nn.Module):

            def __init__(self, N, M):
                if False:
                    print('Hello World!')
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand((N, M), dtype=torch.float32))
                self.linear = torch.nn.Linear(N, M).float()

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                with autocast(enabled=True):
                    output = self.weight.mv(input)
                    output = self.linear(output)
                    return output
        scripted_module = torch.jit.script(TestModule(2, 3)).cuda()
        input = torch.rand(3, dtype=torch.float32, device='cuda')
        result = scripted_module(input)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(True, 'autocast decorators not supported')
    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_autocast_decorator(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        @autocast(enabled=True)
        def fn(a, b):
            if False:
                return 10
            return torch.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_autocast_decorator_outside_jit(self):
        if False:
            i = 10
            return i + 15

        @autocast(enabled=True)
        @torch.jit.script
        def fn(a, b):
            if False:
                print('Hello World!')
            return torch.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_inplace(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def fn(a, b, c):
            if False:
                print('Hello World!')
            with autocast(enabled=True):
                x = torch.addmm(a, b, c)
                y = torch.addmm(a, b, c, out=a)
                z = a.addmm_(b, c)
                return (x, y, z)
        (x, y, z) = fn(self.a_fp32, self.b_fp32, self.c_fp32)
        self.assertEqual(x.dtype, torch.float16)
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(z.dtype, torch.float32)

    def _test_autocast(self, func, cast_op, *args):
        if False:
            return 10
        jit_func = torch.jit.script(func)
        o = func(*args)
        jit_o = jit_func(*args)
        if cast_op is not None:
            FileCheck().check(cast_op).run(jit_func.graph_for(*args))
        for (o0, o1) in zip(o, jit_o):
            self.assertEqual(o0.dtype, o1.dtype)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_autocast_api(self):
        if False:
            for i in range(10):
                print('nop')

        def t_autocast_cpu(x, y):
            if False:
                print('Hello World!')
            with torch.autocast('cpu', dtype=torch.bfloat16):
                return torch.mm(x, y)

        def t_autocast_cuda(x, y):
            if False:
                return 10
            with torch.autocast('cuda', dtype=torch.half):
                return torch.mm(x, y)

        def t_cuda_amp_autocast(x, y):
            if False:
                print('Hello World!')
            with torch.cuda.amp.autocast():
                return torch.mm(x, y)

        def t_cpu_amp_autocast(x, y):
            if False:
                while True:
                    i = 10
            with torch.cpu.amp.autocast():
                return torch.mm(x, y)
        x = torch.randn(5, 5, device='cuda', dtype=torch.float32)
        y = torch.randn(5, 5, device='cuda', dtype=torch.float32)
        self._test_autocast(t_autocast_cpu, 'aten::_autocast_to_reduced_precision', x, y)
        self._test_autocast(t_autocast_cuda, 'aten::_autocast_to_reduced_precision', x, y)
        self._test_autocast(t_cuda_amp_autocast, 'aten::_autocast_to_reduced_precision', x, y)
        self._test_autocast(t_cpu_amp_autocast, 'aten::_autocast_to_reduced_precision', x, y)

    @unittest.skipIf(True, 'we need to provide dtype argument at this moment')
    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_autocast_api_not_supported(self):
        if False:
            for i in range(10):
                print('nop')

        def t_autocast_cpu(x, y):
            if False:
                while True:
                    i = 10
            with torch.autocast('cpu'):
                return torch.mm(x, y)

        def t_autocast_cuda(x, y):
            if False:
                while True:
                    i = 10
            with torch.autocast('cuda'):
                return torch.mm(x, y)
        x = torch.randn(5, 5, device='cuda', dtype=torch.float32)
        y = torch.randn(5, 5, device='cuda', dtype=torch.float32)
        self._test_autocast(t_autocast_cpu, 'aten::_autocast_to_reduced_precision', x, y)
        self._test_autocast(t_autocast_cuda, 'aten::_autocast_to_reduced_precision', x, y)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_autocast_mixed_dtypes(self):
        if False:
            i = 10
            return i + 15

        def t(cpu0, cpu1, cuda0, cuda1):
            if False:
                return 10
            with torch.autocast('cpu', torch.bfloat16):
                with torch.autocast('cuda', torch.float16):
                    cpu_o = torch.mm(cpu0, cpu1)
                    cuda_o = torch.mm(cuda0, cuda1)
                    return (cpu_o, cuda_o)
        jit_t = torch.jit.script(t)
        cpu0 = torch.randn(5, 5, device='cpu', dtype=torch.float32)
        cpu1 = torch.randn(5, 5, device='cpu', dtype=torch.float32)
        cuda0 = torch.randn(5, 5, device='cuda', dtype=torch.float32)
        cuda1 = torch.randn(5, 5, device='cuda', dtype=torch.float32)
        self._test_autocast(t, 'aten::_autocast_to_reduced_precision', cpu0, cpu1, cuda0, cuda1)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_jit_executor_under_autocast(self):
        if False:
            while True:
                i = 10

        def t(cpu0, cpu1, cuda0, cuda1):
            if False:
                i = 10
                return i + 15
            cpu_o = torch.mm(cpu0, cpu1)
            cuda_o = torch.mm(cuda0, cuda1)
            return (cpu_o, cuda_o)
        jit_t = torch.jit.script(t)
        cpu0 = torch.randn(5, 5, device='cpu', dtype=torch.float32)
        cpu1 = torch.randn(5, 5, device='cpu', dtype=torch.float32)
        cuda0 = torch.randn(5, 5, device='cuda', dtype=torch.float32)
        cuda1 = torch.randn(5, 5, device='cuda', dtype=torch.float32)
        with torch.autocast('cpu', torch.bfloat16):
            with torch.autocast('cuda', torch.float16):
                self._test_autocast(t, 'aten::_autocast_to_reduced_precision', cpu0, cpu1, cuda0, cuda1)
        with torch.autocast('cpu', torch.bfloat16):
            self._test_autocast(t, 'aten::_autocast_to_reduced_precision', cpu0, cpu1, cuda0, cuda1)
        with torch.autocast('cuda', torch.float16):
            self._test_autocast(t, 'aten::_autocast_to_reduced_precision', cpu0, cpu1, cuda0, cuda1)
        self._test_autocast(t, None, cpu0, cpu1, cuda0, cuda1)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_autocast_autodiff(self):
        if False:
            while True:
                i = 10

        def t(t0, t1):
            if False:
                return 10
            o = torch.mm(t0, t1)
            return o.relu()
        jit_t = torch.jit.script(t)
        t0 = torch.randn(5, 5, device='cuda', dtype=torch.float32).requires_grad_()
        t1 = torch.randn(5, 5, device='cuda', dtype=torch.float32).requires_grad_()
        for i in range(5):
            with torch.autocast('cuda', torch.float16):
                jit_o = jit_t(t0, t1)
            jit_o.sum().backward()
        t0.grad = None
        t1.grad = None
        ref_t0 = t0.detach().requires_grad_()
        ref_t1 = t1.detach().requires_grad_()
        with torch.autocast('cuda', torch.float16):
            o = t(ref_t0, ref_t1)
            jit_o = jit_t(t0, t1)
        jit_o.sum().backward()
        o.sum().backward()
        self.assertEqual(o, jit_o)
        self.assertEqual(t0.grad, ref_t0.grad)
        self.assertEqual(t1.grad, ref_t1.grad)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(t0.grad.dtype, ref_t0.grad.dtype)
        self.assertEqual(t1.grad.dtype, ref_t1.grad.dtype)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_jit_call_method_under_autocast(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.interface
        class Iface(torch.nn.Module):

            def forward(self, x, y) -> torch.Tensor:
                if False:
                    i = 10
                    return i + 15
                pass

        class Impl(Iface):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                return torch.mm(x, y)

        class Thing1(torch.nn.Module):
            impl: Iface

            def forward(self, x, y):
                if False:
                    return 10
                with torch.cuda.amp.autocast():
                    a = torch.mm(x, y)
                    b = self.impl.forward(a, x)
                    return b
        scripted_impl = torch.jit.script(Impl())
        thing1 = Thing1()
        thing1.impl = scripted_impl
        scripted_thing1 = torch.jit.script(thing1)
        x = torch.rand([2, 2])
        y = torch.rand([2, 2])
        with torch.cuda.amp.autocast():
            ans = scripted_thing1.forward(x, y)
        self.assertEqual(torch.mm(torch.mm(x, y), x), ans)
        self.assertRaises(RuntimeError, lambda : scripted_thing1.forward(x, y))

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_jit_freeze_autocast_basic(self):
        if False:
            print('Hello World!')

        class TestModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                with torch.cuda.amp.autocast():
                    return torch.mm(x, y)
        x = torch.rand((3, 4), dtype=torch.float).cuda()
        y = torch.rand((4, 5), dtype=torch.float).cuda()
        mod = TestModule().eval()
        self._test_autocast(mod, 'aten::_autocast_to_reduced_precision', x, y)
        frozen_mod = torch.jit.freeze(torch.jit.script(mod).eval())
        FileCheck().check_count('aten::_autocast_to_reduced_precision', 2, True).run(frozen_mod.graph)
        frozen_mod(x, y)
        optimized_graph = frozen_mod.graph_for(x, y)
        FileCheck().check_count('aten::_autocast_to_reduced_precision', 2, True).run(optimized_graph)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_jit_freeze_autocast_constants(self):
        if False:
            for i in range(10):
                print('nop')

        class TestModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.x = torch.rand((3, 4), dtype=torch.float).cuda()

            def forward(self, y):
                if False:
                    return 10
                with torch.cuda.amp.autocast():
                    return torch.mm(self.x, y)
        y = torch.rand((4, 5), dtype=torch.float).cuda()
        mod = TestModule().eval()
        frozen_mod = torch.jit.freeze(torch.jit.script(mod).eval())
        FileCheck().check_count('aten::_autocast_to_reduced_precision', 1, True).run(frozen_mod.graph)
        frozen_mod(y)
        optimized_graph = frozen_mod.graph_for(y)
        FileCheck().check_count('aten::_autocast_to_reduced_precision', 1, True).run(optimized_graph)

    @unittest.skipIf(TEST_CUDA, 'CPU-only test')
    def test_jit_autocast_softmax_cpu(self):
        if False:
            print('Hello World!')

        def fn(x):
            if False:
                while True:
                    i = 10
            with torch.cpu.amp.autocast():
                return torch.nn.functional.softmax(x, dim=0)
        fn_s = torch.jit.script(fn)
        x = torch.rand((2, 2), dtype=torch.bfloat16)
        fn_s(x)
        y = fn_s(x)
        self.assertTrue(y.dtype == torch.bfloat16)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_jit_autocast_softmax_gpu(self):
        if False:
            i = 10
            return i + 15

        def fn(x):
            if False:
                return 10
            with torch.cuda.amp.autocast():
                return torch.nn.functional.softmax(x, dim=0)
        fn_s = torch.jit.script(fn)
        x = torch.rand((2, 2), dtype=torch.half).cuda()
        fn_s(x)
        y = fn_s(x)
        self.assertTrue(y.dtype == torch.float)

    def test_ignore_amp(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def foo(x):
            if False:
                print('Hello World!')
            return torch.mm(x, x)
        inp = torch.rand([10, 10], dtype=torch.float)
        foo._set_ignore_amp(True)
        with torch.cpu.amp.autocast():
            foo(inp)
            foo(inp)
        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check_not('_autocast_to_reduced').run(g)

class convbn(torch.nn.Module):

    def __init__(self, bias_enabled=True):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 7, stride=2, bias=bias_enabled)
        self.bn = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.bn(self.conv(x))

@skipIfTorchDynamo('Not a TorchDynamo suitable test')
class TestJitTraceAutocast(JitTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.previous_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.models = [MnistNet(), convbn(bias_enabled=True), convbn(bias_enabled=False)]
        self.inputs = [torch.randn(5, 1, 28, 28, device='cpu'), torch.randn(32, 3, 224, 224, device='cpu'), torch.randn(32, 3, 224, 224, device='cpu')]
        self.previous_jit_autocast_pass = torch._C._jit_set_autocast_mode(False)

    def tearDown(self):
        if False:
            print('Hello World!')
        torch._C._jit_set_autocast_mode(self.previous_jit_autocast_pass)
        torch.set_default_dtype(self.previous_default_dtype)
        super().tearDown()

    def test_generate_autocast_jit_trace_model(self):
        if False:
            for i in range(10):
                print('nop')

        def test_generate_autocast_jit_trace_model(model, x):
            if False:
                return 10
            model.eval()
            with torch.cpu.amp.autocast(cache_enabled=False), torch.no_grad():
                traced_model = torch.jit.trace(model, x)
            traced_model = torch.jit.freeze(traced_model)
        for i in range(self.models.__len__()):
            test_generate_autocast_jit_trace_model(self.models[i], self.inputs[i])

    def test_nchw_autocast_jit_trace_model(self):
        if False:
            print('Hello World!')

        def test_nchw_autocast_jit_trace_model(model, x):
            if False:
                for i in range(10):
                    print('nop')
            model.eval()
            with torch.cpu.amp.autocast(cache_enabled=False), torch.no_grad():
                traced_model = torch.jit.trace(model, x)
            traced_model = torch.jit.freeze(traced_model)
            with torch.no_grad():
                y = traced_model(x.clone())
            with torch.cpu.amp.autocast(), torch.no_grad():
                y2 = model(x.clone())
            torch.testing.assert_close(y.double(), y2.double(), rtol=0.001, atol=0.001)
        for i in range(self.models.__len__()):
            test_nchw_autocast_jit_trace_model(self.models[i], self.inputs[i])

    def test_nhwc_autocast_jit_trace_model(self):
        if False:
            for i in range(10):
                print('nop')

        def test_nhwc_autocast_jit_trace_model(model, x):
            if False:
                i = 10
                return i + 15
            model = model.to(memory_format=torch.channels_last)
            model.eval()
            with torch.cpu.amp.autocast(cache_enabled=False), torch.no_grad():
                traced_model = torch.jit.trace(model, x.to(memory_format=torch.channels_last))
            traced_model = torch.jit.freeze(traced_model)
            with torch.no_grad():
                y = traced_model(x.clone().to(memory_format=torch.channels_last))
            with torch.cpu.amp.autocast(), torch.no_grad():
                y2 = model(x.clone().to(memory_format=torch.channels_last))
            torch.testing.assert_close(y.double(), y2.double(), rtol=0.001, atol=0.001)
        for i in range(self.models.__len__()):
            if self.inputs[i].size().__len__() == 5:
                continue
            test_nhwc_autocast_jit_trace_model(self.models[i], self.inputs[i])

    def test_cat_promote(self):
        if False:
            i = 10
            return i + 15

        class TestModel(torch.nn.Module):

            def forward(self, a, b):
                if False:
                    i = 10
                    return i + 15
                return torch.cat([a, b], 0)
        with torch.jit.fuser('none'):
            for jit_freeze_or_not in [False, True]:
                test_model = TestModel().eval()
                with torch.cpu.amp.autocast(cache_enabled=False, dtype=torch.bfloat16), torch.no_grad():
                    a = torch.rand(24, 128, 128)
                    b = torch.rand(24, 128, 128, dtype=torch.bfloat16)
                    c = test_model(a, b)
                    traced = torch.jit.trace(test_model, (a, b))
                if jit_freeze_or_not:
                    traced = torch.jit.freeze(traced)
                for _ in range(3):
                    c2 = traced(a, b)
                self.assertTrue(c.dtype, torch.float32)
                self.assertTrue(c2.dtype, torch.float32)
                traced_graph = traced.graph_for(a, b)
                self.assertTrue(any((n.kind() == 'aten::to' for n in traced_graph.nodes())))

    def test_script_autocast_cpu(self):
        if False:
            print('Hello World!')

        def fn(x):
            if False:
                return 10
            if torch.is_autocast_cpu_enabled():
                return x.relu()
            else:
                return x.sin()
        fn_s = torch.jit.script(fn)
        x = torch.rand((4, 4)) - 0.5
        with torch.cpu.amp.autocast():
            self.assertEqual(fn_s(x), fn(x))
        with torch.cpu.amp.autocast(enabled=True):
            self.assertEqual(fn_s(x), fn(x))
        self.assertTrue(any(('is_autocast_cpu_enabled' in x.kind() for x in fn_s.graph.nodes())))

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_script_autocast_cuda(self):
        if False:
            i = 10
            return i + 15

        def fn(x):
            if False:
                while True:
                    i = 10
            if torch.is_autocast_enabled():
                return x.relu()
            else:
                return x.sin()
        fn_s = torch.jit.script(fn)
        x = torch.rand((4, 4)) - 0.5
        with torch.cpu.amp.autocast():
            self.assertEqual(fn_s(x), fn(x))
        with torch.cuda.amp.autocast(enabled=True):
            self.assertEqual(fn_s(x), fn(x))
        self.assertTrue(any(('is_autocast_enabled' in x.kind() for x in fn_s.graph.nodes())))

    def test_scripted_aliasing(self):
        if False:
            while True:
                i = 10

        def fn(x):
            if False:
                i = 10
                return i + 15
            if torch.is_autocast_enabled():
                y = True
            else:
                y = False
            with torch.cuda.amp.autocast(enabled=True):
                z = x.relu()
            return (y, z)
        fn_s = torch.jit.script(fn)
        graph = fn_s.graph
        aliasdb = graph.alias_db()
        is_enabled_nodes = graph.findAllNodes('aten::is_autocast_enabled')
        enter_nodes = graph.findAllNodes('prim::Enter')
        self.assertEqual(len(is_enabled_nodes), 1)
        self.assertEqual(len(enter_nodes), 1)
        self.assertFalse(aliasdb.move_after_topologically_valid(is_enabled_nodes[0], enter_nodes[0]))

    def test_script_autocast_enable_and_check(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(x, y) -> Tuple[torch.Tensor, bool, torch.Tensor, bool, torch.Tensor, bool]:
            if False:
                i = 10
                return i + 15
            b1 = torch.is_autocast_cpu_enabled()
            v1 = torch.mm(x, y)
            with torch.cpu.amp.autocast(enabled=True):
                b2 = torch.is_autocast_cpu_enabled()
                v2 = torch.mm(x, y)
                with torch.cpu.amp.autocast(enabled=False):
                    b3 = torch.is_autocast_cpu_enabled()
                    v3 = torch.mm(x, y)
            return (v1, b1, v2, b2, v3, b3)

        def check_fn_results(arr):
            if False:
                return 10
            [v1, b1, v2, b2, v3, b3] = arr
            self.assertTrue((v1.dtype == torch.float) != b1)
            self.assertTrue((v2.dtype == torch.float) != b2)
            self.assertTrue((v3.dtype == torch.float) != b3)
        x = torch.rand((2, 2), dtype=torch.float)
        y = torch.rand((2, 2), dtype=torch.float)
        fn_s = torch.jit.script(fn)
        with torch.cpu.amp.autocast(enabled=False):
            check_fn_results(fn(x, y))
            check_fn_results(fn_s(x, y))
        with torch.cpu.amp.autocast(enabled=True):
            check_fn_results(fn(x, y))
            check_fn_results(fn_s(x, y))
if __name__ == '__main__':
    run_tests()