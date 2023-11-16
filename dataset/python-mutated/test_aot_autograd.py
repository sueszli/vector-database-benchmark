import re
from textwrap import dedent
from unittest.mock import patch
import torch
import torch._dynamo
import torch._dynamo.test_case
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch._dynamo.testing import CompileCounter, expectedFailureDynamic, rand_strided
from torch._functorch.aot_autograd import _aot_export_function, create_functional_call
from torch.profiler import profile
from torch.testing._internal.common_utils import compare_equal_outs_and_grads

def maybe_dupe_op(x):
    if False:
        i = 10
        return i + 15
    y = x + 1
    z = x + 2
    if x.numel() < 5:
        return (y, y)
    else:
        return (y, z)
aten = torch.ops.aten
lib = torch.library.Library('custom', 'DEF')
lib.define('maybe_dupe_op(Tensor a) -> (Tensor, Tensor)')
lib.impl('maybe_dupe_op', maybe_dupe_op, 'CPU')
lib.impl('maybe_dupe_op', maybe_dupe_op, 'Meta')

class AotAutogradFallbackTests(torch._dynamo.test_case.TestCase):

    def test_LSTM(self):
        if False:
            print('Hello World!')

        class Repro(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.self_mod_model_lstm_lstm = torch.nn.LSTM(64, 64, num_layers=2, bidirectional=True)

            def forward(self, permute: torch.Tensor):
                if False:
                    for i in range(10):
                        print('nop')
                self_mod_model_lstm_lstm = self.self_mod_model_lstm_lstm(permute)
                return (self_mod_model_lstm_lstm,)
        mod = Repro()
        aot_mod = torch._dynamo.optimize('aot_eager')(mod)
        args = [((92, 4, 64), (1, 5888, 92), torch.float32, 'cpu', False)]
        args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]
        eager_result = mod(*args)
        aot_result = aot_mod(*args)
        self.assertTrue(torch._dynamo.testing.same(eager_result, aot_result))

    def test_mutation(self):
        if False:
            i = 10
            return i + 15

        def fn(param, y):
            if False:
                for i in range(10):
                    print('nop')
            prev_grad = torch.is_grad_enabled()
            try:
                torch.set_grad_enabled(False)
                param.add_(y)
            finally:
                torch.set_grad_enabled(prev_grad)
            return y
        y = torch.randn(4)
        x = torch.nn.Parameter(torch.randn(4))
        aot_fn = torch._dynamo.optimize('aot_eager')(fn)
        aot_fn(x, y)

    def test_mutation1(self):
        if False:
            return 10

        def fn(_stack0: torch.Tensor, diagonal_chunked_attention_scores: torch.Tensor):
            if False:
                i = 10
                return i + 15
            getitem = diagonal_chunked_attention_scores[slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None)]
            _stack0[slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None)] = getitem
            view = _stack0.view(1, 12, 1024, 513)
            return (view,)
        x = torch.randn(torch.Size([12, 4, 256, 513]))
        y = torch.randn(torch.Size([12, 3, 512, 513]))
        aot_fn = torch._dynamo.optimize('aot_eager')(fn)
        aot_fn(x, y)

    def test_negative_testing_mutation(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(_stack0: torch.Tensor, diagonal_chunked_attention_scores: torch.Tensor):
            if False:
                for i in range(10):
                    print('nop')
            getitem = diagonal_chunked_attention_scores[slice(None, None, None), slice(None, None, None), slice(None, 256, None), slice(None, 257, None)]
            _stack0 = torch.sin(_stack0)
            _stack0[slice(None, None, None), slice(None, -1, None), slice(None, None, None), slice(256, None, None)] = getitem
            view = _stack0.view(1, 12, 1024, 513)
            return (view,)
        x = torch.randn(torch.Size([12, 4, 256, 513]))
        y = torch.randn(torch.Size([12, 3, 512, 513]))
        aot_fn = torch._dynamo.optimize('aot_eager')(fn)
        aot_fn(x, y)

    def test_negative_testing(self):
        if False:
            while True:
                i = 10

        def fn(x, y):
            if False:
                print('Hello World!')
            return torch.sin(x).add_(y)
        y = torch.randn(4)
        x = torch.randn(4)
        aot_fn = torch._dynamo.optimize('aot_eager')(fn)
        aot_fn(x, y)

    def test_call_fn_with_non_const_inputs_aot_safe(self):
        if False:
            while True:
                i = 10

        class ModuleSpecialFwd(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(5, 5))

            def _conv_forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.conv._conv_forward(x, self.conv.weight, self.conv.bias)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self._conv_forward(x)
        mod = ModuleSpecialFwd()
        rx = torch.randn([3, 10, 10])
        real = mod(rx)
        (graph, _) = torch._dynamo.export(mod)(rx)
        self.assertTrue(torch._dynamo.testing.same(real, graph(rx)))
        aot_fn = torch._dynamo.optimize('aot_eager')(graph)
        aot_fn(rx)

    def test_call_fn_with_non_const_inputs_aot_unsafe(self):
        if False:
            print('Hello World!')

        class ModuleSpecialFwd(torch.nn.Module):

            def _some_bad_fwd(self, param, y):
                if False:
                    return 10
                prev_grad = torch.is_grad_enabled()
                try:
                    torch.set_grad_enabled(False)
                    param.add_(y)
                finally:
                    torch.set_grad_enabled(prev_grad)
                return y

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                return self._some_bad_fwd(x, y)
        mod = ModuleSpecialFwd()
        x = torch.nn.Parameter(torch.randn(4))
        y = torch.randn([4])
        real = mod(x, y)
        (graph, _) = torch._dynamo.export(mod)(x, y)
        self.assertTrue(torch._dynamo.testing.same(real, graph(x, y)))
        aot_fn = torch._dynamo.optimize('aot_eager')(graph)
        aot_fn(x, y)

    def test_call_fn_with_non_const_inputs_aot_unsafe_control_flow(self):
        if False:
            return 10

        class ModuleSpecialFwd(torch.nn.Module):

            def _some_bad_fwd(self, param, y):
                if False:
                    while True:
                        i = 10
                if y[0][0] < 3:
                    return y + param
                return param * y

            def forward(self, x, y):
                if False:
                    return 10
                a = x * y
                a = self._some_bad_fwd(a, a)
                b = x + y
                return a * b
        mod = ModuleSpecialFwd()
        x = torch.nn.Parameter(torch.randn([2, 2]))
        y = torch.randn([2, 2])
        real = mod(x, y)
        gms = []
        counter = CompileCounter()

        def capturing_fn(gm, inputs):
            if False:
                print('Hello World!')
            nonlocal gms
            gms.append(gm)
            return counter(gm, inputs)
        optimized_mod = torch._dynamo.optimize(capturing_fn)(mod)
        self.assertTrue(torch._dynamo.testing.same(real, optimized_mod(x, y)))
        self.assertEqual(counter.frame_count, 4)
        self.assertEqual(counter.op_count, 7)
        torch._dynamo.reset()
        aot_fn = torch._dynamo.optimize('aot_eager')(optimized_mod)
        aot_fn(x, y)

    @patch('torch._functorch.config.debug_assert', True)
    def test_requires_grad_fake_via_dynamo_recompiles(self):
        if False:
            for i in range(10):
                print('nop')

        class F(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return (x + y,)
        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        z = torch.randn(3, 3, requires_grad=False)
        cc = torch._dynamo.testing.CompileCounterWithBackend('aot_eager')
        failure_reason = None

        def guard_fail_fn(failure):
            if False:
                print('Hello World!')
            nonlocal failure_reason
            failure_reason = failure[0]
        fxy = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        compare_equal_outs_and_grads(self, F(), fxy, (x, y))
        compare_equal_outs_and_grads(self, F(), fxy, (x, z))
        self.assertExpectedInline(failure_reason, "tensor 'L['y']' requires_grad mismatch. expected requires_grad=1")
        failure_reason = None
        self.assertEqual(cc.frame_count, 2)
        torch._dynamo.reset()
        cc = torch._dynamo.testing.CompileCounterWithBackend('aot_eager')
        fxz = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        compare_equal_outs_and_grads(self, F(), fxz, (x, z))
        compare_equal_outs_and_grads(self, F(), fxz, (x, z))
        self.assertEqual(cc.frame_count, 1)
        self.assertTrue(failure_reason is None)

    def test_double_backward_errors(self):
        if False:
            print('Hello World!')
        for grad_output in (torch.tensor(1.0, requires_grad=True), None):
            x = torch.tensor(1.0, requires_grad=True)
            err = 'torch.compile with aot_autograd does not currently support double backward'

            def f1(x):
                if False:
                    return 10
                y = x.sin().exp()
                (gx,) = torch.autograd.grad(y, x, create_graph=True, grad_outputs=grad_output)
                torch.autograd.grad(gx, x)
                return gx
            compiled_f1 = torch.compile(backend='aot_eager')(f1)
            f1(x)
            with self.assertRaisesRegex(RuntimeError, err):
                compiled_f1(x)

            def f2(x):
                if False:
                    print('Hello World!')
                y = x.sin().exp()
                (gx,) = torch.autograd.grad(y, x, create_graph=True, grad_outputs=grad_output)
                return gx
            compiled_f2 = torch.compile(backend='aot_eager')(f2)
            gx = compiled_f2(x)
            with self.assertRaisesRegex(RuntimeError, err):
                torch.autograd.grad(gx, x)

            def f3(x):
                if False:
                    return 10
                y = x.sin().exp()
                return y
            compiled_f3 = torch.compile(backend='aot_eager')(f3)
            y = compiled_f3(x)
            (gx,) = torch.autograd.grad(y, x, create_graph=True, grad_outputs=grad_output)
            with self.assertRaisesRegex(RuntimeError, err):
                torch.autograd.grad(gx, x)

        def f4(x):
            if False:
                return 10
            y = x.sin().exp()
            return y
        compiled_f4 = torch.compile(backend='aot_eager')(f4)
        x = torch.tensor(1.0, requires_grad=True)
        y = compiled_f4(x)
        (gx,) = torch.autograd.grad(y, x, create_graph=False, grad_outputs=grad_output)

    @patch('torch._functorch.config.debug_assert', True)
    def test_arg_dupe_via_dynamo_recompiles(self):
        if False:
            return 10

        class F(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                x = x.trunc_()
                y = y.trunc_()
                return (x + y,)
        x = torch.randn(3, 3, requires_grad=True)
        (x1, x2, x3, x4) = (x.clone(), x.clone(), x.clone(), x.clone())
        y = torch.randn(3, 3, requires_grad=True)
        (y1, y2, y4) = (y.clone(), y.clone(), y.clone())
        cc = torch._dynamo.testing.CompileCounterWithBackend('aot_eager')
        failure_reason = None

        def guard_fail_fn(failure):
            if False:
                print('Hello World!')
            nonlocal failure_reason
            failure_reason = failure[0]
        fxy = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        fxy(x1, y1)
        fxy(x2, y2)
        self.assertTrue(failure_reason is None)
        failure_reason = None
        self.assertEqual(cc.frame_count, 1)
        torch._dynamo.reset()
        cc = torch._dynamo.testing.CompileCounterWithBackend('aot_eager')
        fxx = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        fxx(x3, x3)
        fxx(x4, y4)
        self.assertEqual(cc.frame_count, 2)
        self.assertExpectedInline(failure_reason, "L['x'] is L['y']")

    @patch('torch._functorch.config.debug_assert', True)
    def test_arg_dupe_via_dynamo_recompiles_many_args_param_non_tensor_arg(self):
        if False:
            while True:
                i = 10

        class F(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.mean = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, a, b, c, d, e, f):
                if False:
                    print('Hello World!')
                a.trunc_()
                b.trunc_()
                c.trunc_()
                d.trunc_()
                return (a + b + c + d + self.mean) * e * f
        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        (a1, a2, a3, a4) = (a.clone(), a.clone(), a.clone(), a.clone())
        (b1, b2, b3, b4) = (b.clone(), b.clone(), b.clone(), b.clone())
        failure_reason = None

        def guard_fail_fn(failure):
            if False:
                while True:
                    i = 10
            nonlocal failure_reason
            failure_reason = failure[0]
        self.assertTrue(failure_reason is None)
        cc = torch._dynamo.testing.CompileCounterWithBackend('aot_eager')
        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a1, a1, a1, a1, 2, 2)
        f(a2, b2, b2, b2, 2, 2)
        self.assertEqual(cc.frame_count, 2)
        self.assertExpectedInline(failure_reason, "L['a'] is L['b']")
        torch._dynamo.reset()
        cc = torch._dynamo.testing.CompileCounterWithBackend('aot_eager')
        c = torch.randn(3, 3, requires_grad=True)
        d = torch.randn(3, 3, requires_grad=True)
        (c3, c4) = (c.clone(), c.clone())
        (d3, d4) = (d.clone(), d.clone())
        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a3, b3, c3, c3, 3, 3)
        f(a4, b4, c4, d4, 3, 3)
        self.assertEqual(cc.frame_count, 2)
        self.assertExpectedInline(failure_reason, "L['c'] is L['d']")

    @patch('torch._functorch.config.debug_assert', True)
    def test_arg_dupe_via_dynamo_recompiles_many_with_global(self):
        if False:
            i = 10
            return i + 15
        z = None

        class F(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.mean = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, a, b, c, d, e, f):
                if False:
                    return 10
                a.trunc_()
                b.trunc_()
                c.trunc_()
                d.trunc_()
                return (a + b + c + d + z + self.mean) * e * f
        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        z = a
        (a1, a2, a3, a4) = (a.clone(), a.clone(), a.clone(), a.clone())
        (b1, b2, b3, b4) = (b.clone(), b.clone(), b.clone(), b.clone())
        failure_reason = None

        def guard_fail_fn(failure):
            if False:
                while True:
                    i = 10
            nonlocal failure_reason
            failure_reason = failure[0]
        self.assertTrue(failure_reason is None)
        cc = torch._dynamo.testing.CompileCounterWithBackend('aot_eager')
        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a1, a1, a1, a1, 2, 2)
        f(a2, b2, b2, b2, 2, 2)
        self.assertEqual(cc.frame_count, 2)
        self.assertExpectedInline(failure_reason, "L['a'] is L['b']")

    @patch('torch._functorch.config.debug_assert', True)
    def test_arg_dupe_via_dynamo_recompiles_many_args_param_non_tensor_arg_list(self):
        if False:
            i = 10
            return i + 15

        class F(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.mean = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, e, f, a, b, c, d):
                if False:
                    print('Hello World!')
                a.trunc_()
                b.trunc_()
                c.trunc_()
                d.trunc_()
                return (a + b + c + d + self.mean) * e[0] * f[0]
        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        (a1, a2, a3, a4) = (a.clone(), a.clone(), a.clone(), a.clone())
        (b1, b2, b3, b4) = (b.clone(), b.clone(), b.clone(), b.clone())
        failure_reason = None

        def guard_fail_fn(failure):
            if False:
                i = 10
                return i + 15
            nonlocal failure_reason
            failure_reason = failure[0]
        self.assertTrue(failure_reason is None)
        cc = torch._dynamo.testing.CompileCounterWithBackend('aot_eager')
        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f([3, 2, 1], [4, 5, 6], a1, a1, a1, a1)
        f([3, 2, 1], [4, 5, 6], a2, b2, b2, b2)
        self.assertEqual(cc.frame_count, 2)
        self.assertExpectedInline(failure_reason, "L['a'] is L['b']")
        torch._dynamo.reset()
        cc = torch._dynamo.testing.CompileCounterWithBackend('aot_eager')
        c = torch.randn(3, 3, requires_grad=True)
        d = torch.randn(3, 3, requires_grad=True)
        (c3, c4) = (c.clone(), c.clone())
        (d3, d4) = (d.clone(), d.clone())
        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f([3, 2, 1], [4, 5, 6], a3, b3, c3, c3)
        f([3, 2, 1], [4, 5, 6], a4, b4, c4, d4)
        self.assertEqual(cc.frame_count, 2)

    @patch('torch._functorch.config.debug_assert', True)
    def test_arg_dupe_via_dynamo_recompiles_many_args_param(self):
        if False:
            while True:
                i = 10

        class F(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.mean = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, a, b, c, d):
                if False:
                    while True:
                        i = 10
                a.trunc_()
                b.trunc_()
                c.trunc_()
                d.trunc_()
                return a + b + c + d + self.mean
        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        (a1, a2, a3, a4) = (a.clone(), a.clone(), a.clone(), a.clone())
        (b1, b2, b3, b4) = (b.clone(), b.clone(), b.clone(), b.clone())
        failure_reason = None

        def guard_fail_fn(failure):
            if False:
                i = 10
                return i + 15
            nonlocal failure_reason
            failure_reason = failure[0]
        self.assertTrue(failure_reason is None)
        cc = torch._dynamo.testing.CompileCounterWithBackend('aot_eager')
        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a1, a1, a1, a1)
        f(a2, b2, b2, b2)
        self.assertEqual(cc.frame_count, 2)
        self.assertExpectedInline(failure_reason, "L['a'] is L['b']")
        torch._dynamo.reset()
        cc = torch._dynamo.testing.CompileCounterWithBackend('aot_eager')
        c = torch.randn(3, 3, requires_grad=True)
        d = torch.randn(3, 3, requires_grad=True)
        (c3, c4) = (c.clone(), c.clone())
        (d3, d4) = (d.clone(), d.clone())
        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a3, b3, c3, c3)
        f(a4, b4, c4, d4)
        self.assertEqual(cc.frame_count, 2)
        self.assertExpectedInline(failure_reason, "L['c'] is L['d']")

    @patch('torch._functorch.config.debug_assert', True)
    def test_arg_dupe_via_dynamo_recompiles_many_args(self):
        if False:
            for i in range(10):
                print('nop')

        class F(torch.nn.Module):

            def forward(self, a, b, c, d):
                if False:
                    return 10
                a.trunc_()
                b.trunc_()
                c.trunc_()
                d.trunc_()
                return (a + b + c + d,)
        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        (a1, a2, a3, a4) = (a.clone(), a.clone(), a.clone(), a.clone())
        (b1, b2, b3, b4) = (b.clone(), b.clone(), b.clone(), b.clone())
        failure_reason = None

        def guard_fail_fn(failure):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal failure_reason
            failure_reason = failure[0]
        self.assertTrue(failure_reason is None)
        cc = torch._dynamo.testing.CompileCounterWithBackend('aot_eager')
        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a1, a1, a1, a1)
        f(a2, b2, b2, b2)
        self.assertEqual(cc.frame_count, 2)
        self.assertExpectedInline(failure_reason, "L['a'] is L['b']")
        torch._dynamo.reset()
        cc = torch._dynamo.testing.CompileCounterWithBackend('aot_eager')
        c = torch.randn(3, 3, requires_grad=True)
        d = torch.randn(3, 3, requires_grad=True)
        (c3, c4) = (c.clone(), c.clone())
        (d3, d4) = (d.clone(), d.clone())
        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a3, b3, c3, c3)
        f(a4, b4, c4, d4)
        self.assertEqual(cc.frame_count, 2)
        self.assertExpectedInline(failure_reason, "L['c'] is L['d']")

    @expectedFailureDynamic
    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    @patch('torch._functorch.config.debug_assert', True)
    def test_multiple_aot_autograd_calls_dupe_args(self):
        if False:
            print('Hello World!')

        class WrapperModule(torch.nn.Module):

            def __init__(self, mod):
                if False:
                    print('Hello World!')
                super().__init__()
                self.mod = mod

            def forward(self, *args):
                if False:
                    print('Hello World!')
                out = self.mod(*args)
                if isinstance(out, (list, tuple)):
                    return out
                return (out,)

        def compile_submod(input_mod, args):
            if False:
                i = 10
                return i + 15
            from functorch.compile import nop
            from torch._functorch.aot_autograd import aot_module_simplified

            class WrapperModule(torch.nn.Module):

                def __init__(self):
                    if False:
                        while True:
                            i = 10
                    super().__init__()
                    self.original = input_mod
                    self.submod = aot_module_simplified(input_mod, args, nop)

                def forward(self, *args):
                    if False:
                        i = 10
                        return i + 15
                    return self.submod(*args)
            return WrapperModule()

        def test_compile(fx_g, example_inps):
            if False:
                print('Hello World!')
            split_gm = torch.fx.passes.split_module.split_module(fx_g, None, lambda node: 1 if 'mul' in str(node) else 0)
            submod_1_inps = split_gm.submod_0(*example_inps)
            split_gm.submod_0 = compile_submod(WrapperModule(split_gm.submod_0), example_inps)
            split_gm.submod_1 = compile_submod(WrapperModule(split_gm.submod_1), submod_1_inps)
            return split_gm

        @torch._dynamo.optimize(test_compile)
        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            (b, c) = torch.ops.custom.maybe_dupe_op(a)
            return (b.mul_(c),)
        f(torch.ones(4))
        f(torch.ones(6))

    def test_nn_parameter_construction(self):
        if False:
            i = 10
            return i + 15

        def fn(x):
            if False:
                i = 10
                return i + 15
            y = x.sin()
            z = torch.nn.Parameter(torch.ones(1))
            return y + z
        x = torch.rand((4, 4))
        opt_fn = torch._dynamo.optimize('aot_eager')(fn)
        self.assertTrue(torch._dynamo.testing.same(fn(x), opt_fn(x)))

    def test_aot_sequence_nr(self):
        if False:
            i = 10
            return i + 15

        class Model(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), stride=1, padding='same', bias=True)
                self.bn1 = torch.nn.BatchNorm2d(num_features=16)
                self.relu1 = torch.nn.ReLU()
                self.fc1 = torch.nn.Linear(in_features=1638400, out_features=1)
                self.loss_fn = torch.nn.L1Loss()

            def forward(self, x, target):
                if False:
                    while True:
                        i = 10
                y = x
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = x + y
                x = torch.flatten(x)
                x = self.fc1(x)
                output = self.loss_fn(x, target)
                return (output,)
        mod = Model()
        mod.train()
        x = torch.rand(100, 16, 32, 32, requires_grad=True)
        target = torch.rand(1)
        (g_mod, _) = torch._dynamo.export(mod, x, target)

        def _prepare_model_args():
            if False:
                while True:
                    i = 10
            named_parameters = dict(g_mod.named_parameters(remove_duplicate=False))
            named_buffers = dict(g_mod.named_buffers(remove_duplicate=False))
            params_and_buffers = {**dict(named_parameters), **dict(named_buffers)}
            (params_and_buffers_flat, params_spec) = pytree.tree_flatten(params_and_buffers)
            params_len = len(params_and_buffers_flat)
            functional_call = create_functional_call(g_mod, params_spec, params_len)
            return (params_and_buffers_flat, functional_call)
        (full_args, fn_to_trace) = _prepare_model_args()
        param_and_buf_len = len(full_args)
        full_args.extend([x, target])
        with torch.enable_grad(), fx_traceback.preserve_node_meta():
            (fx_g, _, _, _) = _aot_export_function(fn_to_trace, full_args, decompositions=None, num_params_buffers=param_and_buf_len, no_tangents=True)
        min_seq_nr = -1
        seq_table = 'SeqNr|OrigAten|SrcFn\n'
        for node in fx_g.graph.nodes:
            if 'call_' in node.op and 'getitem' not in str(node.target):
                seq_nr = node.meta.get('seq_nr', -1)
                if seq_nr < 0:
                    continue
                if min_seq_nr < 0:
                    min_seq_nr = seq_nr
                source_fn_stack = node.meta.get('source_fn_stack', [])
                orig_aten = node.meta.get('original_aten', '')
                mod_name = ''
                if len(source_fn_stack) > 0:
                    mod_name = source_fn_stack[-1][0]
                seq_nr = seq_nr - min_seq_nr
                seq_table = seq_table + f'{seq_nr}|{orig_aten}|{mod_name}\n'
        self.maxDiff = None
        self.assertExpectedInline(seq_table, dedent('SeqNr|OrigAten|SrcFn\n0|aten.convolution.default|l__self___conv1\n0|aten.add.Tensor|l__self___bn1\n1|aten._native_batch_norm_legit_functional.default|l__self___bn1\n2|aten.relu.default|l__self___relu1\n2|aten.detach.default|l__self___relu1\n3|aten.add.Tensor|add\n4|aten.view.default|flatten\n5|aten.view.default|l__self___fc1\n6|aten.t.default|l__self___fc1\n7|aten.addmm.default|l__self___fc1\n8|aten.view.default|l__self___fc1\n9|aten.sub.Tensor|l__self___loss_fn\n10|aten.abs.default|l__self___loss_fn\n11|aten.mean.default|l__self___loss_fn\n11|aten.ones_like.default|\n11|aten.expand.default|\n11|aten.div.Scalar|\n10|aten.sgn.default|\n10|aten.mul.Tensor|\n8|aten.view.default|\n7|aten.t.default|\n7|aten.mm.default|\n7|aten.t.default|\n7|aten.mm.default|\n7|aten.t.default|\n7|aten.sum.dim_IntList|\n7|aten.view.default|\n6|aten.t.default|\n5|aten.view.default|\n4|aten.view.default|\n2|aten.detach.default|\n2|aten.threshold_backward.default|\n1|aten.native_batch_norm_backward.default|\n0|aten.convolution_backward.default|\n11|aten.add.Tensor|\n'))

    def test_aot_export_joint_simple_repro(self):
        if False:
            while True:
                i = 10

        class Mod(torch.nn.Module):

            def __init__(self, *args, **kwargs) -> None:
                if False:
                    i = 10
                    return i + 15
                super().__init__(*args, **kwargs)
                self.linear = torch.nn.Linear(5, 7)

            def forward(self, x):
                if False:
                    return 10
                return self.linear(x)

        def mini_backend(gm, sample_inputs):
            if False:
                return 10
            from torch._functorch.aot_autograd import aot_export_joint_simple
            fake_mode = torch._dynamo.utils.detect_fake_mode(sample_inputs)
            with patch.object(fake_mode, 'allow_non_fake_inputs', True), fake_mode:
                return aot_export_joint_simple(gm, sample_inputs, trace_joint=False)
        sample_inputs = [torch.rand((3, 4, 5))]
        model = Mod()
        m_compiled = torch.compile(model, backend=mini_backend)
        out_ref = model(*sample_inputs)
        out_test = m_compiled(*sample_inputs)
        self.assertEqual(out_ref, out_test)

    def test_eager_sequence_nr(self):
        if False:
            return 10

        class Model(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), stride=1, padding='same', bias=True)
                self.bn1 = torch.nn.BatchNorm2d(num_features=16)
                self.relu1 = torch.nn.ReLU()
                self.fc1 = torch.nn.Linear(in_features=1638400, out_features=1)
                self.loss_fn = torch.nn.L1Loss()

            def forward(self, x, target):
                if False:
                    i = 10
                    return i + 15
                y = x
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = x + y
                x = torch.flatten(x)
                x = self.fc1(x)
                output = self.loss_fn(x, target)
                return (output,)

        def grad_with_create_graph(mod, x, target):
            if False:
                while True:
                    i = 10
            y = mod(x, target)
            (gx,) = torch.autograd.grad(y[0], x, create_graph=True, grad_outputs=grad_output)
            return gx
        x = torch.rand(100, 16, 32, 32, requires_grad=True)
        target = torch.rand(1)
        mod = Model()
        args = [mod, x, target]
        grad_output = torch.tensor(1.0, requires_grad=True)
        compiled_f1 = torch.compile(backend='aot_eager')(grad_with_create_graph)
        model_instance = compiled_f1
        with profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as kineto_prof:
            res = model_instance(*args)
        bwd_set = set()
        prof_str = 'SeqNr|Thread|FwdThread|Name\n'
        for event in kineto_prof.events():
            if event.sequence_nr >= 0:
                prof_str = prof_str + f'{event.sequence_nr}|{event.thread}|{event.fwd_thread}|{event.name}|\n'
                if re.search('Backward[01]', event.name):
                    bwd_set.add(event.sequence_nr)
        self.assertTrue(len(bwd_set), 13)

    def test_aot_grad_mode_mutation(self):
        if False:
            for i in range(10):
                print('nop')
        for compiler in ['aot_eager', 'inductor']:

            def f(x):
                if False:
                    print('Hello World!')
                y = x * x
                torch.set_grad_enabled(False)
                return (y.clone(), y)
            f_compiled = torch.compile(f, backend=compiler, fullgraph=True)
            torch.set_grad_enabled(True)
            x = torch.ones(3, requires_grad=True) * 3
            y_ref = f(x)
            self.assertEqual(torch.is_grad_enabled(), False)
            torch.set_grad_enabled(True)
            y = f_compiled(x)
            self.assertEqual(torch.is_grad_enabled(), False)
            torch.set_grad_enabled(True)
            self.assertEqual(y_ref, y)
            self.assertIsNone(y_ref[0].grad_fn)
            self.assertIsNone(y[0].grad_fn)
            self.assertIsNotNone(y_ref[1].grad_fn)
            self.assertIsNotNone(y[1].grad_fn)
            self.assertEqual(sum(y_ref[1].grad_fn(torch.tensor([-1.0, 2.0, 0.0]))), sum((x for x in y[1].grad_fn.apply(None, torch.tensor([-1.0, 2.0, 0.0])) if x is not None)))
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    run_tests()