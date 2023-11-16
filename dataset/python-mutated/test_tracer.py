import unittest
import io
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.testing import FileCheck
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import suppress_warnings, skipIfCompiledWithoutNumpy, enable_profiling_mode_for_profiling_tests, IS_SANDCASTLE, TemporaryFileName, skipIfCrossRef, skipIfTorchDynamo
from torch.testing._internal.jit_utils import JitTestCase, enable_cpu_fuser, _tmp_donotuse_dont_inline_everything, _trace, RUN_CUDA, RUN_CUDA_MULTI_GPU, make_global
from torch.testing._internal.common_cuda import with_tf32_off
from torch import Tensor
from collections import namedtuple
from itertools import chain
from typing import Dict, List, Optional, Tuple
import warnings
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

@skipIfTorchDynamo('Not a suitable test for TorchDynamo')
class TestTracer(JitTestCase):

    @unittest.skipIf(not RUN_CUDA, 'requires CUDA')
    def test_large_nbr_kernel_args(self):
        if False:
            print('Hello World!')

        class Recurrence(nn.Module):

            def __init__(self, seq_len):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.seq_len = seq_len

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                input = input.transpose(0, 1)
                output = []
                for i in range(self.seq_len):
                    b = input[i] * 2
                    output.append(b)
                output = torch.cat(output, 0).view(input.size(0), *output[0].size())
                output = output.transpose(0, 1)
                return output
        input_size = 8
        batch_size = 2
        seq_len = 130
        rec = Recurrence(seq_len)
        input = torch.rand(batch_size, seq_len, input_size)
        torch.cuda.set_device(0)
        rec = rec.cuda()
        input = input.cuda()
        traced_rec = torch.jit.trace(rec, input)

    def test_trace_legacy_ctor(self):
        if False:
            return 10

        class MyModule(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return (x + 1, torch.FloatTensor([0]))
        traced_rec = torch.jit.trace(MyModule(), torch.randn(2, 2))

    def test_simple(self):
        if False:
            while True:
                i = 10
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)

        def f(x, y):
            if False:
                while True:
                    i = 10
            return torch.sigmoid(torch.tanh(x * (x + y)))
        self.checkTrace(f, (x, y))

    def test_trace_checking_with_global_name(self):
        if False:
            i = 10
            return i + 15

        class MyClass(torch.nn.Module):

            def forward(self, xs: List[Tensor]):
                if False:
                    print('Hello World!')
                y = torch.cat(xs, dim=0)
                return y
        model = MyClass()
        global input1, input2
        input1 = torch.ones(2, 2)
        input2 = torch.ones(2, 2)
        m2 = torch.jit.trace(model, ((input1, input2),))

    def test_trace_aliased_parameter(self):
        if False:
            for i in range(10):
                print('nop')

        class M(nn.Module):

            def __init__(self, x):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.x = nn.Parameter(x)

            def forward(self, y):
                if False:
                    print('Hello World!')
                return self.x + y
        m = M(torch.rand(3, 4))
        r = torch.jit.trace(m, m.x)
        t2 = torch.rand(3, 4)
        self.assertEqual(r(t2), m.x + t2)

    def test_trace_nested_fn(self):
        if False:
            while True:
                i = 10

        class TracedInlineDecision(torch.nn.Module):

            def forward(self, x, flag):
                if False:
                    print('Hello World!')

                @torch.jit.script
                def make_decision(flag, x):
                    if False:
                        i = 10
                        return i + 15
                    if flag:
                        return x
                    else:
                        return torch.zeros_like(x)
                x = torch.neg(x)
                return make_decision(flag, x)
        decision = TracedInlineDecision()
        torch.jit.trace(decision, (torch.rand(3, 4), torch.tensor([True], dtype=torch.bool)), check_trace=True)

    def test_trace_single_tuple(self):
        if False:
            while True:
                i = 10
        x = torch.tensor(2.0)

        def f2(x):
            if False:
                while True:
                    i = 10
            return (x,)
        jit_f2 = torch.jit.trace(f2, x)
        assert f2(x) == jit_f2(x)

    def test_trace_out_operator_with_two_output(self):
        if False:
            i = 10
            return i + 15
        example_input = torch.rand(2, 8)
        (out_1, out_2) = torch.cummax(example_input, 1)

        def run_cummax(example_input, out_1, out_2):
            if False:
                while True:
                    i = 10
            (output_1, output_2) = torch.cummax(example_input, 1, out=(out_1, out_2))
            return (output_1, output_2)
        trace_model = torch.jit.trace(run_cummax, (example_input, out_1, out_2))

    def test_trace_namedtuple(self):
        if False:
            while True:
                i = 10
        Point = namedtuple('point', ['x', 'y'])

        def f(p):
            if False:
                i = 10
                return i + 15
            if type(p) is tuple:
                p = Point(*p)
            return p.x + p.y
        p = Point(torch.randn(1), torch.randn(1))
        traced = torch.jit.trace(f, (p,))
        self.assertEqual(f(p), traced(p))

    def test_trace_topk(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return x.topk(y, dim=1)[1]
        mod = M()
        inputs = (torch.randint(0, 10, (20, 20)), torch.tensor(17))
        traced_func = torch.jit.trace(mod, inputs)
        test_inputs = (torch.randint(0, 9, (9, 9)), torch.tensor(8))
        eager_out = mod(*test_inputs)
        traced_out = traced_func(*test_inputs)
        self.assertNotWarn(lambda : traced_func(*test_inputs), "Shouldn't throw slicing related warn here")
        self.assertEqual(eager_out, traced_out)
        test_inputs = (torch.randint(0, 50, (50, 50)), torch.tensor(12))
        eager_out = mod(*test_inputs)
        traced_out = traced_func(*test_inputs)
        self.assertNotWarn(lambda : traced_func(*test_inputs), "Shouldn't throw slicing related warn here")
        self.assertEqual(eager_out, traced_out)

    def test_typeas_trace_check(self):
        if False:
            i = 10
            return i + 15
        a = torch.tensor([0.4], requires_grad=True)
        b = torch.tensor([0.7], requires_grad=True)

        def f(x, y):
            if False:
                i = 10
                return i + 15
            return x.type_as(y)
        trace = torch.jit.trace(f, (a, b))

    def test_trace_index(self):
        if False:
            print('Hello World!')
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0], dtype=torch.int64)

        def fn(x, y):
            if False:
                return 10
            return x[y]
        fn_traced = torch.jit.trace(fn, (x, y))
        self.assertEqual(fn(x, y), fn_traced(x, y))

    def test_trace_index_constant(self):
        if False:
            i = 10
            return i + 15
        x = torch.tensor([0.4], requires_grad=True)

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x[0]

        def run(f):
            if False:
                return 10
            y = f(x)
            grad = torch.autograd.grad(y, x)[0].clone()
            return (y, grad)
        traced_fn = torch.jit.trace(fn, torch.ones(1))
        self.assertEqual(run(fn), run(traced_fn))

    def test_index_put(self):
        if False:
            return 10
        ten = torch.zeros(3, 3)
        mask = torch.tensor([[True, True, True], [True, False, False], [True, True, False]])

        def test_fn(ten, mask):
            if False:
                for i in range(10):
                    print('nop')
            ten[mask] = torch.ones(6)
            return ten
        traced_test_fn = torch.jit.trace(test_fn, (ten, mask))
        ten = torch.rand(3, 3)
        self.assertEqual(test_fn(ten, mask), traced_test_fn(ten, mask))

    def test_canonicalize_tensor_iterator(self):
        if False:
            print('Hello World!')
        x = torch.randn(4, 4)

        def f(x):
            if False:
                return 10
            x = x + 2
            x = x - 4
            x = x * 6
            x = x / 8
            return x
        traced = torch.jit.trace(f, (x,))
        f(x)
        graph = traced.graph_for(x)
        self.assertTrue(str(traced.graph_for(x)).count(': int = prim::Constant') == 5)

    @suppress_warnings
    def test_constant(self):
        if False:
            i = 10
            return i + 15
        x = torch.randn(2, 2, requires_grad=True)

        def f(x):
            if False:
                print('Hello World!')
            return x.matmul(torch.diag(torch.tensor([2.0, 2.0])))
        self.checkTrace(f, (x,), (torch.ones(2, 2, requires_grad=True),))

    def test_wrapped_number(self):
        if False:
            while True:
                i = 10

        def foobar():
            if False:
                for i in range(10):
                    print('nop')
            x = -10000.0
            result = x * torch.ones(1, dtype=torch.float)
            return result
        scripted = torch.jit.trace(foobar, (), check_trace=True)

    def test_inplace_transplant(self):
        if False:
            print('Hello World!')
        x = torch.tensor([0.0], requires_grad=True)

        def fn(x):
            if False:
                print('Hello World!')
            y = x.clone()
            y.add_(2)
            y.add_(3)
            return y
        (g, _) = torch.jit._get_trace_graph(fn, (x,))
        self.run_pass('dce', g)
        FileCheck().check_count('aten::clone', 1, exactly=True).check_count('aten::add_', 2, exactly=True).check_next('return').run(str(g))
        self.assertExportImport(g, (x,))

    def test_inplace_flags(self):
        if False:
            for i in range(10):
                print('nop')

        class InplaceFn(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                ctx.mark_dirty(x)
                return x.add_(1)

            @staticmethod
            def backward(ctx, go):
                if False:
                    i = 10
                    return i + 15
                return go

        class RegularFn(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                return x.add(1)

            @staticmethod
            def backward(ctx, go):
                if False:
                    i = 10
                    return i + 15
                return go
        x = torch.tensor([0.0], requires_grad=True)

        def fn(x):
            if False:
                return 10
            y = RegularFn.apply(x)
            y = InplaceFn.apply(y)
            y = InplaceFn.apply(y)
            y = RegularFn.apply(y)
            return y
        (trace_graph, _) = torch.jit._get_trace_graph(fn, (x,), _force_outplace=True)
        self.run_pass('dce', trace_graph)
        ops = list(trace_graph.nodes())
        for op in ops:
            self.assertTrue(op.hasAttribute('inplace'))
        inplace_flags = [False, True, True, False]
        for (op, is_inplace) in zip(ops, inplace_flags):
            self.assertEqual(op.i('inplace'), is_inplace)

    def test_inplace_check(self):
        if False:
            for i in range(10):
                print('nop')

        class MyInplaceFn(Function):

            @staticmethod
            def forward(self, x):
                if False:
                    return 10
                x.add_(1)
                self.mark_dirty(x)
                return x

            @staticmethod
            def backward(self, grad):
                if False:
                    for i in range(10):
                        print('nop')
                return grad

        def fn(x):
            if False:
                return 10
            return MyInplaceFn.apply(x)
        x = torch.randn(5, 5)
        ge = torch.jit.trace(fn, (x,), _force_outplace=True, check_trace=False)
        with self.assertRaisesRegex(RuntimeError, 'inplace MyInplaceFn'):
            ge(x)

    def test_force_outplace_check_fill(self):
        if False:
            return 10

        def f(x):
            if False:
                return 10
            return torch.empty(x.shape).fill_(7)
        x = torch.randn(10, 15)
        ft = torch.jit.trace(f, x, _force_outplace=True)
        self.assertEqual(f(x), ft(x))

    def test_force_outplace_check_zero(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.empty(x.shape).zero_()
        x = torch.randn(10, 15)
        ft = torch.jit.trace(f, x, _force_outplace=True)
        self.assertEqual(f(x), ft(x))

    def do_trace_size(self, requires_grad):
        if False:
            return 10

        def fn(x):
            if False:
                while True:
                    i = 10
            return x.view(x.shape[1] * 2, x.size(0), 2)
        x = torch.randn(5, 2, 4, requires_grad=requires_grad)
        y = torch.randn(4, 8, 4, requires_grad=requires_grad)
        traced_fn = torch.jit.trace(fn, x)
        self.assertEqual(traced_fn(y), fn(y))
        self.assertEqual(traced_fn(x), fn(x))

    def test_trace_size(self):
        if False:
            return 10
        self.do_trace_size(False)

    def test_trace_size_with_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.do_trace_size(True)

    def test_trace_numel(self):
        if False:
            while True:
                i = 10

        def fn(x):
            if False:
                print('Hello World!')
            return x.numel()
        x = torch.randn(2, 3, 4)
        y = torch.randn(4, 5, 6)
        traced_fn = torch.jit.trace(fn, x)
        self.assertEqual(traced_fn(y), fn(y))
        self.assertEqual(traced_fn(x), fn(x))

    def do_trace_arange(self, requires_grad):
        if False:
            i = 10
            return i + 15

        def arange(x):
            if False:
                print('Hello World!')
            return torch.arange(x.shape[0])

        def arange_scalar(x):
            if False:
                i = 10
                return i + 15
            return torch.arange(12)

        def arange_start_end(x):
            if False:
                i = 10
                return i + 15
            return torch.arange(start=x.shape[0], end=x.shape[0] + 5)
        x = torch.randn(5, 3, 2, requires_grad=requires_grad)
        y = torch.randn(8, 2, 4, requires_grad=requires_grad)
        traced_arange = torch.jit.trace(arange, x)
        self.assertEqual(traced_arange(y), arange(y))
        self.assertEqual(traced_arange(x), arange(x))
        traced_arange_scalar = torch.jit.trace(arange_scalar, x)
        self.assertEqual(traced_arange_scalar(y), arange_scalar(y))
        self.assertEqual(traced_arange_scalar(x), arange_scalar(x))
        traced_arange_start_end = torch.jit.trace(arange_start_end, x)
        self.assertEqual(traced_arange_start_end(y), arange_start_end(y))
        self.assertEqual(traced_arange_start_end(x), arange_start_end(x))

    def test_trace_arange(self):
        if False:
            i = 10
            return i + 15
        self.do_trace_arange(False)

    def test_trace_arange_with_grad(self):
        if False:
            while True:
                i = 10
        self.do_trace_arange(True)

    def test_trace_full_dynamic_shape(self):
        if False:
            return 10

        def full_with_shape_like(x):
            if False:
                print('Hello World!')
            return torch.full(x.shape, 2.0)
        x = torch.randn(3, 4)
        ge = torch.jit.trace(full_with_shape_like, example_inputs=x)
        y = torch.randn(2, 7)
        self.assertEqual(ge(y).shape, y.shape)
        self.assertEqual(ge(x).shape, x.shape)

    def test_trace_slice_setitem_dynamic_shape(self):
        if False:
            while True:
                i = 10

        def slice_setitem(x, y):
            if False:
                i = 10
                return i + 15
            x[:, 2] = y + 1
            return x
        x = torch.randn(3, 4)
        traced = torch.jit.trace(slice_setitem, (x, x[:, 0]))
        x = torch.randn(10, 5)
        self.assertEqual(traced(x.clone(), x[:, 0]), slice_setitem(x.clone(), x[:, 0]))

    @suppress_warnings
    def do_trace_slice(self, requires_grad):
        if False:
            i = 10
            return i + 15

        def slice(x):
            if False:
                return 10
            results = []
            for i in range(4):
                results.append(x[:x.size(0) - i, i:x.size(2), i:3])
            return tuple(results)

        def slice_select(x):
            if False:
                i = 10
                return i + 15
            results = []
            for i in range(4):
                results.append(x[:, i:, x.size(2) - 5])
            return tuple(results)
        x = torch.randn(5, 6, 7, requires_grad=requires_grad)
        y = torch.randn(7, 8, 9, requires_grad=requires_grad)
        traced_slice = torch.jit.trace(slice, x)
        self.assertEqual(traced_slice(y), slice(y))
        self.assertEqual(traced_slice(x), slice(x))
        traced_slice_select = torch.jit.trace(slice_select, x)
        self.assertEqual(traced_slice_select(y), slice_select(y))
        self.assertEqual(traced_slice_select(x), slice_select(x))

    def test_trace_slice(self):
        if False:
            print('Hello World!')
        self.do_trace_slice(False)

    def test_trace_slice_with_grad(self):
        if False:
            return 10
        self.do_trace_slice(True)

    def test_trace_casts(self):
        if False:
            while True:
                i = 10
        casts = [lambda x: x.byte(), lambda x: x.float(), lambda x: x.cpu(), lambda x: x.to(device='cpu'), lambda x: x.to(dtype=torch.int64), lambda x: x.to(device='cpu', dtype=torch.float), lambda x: x.to(x)]

        def assertContainsCast(trace):
            if False:
                i = 10
                return i + 15
            self.assertEqual(sum((n.kind() == 'aten::to' for n in trace.graph.nodes())), 1)
        for cast in casts:
            trace = torch.jit.trace(cast, torch.randn(2, 2))
            assertContainsCast(trace)
            x = torch.randn(2, 2)
            self.assertEqual(trace(x), cast(x))

        def to_tensor(x, y):
            if False:
                while True:
                    i = 10
            return x.to(y)
        to_tensor_trace = torch.jit.trace(to_tensor, (torch.randn(2, 2), torch.randn(1, 8)))
        assertContainsCast(to_tensor_trace)
        (x, y) = (torch.randn(2, 2), torch.randn(1, 10))
        self.assertEqual(to_tensor_trace(x, y), to_tensor(x, y))

    @skipIfCompiledWithoutNumpy
    @skipIfCrossRef
    def test_trace_warn(self):
        if False:
            i = 10
            return i + 15

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            int(x)
            y = x * 1
            if y:
                pass
            q = [x, x * 4]
            z = q[y]
            float(z)
            z.tolist()
            z.numpy()
            for _ in torch.ones(4, 4):
                pass
            return z + 4
        with warnings.catch_warnings(record=True) as warns:
            traced_fn = torch.jit.trace(fn, torch.tensor([1]))
        for warn in warns:
            self.assertIs(warn.category, torch.jit.TracerWarning)
        warns = [str(w.message) for w in warns]
        self.assertIn('a Python integer', warns[0])
        self.assertIn('a Python boolean', warns[1])
        self.assertIn('a Python float', warns[2])
        self.assertIn('a Python list', warns[3])
        self.assertIn('a NumPy array', warns[4])
        self.assertIn('Iterating over', warns[5])

    def test_trace_tuple(self):
        if False:
            print('Hello World!')

        def fn(x, y):
            if False:
                i = 10
                return i + 15
            return (x, (x * y[1], x * y[0]))
        (x, y) = (torch.randn(2, 2), (torch.ones(2, 2), torch.randn(2, 2)))
        traced_fn = torch.jit.trace(fn, (x, y))
        self.assertEqual(traced_fn(x, y), fn(x, y))
        FileCheck().check_count('prim::TupleConstruct', 2, exactly=True).check_next('return').run(str(traced_fn.graph))
        self.assertExportImport(traced_fn.graph, (x, y))

    def test_trace_random(self):
        if False:
            for i in range(10):
                print('nop')

        def f(mean, std):
            if False:
                i = 10
                return i + 15
            return torch.normal(mean, std)
        traced = torch.jit.trace(f, (torch.zeros(2, 3), torch.ones(2, 3)), check_trace=False)
        (mean, std) = (torch.zeros(5, 5), torch.ones(5, 5))
        with torch.random.fork_rng(devices=[]):
            output = f(mean, std)
        traced_output = traced(mean, std)
        self.assertEqual(output, traced_output)

    def test_trace_tensor_factory(self):
        if False:
            print('Hello World!')

        def run(**kwargs):
            if False:
                print('Hello World!')
            inputs_require_grads = kwargs.pop('inputs_require_grads', True)

            def fn(x):
                if False:
                    for i in range(10):
                        print('nop')
                return x + torch.ones(2, 3, **kwargs)
            input_kwargs = kwargs.copy()
            if 'out' in input_kwargs:
                del input_kwargs['out']
            input = torch.ones(2, 3, **input_kwargs)
            self.checkTrace(fn, (input,), inputs_require_grads=inputs_require_grads)
            tfn = torch.jit.trace(fn, input)
            self.assertTrue('ones' in str(tfn.graph))
        run()
        run(dtype=torch.int, inputs_require_grads=False)
        run(out=torch.tensor([]))
        if RUN_CUDA:
            run(device='cuda:0')
        if RUN_CUDA_MULTI_GPU:
            run(device='cuda:1')

    def test_trace_indexed_assignment(self):
        if False:
            while True:
                i = 10

        def stuff(x, y):
            if False:
                return 10
            x = x.clone()
            x[0] = y
            return x
        example = torch.rand(3, 4)
        self.checkTrace(stuff, (example, example[0] + 1))

    @unittest.expectedFailure
    def test_output_unflatten(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that outputs of traced functions retain the original structure and nesting'

        def fn(x):
            if False:
                while True:
                    i = 10
            return (x * 2, (x ** 2, x + 4, (x + 2,)), x * 4)
        self.checkTrace(fn, (torch.randn(2, 2),))

    def test_input_flatten(self):
        if False:
            i = 10
            return i + 15
        'Check that inputs to traced functions are flattened'

        def fn(x, t):
            if False:
                print('Hello World!')
            (y, z) = t
            return x * y * z
        inputs = (torch.randn(1), (torch.randn(1), torch.randn(1)))
        self.checkTrace(fn, inputs)

    def test_input_dict_empty(self):
        if False:
            print('Hello World!')

        def test(d):
            if False:
                while True:
                    i = 10
            pass
        with self.assertRaises(RuntimeError):
            self.checkTrace(test, {})

    def test_input_dict_remembers_keys(self):
        if False:
            i = 10
            return i + 15
        'Check that the trace remembers which keys were in a dict input'

        class TestModule(torch.nn.Module):

            def forward(self, dict_input):
                if False:
                    i = 10
                    return i + 15
                return dict_input['x']
        input_1 = {'x': torch.tensor(1)}
        m = TestModule()
        m_traced = torch.jit.trace(m, (input_1,))
        self.assertEqual(m_traced(input_1), torch.tensor(1))
        input_same_key_different_value = {'x': torch.tensor(2)}
        self.assertEqual(m_traced(input_same_key_different_value), torch.tensor(2))
        input_different_key = {'y': torch.tensor(3)}
        with self.assertRaises(RuntimeError):
            m_traced(input_different_key)
        input_additional_key = {'x': torch.tensor(4), 'y': torch.tensor(3)}
        self.assertEqual(m_traced(input_additional_key), torch.tensor(4))

    def test_input_dict_insertion_order(self):
        if False:
            while True:
                i = 10
        "Check that dictionary access doesn't care about insertion order"

        class TestModule(torch.nn.Module):

            def forward(self, dict_input):
                if False:
                    print('Hello World!')
                return (dict_input['x'], dict_input['y'])
        input_x_then_y = {}
        input_x_then_y['x'] = torch.tensor(1)
        input_x_then_y['y'] = torch.tensor(2)
        m = TestModule()
        m_traced = torch.jit.trace(m, (input_x_then_y,))
        self.assertEqual(m_traced(input_x_then_y), (torch.tensor(1), torch.tensor(2)))
        input_y_then_x = {}
        input_y_then_x['y'] = torch.tensor(4)
        input_y_then_x['x'] = torch.tensor(3)
        self.assertEqual(m_traced(input_y_then_x), (torch.tensor(3), torch.tensor(4)))

    def test_input_dict_recursive(self):
        if False:
            while True:
                i = 10

        class TestModule(torch.nn.Module):

            def forward(self, dict_input):
                if False:
                    return 10
                return dict_input['x'][1]
        input_1 = {'x': {1: torch.tensor(1)}}
        m = TestModule()
        m_traced = torch.jit.trace(m, (input_1,))
        input_2 = {'x': {1: torch.tensor(2)}}
        self.assertEqual(m_traced(input_2), torch.tensor(2))

    def test_input_dict_checkTrace_mut(self):
        if False:
            while True:
                i = 10

        def test(d):
            if False:
                i = 10
                return i + 15
            d['x'].tanh_()
            return d['x']
        inputs = {'x': torch.rand(3, 4), 'y': torch.rand(3, 4)}
        self.checkTrace(test, (inputs,), inputs_require_grads=False)

    def test_input_dict_unify(self):
        if False:
            while True:
                i = 10

        def test(d):
            if False:
                while True:
                    i = 10
            return (d['int'], d['float'])
        inputs = {'int': torch.ones((2, 2), dtype=torch.int32), 'float': torch.ones((2, 2), dtype=torch.float32)}
        self.checkTrace(test, (inputs,), inputs_require_grads=False)

    def test_input_tuple_of_dicts(self):
        if False:
            while True:
                i = 10

        def test(t):
            if False:
                while True:
                    i = 10
            d = t[0]
            return d['x']['y']
        inputs = {'x': {'y': torch.rand(2, 3)}}
        self.checkTrace(test, ((inputs, inputs),), allow_unused=True)

    def test_input_dict_of_dicts(self):
        if False:
            return 10

        def test(d):
            if False:
                return 10
            return d['x']['y']
        nested_input = {'y': torch.rand(2, 3)}
        unified_nested = {'y': torch.rand(3, 2)}
        inputs = {'x': nested_input, 'force_unify': unified_nested}
        self.checkTrace(test, (inputs,), allow_unused=True)

    def test_input_dict_of_lists(self):
        if False:
            return 10

        def test(d):
            if False:
                print('Hello World!')
            return d['x'][0]
        inputs = {'x': [torch.rand(3, 2)]}
        self.checkTrace(test, (inputs,))

    def test_input_list_toplevel_flatten(self):
        if False:
            while True:
                i = 10

        def test(t1, t2):
            if False:
                return 10
            return torch.add(t1, t2)
        inputs = [torch.ones(2, 2), torch.rand(2, 2)]
        self.checkTrace(test, inputs)

    def test_input_list_toplevel_flatten_direct(self):
        if False:
            return 10

        class Test(torch.nn.Module):

            def forward(self, t1, t2):
                if False:
                    return 10
                return torch.add(t1, t2)
        inputs = [torch.ones(2, 2), torch.rand(2, 2)]
        torch.jit.trace(Test(), inputs)

    def test_input_list_of_tuples(self):
        if False:
            i = 10
            return i + 15

        def test(l):
            if False:
                print('Hello World!')
            return l[0][0]
        inputs = [(torch.ones(2, 2),)]
        self.checkTrace(test, (inputs,))

    def test_input_dict_empty_list(self):
        if False:
            print('Hello World!')

        def test(d):
            if False:
                return 10
            pass
        inputs = {1: []}
        with self.assertRaisesRegex(RuntimeError, 'List trace'):
            self.checkTrace(test, (inputs,))

    def test_input_list_mixed_type(self):
        if False:
            i = 10
            return i + 15

        def test(d):
            if False:
                return 10
            pass
        inputs = [torch.rand(2, 3), (torch.ones(2), torch.ones(2))]
        with self.assertRaisesRegex(RuntimeError, 'consistent'):
            self.checkTrace(test, (inputs,))

    def test_conv(self):
        if False:
            while True:
                i = 10
        x = torch.ones(20, 16, 50, 40)
        (g, outputs, inputs) = torch.jit._get_trace_graph(nn.Conv2d(16, 13, 3, bias=False), x, return_inputs=True)
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))

    def test_max_pool(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.rand(20, 16, 10, 10)

        def max_pool2d(x):
            if False:
                return 10
            return F.max_pool2d(x, 2) + 2
        trace = torch.jit.trace(max_pool2d, x)
        graph = trace.graph_for(x)
        FileCheck().check('aten::max_pool2d(').run(graph)
        self.assertEqual(max_pool2d(x), trace(x))

    def test_nested_inplace(self):
        if False:
            while True:
                i = 10
        x = torch.randn(2, 2)
        (g, outputs, inputs) = torch.jit._get_trace_graph(lambda x: F.threshold(x, 0, 0, inplace=True), (x,), return_inputs=True)
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))
        FileCheck().check('threshold_').run(str(g))
        self.assertExportImport(g, (x,))

    def test_repeated_input(self):
        if False:
            while True:
                i = 10

        def fn(a, b):
            if False:
                while True:
                    i = 10
            return a + b
        ge = self.checkTrace(fn, [torch.randn(2, 2)] * 2)
        inputs = set(ge.graph.inputs())
        self.assertTrue(len(inputs) == 3)

    def test_repeated_output(self):
        if False:
            print('Hello World!')

        def fn(a, b):
            if False:
                i = 10
                return i + 15
            z = a + b
            return (z, z)
        ge = self.checkTrace(fn, [torch.randn(2, 2) for _ in range(2)])
        tuple_output = list(ge.graph.outputs())[0]
        tuple_inputs = list(tuple_output.node().inputs())
        self.assertTrue(tuple_inputs[0] == tuple_inputs[1])

    def test_inplace_copy(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(4, 4, requires_grad=True)

        def f(x):
            if False:
                i = 10
                return i + 15
            out = torch.zeros(x.size())
            out.copy_(x)
            return out
        (g, outputs, inputs) = torch.jit._get_trace_graph(f, (x,), return_inputs=True)
        self.run_pass('dce', g)
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))
        self.assertExportImport(g, (x,))

    def test_inplace_copy_force_outplace(self):
        if False:
            while True:
                i = 10
        x = torch.randn(4, 4, requires_grad=True)

        def f(x):
            if False:
                while True:
                    i = 10
            out = torch.zeros(x.size())
            out.copy_(x)
            return out
        (g, outputs, inputs) = torch.jit._get_trace_graph(f, (x,), return_inputs=True, _force_outplace=True)
        self.run_pass('dce', g)
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))
        self.assertExportImport(g, (x,))
        FileCheck().check('expand_as').run(str(g))

    def test_shared_param(self):
        if False:
            print('Hello World!')

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.b = self.a = nn.Parameter(torch.randn(2, 2))

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x * self.a + self.b
        m = MyModule()
        (g, _) = torch.jit._get_trace_graph(m, (torch.randn(2, 2),))
        self.run_pass('dce', g)
        self.assertEqual(len(list(g.inputs())), 2)
        FileCheck().check('mul').check('add').run(str(g))

    def test_trace_c10_ops(self):
        if False:
            i = 10
            return i + 15
        try:
            _ = torch.ops._caffe2.GenerateProposals
        except AttributeError:
            self.skipTest('Skip the test since c2 ops are not registered.')

        class MyModel(torch.nn.Module):

            def forward(self, scores, bbox_deltas, im_info, anchors):
                if False:
                    while True:
                        i = 10
                (a, b) = torch.ops._caffe2.GenerateProposals(scores, bbox_deltas, im_info, anchors, 2.0, 6000, 300, 0.7, 16, True, -90, 90, 1.0, True)
                return (a, b)
        model = MyModel()
        A = 4
        H = 10
        W = 8
        img_count = 3
        scores = torch.ones(img_count, A, H, W, dtype=torch.float32)
        bbox_deltas = torch.linspace(0, 10, steps=img_count * 4 * A * H * W, dtype=torch.float32)
        bbox_deltas = bbox_deltas.view(img_count, 4 * A, H, W)
        im_info = torch.ones(img_count, 3, dtype=torch.float32)
        anchors = torch.ones(A, 4, dtype=torch.float32)
        inputs = (scores, bbox_deltas, im_info, anchors)
        traced_model = torch.jit.trace(model, inputs)
        self.assertEqual(traced_model(*inputs), model(*inputs))
        self.assertExportImportModule(traced_model, (scores, bbox_deltas, im_info, anchors))

    def run_ge_tests(self, optimize, use_cuda):
        if False:
            i = 10
            return i + 15
        with enable_profiling_mode_for_profiling_tests():
            with torch.jit.optimized_execution(optimize):

                def rand(*args):
                    if False:
                        return 10
                    t = torch.rand(*args).float()
                    if use_cuda:
                        t = t.cuda()
                    return t
                self.checkTrace(lambda a, b: a * b + b, [rand(1), rand(1)], [rand(2, 3), rand(2, 3)])
                self.checkTrace(lambda a, b: (b, a), [rand(1), rand(1)])

                def foo(a):
                    if False:
                        for i in range(10):
                            print('nop')
                    t = a * a
                    return (t * t, 4 * t)
                self.checkTrace(foo, [rand(1)])
                self.checkTrace(lambda a, b: a * a, [rand(1), rand(1)], allow_unused=True)
                self.checkTrace(foo, [rand(1)], drop=1)
                self.checkTrace(lambda a, b: a * b / (a - 2 * b) + b, [rand(1), rand(1)])

    def test_ge_unoptimized(self):
        if False:
            i = 10
            return i + 15
        self.run_ge_tests(False, False)

    @unittest.skipIf(IS_SANDCASTLE, 'NYI: fuser support for Sandcastle')
    @enable_cpu_fuser
    def test_ge_optimized(self):
        if False:
            return 10
        with enable_profiling_mode_for_profiling_tests():
            self.run_ge_tests(True, False)

    @unittest.skipIf(not RUN_CUDA, 'requires CUDA')
    def test_ge_cuda(self):
        if False:
            print('Hello World!')
        self.run_ge_tests(True, True)

    def test_ge(self):
        if False:
            while True:
                i = 10

        def foo(a, b):
            if False:
                return 10
            return a * b / (a - b) + b
        V = Variable
        (a, b) = (V(torch.rand(1)), V(torch.rand(1)))
        ge = torch.jit.trace(foo, (a, b))
        (a, b) = (V(torch.rand(1), requires_grad=True), V(torch.rand(1), requires_grad=True))
        (r,) = ge(a, b)
        (da, db) = torch.autograd.grad(r + 3, [a, b], create_graph=True)
        l2 = da * db + db * db
        g2result = torch.autograd.grad(l2, [da, db])
        r = foo(a, b)
        (da2, db2) = torch.autograd.grad(r + 3, [a, b], create_graph=True)
        self.assertEqual(da, da2)
        self.assertEqual(db, db2)
        l3 = da2 * db2 + db2 * db2
        g2result2 = torch.autograd.grad(l3, [da2, db2])
        self.assertEqual(g2result, g2result2)

    def test_trace_annotation(self):
        if False:
            return 10

        @_trace(torch.rand(1))
        def foo(a):
            if False:
                while True:
                    i = 10
            return a + a + a
        x = torch.randn(5, 5)
        self.assertEqual(foo(x), x + x + x)

    @unittest.skipIf(not RUN_CUDA, 'calls .cuda()')
    @with_tf32_off
    def test_traced_module_cuda(self):
        if False:
            while True:
                i = 10

        class Model(nn.Module):

            def __init__(self, num_features, num_layers):
                if False:
                    print('Hello World!')
                super().__init__()
                self.num_layers = num_layers
                layers = [[nn.Linear(num_features, num_features), nn.Sigmoid()] for _ in range(num_layers)]
                self.submodule = nn.Sequential(*chain(*layers))

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                for i in range(self.num_layers):
                    x = self.submodule[i](x) + x
                return x
        model = Model(5, 3)
        x = torch.randn(2, 5)
        traced_model = torch.jit.trace(model, x)
        model.__repr__()
        linear_submodule = next(iter(traced_model.submodule._modules.values()))
        with self.assertRaises(AttributeError):
            linear_submodule.in_features
        linear_submodule.weight
        linear_submodule.weight = nn.Parameter(torch.randn(linear_submodule.weight.shape))
        with self.assertRaises(RuntimeError):
            del linear_submodule.weight
        with self.assertRaises(RuntimeError):
            linear_submodule(x)
        linear_submodule.cuda()
        traced_model.float().cuda()
        cuda_out = traced_model(x.float().cuda())
        traced_model.cpu()
        cpu_out = traced_model(x.float())
        self.assertEqual(cpu_out, cuda_out)
        traced_model.to('cuda')
        cuda_out = traced_model(x.float().cuda())
        traced_model.to('cpu')
        cpu_out = traced_model(x.float())
        self.assertEqual(cpu_out, cuda_out)
        traced_model.to(torch.get_default_dtype())
        state = {k: v.clone() for (k, v) in traced_model.state_dict().items()}
        new_state = {k: v.clone().fill_(1) for (k, v) in state.items()}
        out = traced_model(x)
        traced_model.load_state_dict(new_state)
        out_ones = traced_model(x)
        traced_model.load_state_dict(state)
        out_state = traced_model(x)
        self.assertEqual(out, out_state)
        self.assertNotEqual(out, out_ones)

    @unittest.skipIf(not RUN_CUDA, 'uses cuda')
    def test_type_same_device(self):
        if False:
            return 10

        class Model(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.dtype = torch.float16

            def forward(self, x=None):
                if False:
                    i = 10
                    return i + 15
                h = x.type(self.dtype)
                return h
        a = Model()
        b = torch.jit.trace(a, example_inputs=(torch.ones([1], device=torch.device('cuda')),))
        FileCheck().check_not('device').run(b.code)

    def test_export_no_reorder(self):
        if False:
            for i in range(10):
                print('nop')

        def func(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return a * b / (a - 2 * b) + b
        recording_inputs = [torch.tensor([0.5561978816986084], dtype=torch.float32, requires_grad=True), torch.tensor([0.25947844982147217], dtype=torch.float32, requires_grad=True)]
        ge1 = torch.jit.trace(func, recording_inputs)
        ge2 = self.getExportImportCopy(ge1)
        outputs_ge1 = ge1(*recording_inputs)
        outputs_ge2 = ge2(*recording_inputs)
        grad_ge1 = torch.autograd.grad(outputs_ge1, recording_inputs)
        grad_ge2 = torch.autograd.grad(outputs_ge2, recording_inputs)
        self.assertTrue(outputs_ge1 == outputs_ge2)
        self.assertTrue(grad_ge1 == grad_ge2)

    def test_python_function(self):
        if False:
            for i in range(10):
                print('nop')

        class MyFn(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    return 10
                return x + 1

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    while True:
                        i = 10
                return grad_output

        @_trace(torch.zeros(2))
        def fn(x):
            if False:
                while True:
                    i = 10
            return MyFn.apply(x + 2) + 3
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.randn(2, 2, requires_grad=True)
        fn(x)
        fn(y)

    def test_python_function_tup(self):
        if False:
            return 10

        class MyFn(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    i = 10
                    return i + 15
                return (x + 1, x - 1)

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    print('Hello World!')
                return (grad_output, grad_output)

        @_trace(torch.zeros(2))
        def fn(x):
            if False:
                print('Hello World!')
            (a, b) = MyFn.apply(x + 2)
            return a + b + 3
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.randn(2, 2, requires_grad=True)
        fn(x)
        fn(y)

    def test_trace_detach(self):
        if False:
            i = 10
            return i + 15

        def foo(x, w):
            if False:
                return 10
            return torch.matmul(x, w).detach()
        traced = torch.jit.trace(foo, (torch.rand(3, 4), torch.rand(4, 5)))
        FileCheck().check('matmul').check('detach').run(str(traced.graph))
        (x, w) = (torch.rand(3, 4), torch.rand(4, 5, requires_grad=True))
        traced_result = traced(x, w)
        self.assertEqual(foo(x, w), traced_result)
        self.assertFalse(traced_result.requires_grad)
        self.assertIsNone(traced_result.grad_fn)

    def test_trace_detach_redispatch(self):
        if False:
            return 10

        def foo(x, w):
            if False:
                print('Hello World!')
            y = torch.matmul(x, w)
            assert y.requires_grad
            y = y.detach()
            assert not y.requires_grad
            return y
        (x, w) = (torch.rand(3, 4), torch.rand(4, 5, requires_grad=True))
        torch.jit.trace(foo, (x, w), check_trace=False)

    def test_trace_detach_inplace(self):
        if False:
            print('Hello World!')

        def foo(x, w):
            if False:
                while True:
                    i = 10
            y = torch.matmul(x, w)
            y.detach_()
            return y
        traced = torch.jit.trace(foo, (torch.rand(3, 4), torch.rand(4, 5)))
        FileCheck().check('matmul').check('detach(').run(str(traced.graph))
        (x, w) = (torch.rand(3, 4), torch.rand(4, 5, requires_grad=True))
        traced_result = traced(x, w)
        self.assertEqual(foo(x, w), traced_result)
        self.assertFalse(traced_result.requires_grad)
        self.assertIsNone(traced_result.grad_fn)

    def test_trace_detach_inplace_redispatch(self):
        if False:
            while True:
                i = 10

        def foo(x, w):
            if False:
                return 10
            y = torch.matmul(x, w)
            assert y.requires_grad
            y.detach_()
            assert not y.requires_grad
            return y
        (x, w) = (torch.rand(3, 4), torch.rand(4, 5, requires_grad=True))
        torch.jit.trace(foo, (x, w), check_trace=False)

    def test_trace_slice_full_dim(self):
        if False:
            while True:
                i = 10

        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return x[0:5, 0] + 1.0
        traced = torch.jit.trace(foo, (torch.rand(5, 4),))
        test_x = torch.rand(6, 3)
        self.assertEqual(foo(test_x), traced(test_x))

    def test_trace_dict_input(self):
        if False:
            for i in range(10):
                print('nop')

        class Bar(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.foo = Foo()

            def forward(self, a, b):
                if False:
                    for i in range(10):
                        print('nop')
                return self.foo({'a': a, 'b': b})['a']

        class Foo(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return {'a': x['a'] * x['b']}
        x = (torch.rand(3), torch.rand(3))
        model = Bar()
        self.checkTrace(model, x)

    def test_trace_dict_output(self):
        if False:
            i = 10
            return i + 15

        class TraceDictStrTensor(torch.nn.Module):

            def forward(self, a, b):
                if False:
                    return 10
                return {'a': a, 'b': b}

        class TraceDictTensorTensor(torch.nn.Module):

            def forward(self, a, b):
                if False:
                    while True:
                        i = 10
                return {a: b, b: a}
        x = (torch.rand(3), torch.rand(3))
        with self.assertRaisesRegex(RuntimeError, 'Encountering a dict at the output'):
            torch.jit.trace(TraceDictStrTensor(), x)
        traced_dict_str_mod = torch.jit.trace(TraceDictStrTensor(), x, strict=False)
        self.assertEqual(traced_dict_str_mod(*x), {'a': x[0], 'b': x[1]})
        traced_dict_tensor_mod = torch.jit.trace(TraceDictTensorTensor(), x, strict=False)
        self.assertEqual(traced_dict_tensor_mod(*x), {x[0]: x[1], x[1]: x[0]})

    def test_trace_with_tensor_list_output(self):
        if False:
            while True:
                i = 10

        def f():
            if False:
                print('Hello World!')
            return [torch.zeros(1), torch.zeros(5)]
        with self.assertWarnsRegex(torch.jit.TracerWarning, 'cause the trace to be incorrect'):
            torch.jit.trace(f, [])
        traced_non_strict_f = torch.jit.trace(f, [], strict=False)
        self.assertEqual(traced_non_strict_f(), f())

    def test_trace_with_number_list_output(self):
        if False:
            return 10

        def f():
            if False:
                while True:
                    i = 10
            return [1, 5]
        with self.assertRaisesRegex(RuntimeError, 'Only tensors.+can be output from traced functions'):
            traced_f = torch.jit.trace(f, [])

    def test_trace_with_nested_tensor_list_output(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                while True:
                    i = 10
            return [[torch.zeros(1)], [torch.zeros(5)]]
        with self.assertRaisesRegex(RuntimeError, 'Only tensors.+can be output from traced functions'):
            traced_f = torch.jit.trace(f, [])

    def test_trace_variable_instantiation(self):
        if False:
            for i in range(10):
                print('nop')

        def random_foo(x):
            if False:
                i = 10
                return i + 15
            return Variable(Variable(x) + 1.0)
        random_foo_traced = torch.jit.trace(random_foo, (torch.rand(3, 4),))
        x = torch.rand(5, 6)
        self.assertEqual(random_foo(x), random_foo_traced(x))

    def test_trace_slice_expr_complete_type(self):
        if False:
            while True:
                i = 10

        def random_foo(x):
            if False:
                while True:
                    i = 10
            return x + 1.0
        random_foo_traced = torch.jit.trace(random_foo, (torch.rand(3, 4),))

        @torch.jit.script
        def random_bar(x):
            if False:
                while True:
                    i = 10
            return random_foo_traced(x)[0:1]
        x = torch.rand(3, 4)
        self.assertEqual(random_bar(x), (x + 1)[0:1])

    def test_trace_inline_shape(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def tensor_size(x: torch.Tensor) -> torch.Tensor:
            if False:
                print('Hello World!')
            return torch.tensor([x.size()[0]])
        self.assertEqual(tensor_size(torch.rand(15)), torch.tensor([15]))
        traced_tensor_size = torch.jit.trace(tensor_size, torch.rand(7))
        self.assertEqual(traced_tensor_size(torch.rand(15)), torch.tensor([15]))

        @torch.jit.script
        def use_device(x):
            if False:
                return 10
            return torch.zeros_like(x, device=x.device)

        def foo(x):
            if False:
                while True:
                    i = 10
            return use_device(x)
        traced_tensor_size = torch.jit.trace(foo, torch.rand(7))
        self.run_pass('inline', traced_tensor_size.graph)
        FileCheck().check('prim::device').run(traced_tensor_size.graph)

    def test_trace_save(self):
        if False:
            while True:
                i = 10

        def fn(x):
            if False:
                i = 10
                return i + 15
            return x + 2

        def check(func):
            if False:
                print('Hello World!')
            with TemporaryFileName() as fname:
                func.save(fname)
                loaded = torch.jit.load(fname)
                input = torch.randn(2, 2)
                self.assertEqual(func(input), loaded(input))
        out = torch.jit.trace(fn, (torch.ones(2, 2),))
        check(out)

    def test_trace_optioanl_dtype(self):
        if False:
            return 10

        class Test(torch.nn.Module):

            def forward(self):
                if False:
                    return 10
                return torch.arange(5)
        traced = torch.jit.trace(Test(), ())
        torch.allclose(traced(), Test()())

    def test_trace_save_load_copy(self):
        if False:
            return 10

        class Test(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.conv(x)
        traced = torch.jit.trace(Test(), torch.rand(1, 3, 224, 224))
        buffer = io.BytesIO()
        torch.jit.save(traced, buffer)
        buffer.seek(0)
        loaded = torch.jit.load(buffer)
        copy.copy(loaded)
        copy.deepcopy(loaded)

    def test_trace_export_fns(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.a = 3

            @torch.jit.export
            def __getstate__(self):
                if False:
                    print('Hello World!')
                return (3, self.training)

            @torch.jit.export
            def __setstate__(self, state):
                if False:
                    return 10
                self.a = state[0]
                self.training = state[1]

            def forward(self, x):
                if False:
                    return 10
                return x + self.a
        f = Foo()
        traced = torch.jit.trace(f, (torch.rand(3, 4),))
        expected_names = ['__getstate__', '__setstate__']

        def check(mod):
            if False:
                for i in range(10):
                    print('nop')
            self.assertTrue(all((name in mod._c._method_names() for name in expected_names)))
        check(traced)
        imported = self.getExportImportCopy(traced)
        check(imported)

    def test_trace_export_fns_recursive(self):
        if False:
            return 10

        class Foo(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.a = 3

            @torch.jit.export
            def __getstate__(self):
                if False:
                    i = 10
                    return i + 15
                return (3, self.training)

            @torch.jit.export
            def __setstate__(self, state):
                if False:
                    print('Hello World!')
                self.a = state[0]
                self.training = state[1]

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x + self.a

        class Wrapper(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.foo = Foo()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.foo(x)
        f = Wrapper()
        traced = torch.jit.trace(f, (torch.rand(3, 4),))
        expected_names = ['__getstate__', '__setstate__']

        def check(mod):
            if False:
                for i in range(10):
                    print('nop')
            self.assertTrue(all((name in mod._c._method_names() for name in expected_names)))
        check(traced.foo)
        imported = self.getExportImportCopy(traced)
        check(imported.foo)

        class Bar(nn.Module):

            @torch.jit.export
            def addTwo(self, x):
                if False:
                    while True:
                        i = 10
                return x + 2

            def forward(self, input):
                if False:
                    return 10
                return (lambda a: a + 1)(input)

        class WrapperExports(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.bar = Bar()

            @torch.jit.export
            def addOne(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x + 1

            def forward(self, x):
                if False:
                    return 10
                return self.bar(x)
        f = WrapperExports()
        traced = torch.jit.trace(f, (torch.rand(3, 4),))
        expected_names = ['addOne']
        check(traced)

    def test_trace_autograd_function(self):
        if False:
            while True:
                i = 10

        class TestFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input):
                if False:
                    while True:
                        i = 10
                return torch.neg(input)

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.neg(grad_output)

        class TracedModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.relu(TestFunc.apply(x))

        class Wrapper(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.tm = TracedModule()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.tm(x)
        traced = torch.jit.trace(Wrapper(), (torch.rand(3, 4),))

    def test_trace_multi_output_function(self):
        if False:
            print('Hello World!')

        class Foo(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x, y):
                if False:
                    print('Hello World!')
                return (y, x)

            @staticmethod
            def backward(ctx, du, dv):
                if False:
                    return 10
                return (dv, du)

        class Bar(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                x = x.relu()
                y = y.relu()
                z = Foo.apply(x, y)
                return z
        x = torch.rand(3, 2, dtype=torch.double)
        y = torch.rand(1, 2, dtype=torch.double)
        traced = torch.jit.trace(Bar(), (x, y))
        print(traced.graph)
        schema = '(Double(1, 2, strides=[2, 1], requires_grad=0, device=cpu), Double(3, 2, strides=[2, 1], requires_grad=0, device=cpu)) = ^Foo'
        FileCheck().check(schema).run(traced.graph)
        (u, v) = traced(x, y)
        self.assertEqual(u, y)
        self.assertEqual(v, x)

    def test_interpolate_trace(self):
        if False:
            return 10

        class test(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)

            def forward(self, x):
                if False:
                    return 10
                y = self.conv(x)
                w = nn.functional.interpolate(y, mode='bilinear', align_corners=False, scale_factor=3)
                return w
        f = test()
        g = torch.jit.trace(f, (torch.zeros(1, 1, 28, 28),))
        x = torch.zeros(1, 1, 14, 14)
        self.assertEqual(g(x), f(x))

    @_tmp_donotuse_dont_inline_everything
    def test_trace_optional(self):
        if False:
            return 10

        @torch.jit.script
        def test(x: Optional[Tensor]):
            if False:
                i = 10
                return i + 15
            if x is None:
                return torch.zeros(1)
            else:
                return x

        def test_none():
            if False:
                while True:
                    i = 10
            return test(None)

        def test_tensor():
            if False:
                print('Hello World!')
            return test(torch.zeros(2))
        f_none = torch.jit.trace(test_none, ())
        self.assertEqual(f_none(), torch.zeros(1))
        f_tensor = torch.jit.trace(test_tensor, ())
        self.assertEqual(f_tensor(), torch.zeros(2))
        graph = f_tensor.graph
        FileCheck().check('name="test"').check_next('prim::CallFunction').run(graph)

    def test_trace_nested_datatypes(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return [[x + 1, x - 1], [x + 2, x - 2]]

        def bar(x):
            if False:
                print('Hello World!')
            list_stuff = foo(x)
            return (list_stuff[0][0], list_stuff[1][1])
        traced = torch.jit.trace(bar, torch.rand(3, 4))
        x = torch.rand(5, 6)
        self.assertEqual(bar(x), traced(x))

    @_tmp_donotuse_dont_inline_everything
    def test_call_traced_fn_from_traced_module(self):
        if False:
            return 10

        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            if False:
                print('Hello World!')
            return torch.neg(x)

        class TracedModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return traced_fn(torch.mm(x, self.param))
        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))
        FileCheck().check('aten::mm').check('name="traced_fn"').check_next('prim::CallFunction').run(str(tm.graph))

    @_tmp_donotuse_dont_inline_everything
    def test_call_traced_module_from_traced_module(self):
        if False:
            print('Hello World!')

        class TracedModule1(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(5, 7))

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.mm(x, self.param)

        class TracedModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))
                self.mod = torch.jit.trace(TracedModule1(), torch.rand(3, 5))

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.mod(torch.mm(x, self.param)) + 1.0
        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))
        FileCheck().check('aten::mm').check('prim::CallMethod').check_same('forward').check('aten::add').run(str(tm.graph))

    def test_index_put_trace_with_view(self):
        if False:
            for i in range(10):
                print('nop')

        @_trace(torch.rand(100), torch.tensor([1, 2, 3, 4]), torch.rand(1, 1, 1, 4))
        def test_index_put(target, indices, rhs):
            if False:
                while True:
                    i = 10
            target[indices] = rhs
            return target
        FileCheck().check('aten::view').check('index_put_').run(str(test_index_put.graph))

    def test_index_put_trace_without_view(self):
        if False:
            i = 10
            return i + 15

        @_trace(torch.rand(100), torch.tensor([1, 2, 3, 4]), torch.rand(4))
        def test_index_put(target, indices, rhs):
            if False:
                print('Hello World!')
            target[indices] = rhs
            return target
        FileCheck().check_not('aten::view').check('index_put_').run(str(test_index_put.graph))

    @suppress_warnings
    def test_trace_checker_dot_data(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(torch.jit.TracingCheckError, 'Tensor-valued Constant nodes differed in value across invocations'):

            @_trace(torch.rand(3, 4), check_inputs=[(torch.rand(3, 4),)])
            def foo(x):
                if False:
                    print('Hello World!')
                y = x.data
                return x + y

    @suppress_warnings
    def test_trace_checker_control_flow(self):
        if False:
            i = 10
            return i + 15

        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            for _ in range(x.size(0)):
                x = torch.neg(x)
            return x
        with self.assertRaisesRegex(torch.jit.TracingCheckError, 'Graphs differed across invocations!'):
            torch.jit.trace(foo, torch.randn(3, 4), check_inputs=[torch.randn(4, 4)])

    @suppress_warnings
    def test_trace_checker_memoization(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(torch.jit.TracingCheckError, 'Graphs differed across invocations!'):

            def foo(x):
                if False:
                    return 10
                if not hasattr(foo, 'cache'):
                    foo.cache = torch.neg(x)
                return x + foo.cache
            traced = torch.jit.trace(foo, torch.rand(3, 4), check_inputs=[(torch.rand(3, 4),)])

    def test_trace_checker_slice_lhs(self):
        if False:
            print('Hello World!')

        def foo(x):
            if False:
                print('Hello World!')
            for i in range(3):
                x[i, :] = torch.zeros(4)
            return x
        self.checkTrace(foo, (torch.rand(3, 4),), inputs_require_grads=False)

    def test_trace_checker_inplace_on_view(self):
        if False:
            i = 10
            return i + 15

        def foo(x):
            if False:
                while True:
                    i = 10
            x.view(-1).add_(-x.view(-1))
            return x
        with self.assertWarnsRegex(torch.jit.TracerWarning, 'Output nr 1. of the traced function does not match the corresponding output of the Python function'):
            torch.jit.trace(foo, torch.rand(3, 4), check_inputs=[torch.rand(5, 6)], _force_outplace=True)

    def test_lhs_index_fails(self):
        if False:
            print('Hello World!')

        def foo(x):
            if False:
                return 10
            x[0, 1] = 4
            return x
        with self.assertWarnsRegex(torch.jit.TracerWarning, 'cause the trace to be incorrect'):
            torch.jit.trace(foo, torch.rand(3, 4), _force_outplace=True)

    def test_lhs_index_trivial(self):
        if False:
            print('Hello World!')

        def foo(y, x):
            if False:
                return 10
            y[...] = x
            return y
        self.checkTrace(foo, (torch.rand(3, 4), torch.rand(4)), inputs_require_grads=False)

    def test_inplace_warn(self):
        if False:
            return 10

        def foo(x):
            if False:
                i = 10
                return i + 15
            x.view(-1).add_(-x.view(-1))
            return x
        with self.assertWarnsRegex(torch.jit.TracerWarning, 'cause the trace to be incorrect'):
            torch.jit.trace(foo, torch.rand(3, 4), _force_outplace=True)

    @suppress_warnings
    def test_trace_checker_dropout_train(self):
        if False:
            return 10

        def foo(x):
            if False:
                while True:
                    i = 10
            return torch.dropout(x, p=0.5, train=True)
        with self.assertWarnsRegex(torch.jit.TracerWarning, 'Output nr 1. of the traced function does not match the corresponding output of the Python function'):
            torch.jit.trace(foo, torch.rand(3, 4), check_inputs=[torch.rand(5, 6)])
        with self.assertWarnsRegex(torch.jit.TracerWarning, 'Trace had nondeterministic nodes'):
            torch.jit.trace(foo, torch.rand(3, 4), check_inputs=[torch.rand(5, 6)])

    def test_trace_checker_dropout_notrain(self):
        if False:
            print('Hello World!')
        input = torch.rand(3, 4)

        @_trace(input)
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.dropout(x, p=0.5, train=False)
        self.assertEqual(foo(input), input)

    def test_trace_contiguous(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return x[:, :, ::2].contiguous().view(12)
        x = torch.rand(2, 3, 4)
        traced = torch.jit.trace(foo, (x,))
        y = traced(x)
        self.assertNotEqual(x.storage().data_ptr(), y.storage().data_ptr())

    def test_trace_contiguous_short_circuit(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(x):
            if False:
                print('Hello World!')
            return x.contiguous()
        x = torch.rand(2, 3, 4)
        traced = torch.jit.trace(foo, (x,))
        FileCheck().check('aten::contiguous').run(str(traced.graph))

    def test_trace_inverse(self):
        if False:
            return 10

        def foo(x):
            if False:
                while True:
                    i = 10
            return ~x
        foo_traced = torch.jit.trace(foo, torch.zeros(3, 4, dtype=torch.uint8))
        eg = torch.zeros(3, dtype=torch.uint8)
        self.assertEqual(foo_traced(eg), foo(eg))

    def test_trace_modulelist(self):
        if False:
            i = 10
            return i + 15

        class MySubmod(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                if False:
                    return 10
                return self.relu(x)

        class MyMod(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.ml = torch.nn.ModuleList([MySubmod(), MySubmod()])

            def forward(self, x):
                if False:
                    print('Hello World!')
                for mod in self.ml:
                    x = mod(x)
                return x
        traced = torch.jit.trace(MyMod(), (torch.rand(3, 4),))

    def test_trace_fork_join_and_module(self):
        if False:
            for i in range(10):
                print('nop')

        class MySubmod(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return (self.relu(x), torch.neg(x))

        class Mod(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.ml = torch.nn.ModuleList([MySubmod() for i in range(2)])

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                futs = []
                for i in range(2):
                    futs.append(torch.jit._fork(self.ml[i], x))
                results = []
                for i in range(2):
                    results.append(torch.jit._wait(futs[i])[0])
                return torch.stack(results)
        m = Mod()
        traced = torch.jit.trace(m, torch.rand(3, 4))

    def test_trace_invert_module_hierarchy(self):
        if False:
            for i in range(10):
                print('nop')

        class MySubmod(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                if False:
                    return 10
                return (self.relu(x), torch.neg(x))

        class MyFunctionalMod(torch.nn.Module):

            def forward(self, x, submod):
                if False:
                    for i in range(10):
                        print('nop')
                return submod(x)

        class Mod(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.sm = MySubmod()
                self.fm = MyFunctionalMod()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.fm(x, self.sm)
        torch.jit.trace(Mod(), (torch.rand(3, 4),))

    @skipIfCrossRef
    def test_trace_records_names(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(bar, baz):
            if False:
                i = 10
                return i + 15
            baz = bar + 3
            quick_brown_fox = torch.neg(baz)
            for _ in range(20):
                yeet = quick_brown_fox - 3.14
            return yeet
        traced = torch.jit.trace(foo, (torch.rand(3, 3), torch.rand(3, 3)))
        graph_str = str(traced.graph)
        assert 'bar' in graph_str
        assert 'baz' in graph_str
        assert 'quick_brown_fox' in graph_str

    @skipIfTorchDynamo('Not a suitable test for TorchDynamo')
    def test_tracing_hooks(self):
        if False:
            print('Hello World!')

        class Net(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x + x

        def test_hook(is_post_hook, hook, fc):
            if False:
                while True:
                    i = 10
            n = Net()
            if is_post_hook:
                n.register_forward_hook(hook)
            else:
                n.register_forward_pre_hook(hook)
            module = torch.jit.trace(n, (torch.tensor(1.0),))
            eager_input = torch.tensor(1.0)
            eager_out = n(eager_input)
            fc.run(module.forward.graph)
            input = torch.tensor(1.0)
            output = module(input)
            self.assertEqual(input, eager_input)
            self.assertEqual(output, eager_out)

        def hook_no_return(mod, input, output):
            if False:
                while True:
                    i = 10
            input[0].add_(1)
            output.sub_(1)
        fc = FileCheck().check('add(').check('add_(').check('sub_(')
        test_hook(True, hook_no_return, fc)

        def hook_return(mod, input, output):
            if False:
                for i in range(10):
                    print('nop')
            input[0].add_(1)
            return output - 3
        fc = FileCheck().check('add(').check('add_(').check('sub(')
        test_hook(True, hook_return, fc)
        b = torch.tensor(3.0)

        def captured_hook(mod, input, output):
            if False:
                print('Hello World!')
            return output - b
        fc = FileCheck().check('add(').check('sub(')
        test_hook(True, captured_hook, fc)

        def pre_hook_no_ret(mod, input):
            if False:
                while True:
                    i = 10
            input[0].add_(3)
        fc = FileCheck().check('add_(').check('add(')
        test_hook(False, pre_hook_no_ret, fc)

        def pre_hook_ret(mod, input):
            if False:
                print('Hello World!')
            return input[0] - 4
        fc = FileCheck().check('sub(').check('add(')
        test_hook(False, pre_hook_ret, fc)

    def test_tracing_backward_hook_error(self):
        if False:
            return 10

        class Net(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x + x
        n = Net()

        def backward_hook(module, grad_input, grad_output):
            if False:
                i = 10
                return i + 15
            pass
        n.register_backward_hook(backward_hook)
        with self.assertRaisesRegex(Exception, 'backward hooks assigned'):
            torch.jit.trace(n, (torch.tensor(1.0),))

    def test_tracing_multiple_methods(self):
        if False:
            i = 10
            return i + 15

        class Net(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.conv(x)

            def weighted_kernel_sum(self, weight):
                if False:
                    i = 10
                    return i + 15
                return weight * self.conv.weight
        example_weight = torch.rand(1, 1, 3, 3)
        example_forward_input = torch.rand(1, 1, 3, 3)
        inputs = {'forward': example_forward_input, 'weighted_kernel_sum': example_weight}
        n = Net()
        module = torch.jit.trace_module(n, inputs)
        check_inputs = []
        for i in range(2):
            check_weight = torch.rand(1, 1, 3, 3)
            check_forward_input = torch.rand(1, 1, 3, 3)
            check_inputs.append({'forward': check_forward_input, 'weighted_kernel_sum': check_weight})
        module = torch.jit.trace_module(n, inputs, check_trace=True, check_inputs=check_inputs)
        self.assertTrue(module._c._has_method('forward'))
        self.assertTrue(module._c._has_method('weighted_kernel_sum'))
        module = torch.jit.trace(n.forward, example_forward_input)
        module = torch.jit.trace(n.forward, example_forward_input, check_trace=True, check_inputs=[example_forward_input])
        with self.assertRaisesRegex(AttributeError, "trace doesn't support compiling individual module's functions"):
            module = torch.jit.trace(n.weighted_kernel_sum, inputs)

    def test_tensor_with_grad_as_constant(self):
        if False:
            i = 10
            return i + 15
        param = torch.randn(3).requires_grad_()
        x = torch.randn(3)

        def f(x):
            if False:
                i = 10
                return i + 15
            return x + param
        with self.assertRaisesRegex(RuntimeError, 'Cannot insert a Tensor that requires grad as a constant'):
            torch.jit.trace(f, x)

    def test_non_tensor_tracing(self):
        if False:
            return 10

        def f(x):
            if False:
                while True:
                    i = 10
            return x + param
        with self.assertRaisesRegex(RuntimeError, "Type 'Tuple\\[int\\]' cannot be traced"):
            torch.jit.trace(f, (1,))

    def test_trace_skip_none_submodule(self):
        if False:
            while True:
                i = 10

        class TestModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.submod = torch.nn.Linear(3, 4)
                self.submod = None

            def forward(self, inputs):
                if False:
                    while True:
                        i = 10
                return inputs
        m = TestModule()
        tm = torch.jit.trace(m, torch.tensor(1.0))
        self.assertFalse(hasattr(tm, 'submod'))

    def test_trace_with_conditional_property(self):
        if False:
            return 10

        class Net(nn.Module):

            def __init__(self, attr=None):
                if False:
                    while True:
                        i = 10
                super().__init__()
                if attr is not None:
                    self._attr = attr
                self.attr_name = '_attr'

            @property
            def attr(self):
                if False:
                    return 10
                return getattr(self, self.attr_name)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x
        x = torch.ones(1)
        torch.jit.trace(Net(), x)

    def test_trace_func_argument_names_captured(self):
        if False:
            print('Hello World!')

        def fn(first_arg: torch.Tensor, second_arg: torch.Tensor) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            return first_arg + second_arg
        traced_fn = torch.jit.trace(fn, (torch.ones(1), torch.ones(1)))
        FileCheck().check('first_arg').check_next('second_arg').run(str(traced_fn.graph))

    def test_trace_partial_func_argument_names_captured(self):
        if False:
            print('Hello World!')

        def fn(first_arg: torch.Tensor, second_arg=1) -> torch.Tensor:
            if False:
                return 10
            return first_arg + second_arg
        traced_fn = torch.jit.trace(fn, (torch.ones(1),))
        FileCheck().check('first_arg').check_not('second_arg').run(str(traced_fn.graph))

    def test_trace_module_argument_names_captured(self):
        if False:
            print('Hello World!')

        class TestModule(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, first_arg: torch.Tensor, second_arg: torch.Tensor):
                if False:
                    while True:
                        i = 10
                return self.conv(first_arg) + second_arg
        m = TestModule()
        example_input = (torch.ones(1, 1, 3, 3), torch.ones(1, 1, 3, 3))
        traced_module_forward = torch.jit.trace(m.forward, example_input)
        FileCheck().check('first_arg').check_next('second_arg').run(str(traced_module_forward.graph))
        traced_module = torch.jit.trace(m, example_input)
        FileCheck().check('first_arg').check_next('second_arg').run(str(traced_module.graph))

    def test_trace_checking_with_deprecated_name(self):
        if False:
            print('Hello World!')

        class MyClass(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super(MyClass, self).__init__()

            def forward(self, x, y, **deprecated_arguments):
                if False:
                    i = 10
                    return i + 15
                if len(deprecated_arguments) > 0:
                    raise RuntimeError(f'Got unexpected arguments: {deprecated_arguments}')
                return x + y
        model = MyClass()
        m2 = torch.jit.trace(model, (torch.ones(1), torch.ones(1)))
        m3 = torch.jit.trace(model, example_kwarg_inputs={'x': torch.ones(1), 'y': torch.ones(1)}, strict=False)

    def test_trace_with_tuple_tensor(self):
        if False:
            while True:
                i = 10

        class MyClass(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super(MyClass, self).__init__()

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return x + y[0] + y[1]
        model = MyClass()
        traced_model = torch.jit.trace(model, (torch.ones(1), (torch.ones(1), torch.ones(1))))
        input_dict = {'x': torch.tensor([2, 3]), 'y': (torch.tensor([5, 6]), torch.tensor([7, 8]))}
        self.assertEqual(model(**input_dict), traced_model(**input_dict))
        traced_model = torch.jit.trace(model, example_kwarg_inputs={'x': torch.ones(1), 'y': (torch.ones(1), torch.ones(1))})
        self.assertEqual(model(**input_dict), traced_model(**input_dict))

    def test_trace_no_duplicated_lifted_input_output(self):
        if False:
            for i in range(10):
                print('nop')

        class Normalize(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.norm = nn.GroupNorm(num_groups=32, num_channels=32)

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                if y is None:
                    y = x
                else:
                    y = self.norm(y)
                y = y * 2
                return y

        class G(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.norm = Normalize()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                A = self.norm(x, None)
                B = F.relu(A)
                return (A, B)

        class Net(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.g = G()
                self.norm_1 = Normalize()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                hs = self.g(x)
                (A, B) = hs
                h = self.norm_1(B, A)
                return h
        net = Net()
        net = net.eval()
        x = torch.randn(1, 32, 16, 16)
        traced = torch.jit.trace(net, x)
        FileCheck().check_not('prim::TupleUnpack').run(str(traced.graph))

@skipIfTorchDynamo('Not a suitable test for TorchDynamo')
class TestMixTracingScripting(JitTestCase):

    def test_trace_script(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def func1(x: Tuple[Tensor, Tensor]) -> Tensor:
            if False:
                return 10
            return x[0] + x[1]

        @torch.jit.script
        def func2(x: List[Tensor]) -> Tensor:
            if False:
                while True:
                    i = 10
            return x[0] + x[1]
        a = torch.randn(5)
        b = torch.randn(5)
        self.checkTrace(func1, ((a, b),))
        self.checkTrace(func2, ((a, b),))

        @torch.jit.script
        def func3(x: Tensor, method: str='bilinear', align_corners: bool=True) -> Tensor:
            if False:
                while True:
                    i = 10
            hw = x.shape[2:4]
            return F.interpolate(x, hw, mode=method, align_corners=align_corners)
        inp = torch.rand(1, 3, 6, 6)
        self.checkTrace(func3, (inp,))

        @torch.jit.script
        def func4(x: Tensor, a: List[Optional[str]]) -> Tensor:
            if False:
                i = 10
                return i + 15
            if len(a) == 2:
                return x + 2
            else:
                return x

    def test_trace_mixed_by_script_with_dict_output(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def return_dict(input: torch.Tensor) -> Dict[str, torch.Tensor]:
            if False:
                i = 10
                return i + 15
            return {'foo': input + 1}

        class TraceModule(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                dict = return_dict(input)
                return dict['foo'] + dict['foo']
        x = torch.ones(1)
        tm = torch.jit.trace(TraceModule(), x)
        self.assertEqual(tm(x), x + 1 + x + 1)

    def test_trace_of_script(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def foo(a, c):
            if False:
                for i in range(10):
                    print('nop')
            b = 0.0
            if bool(a == 0.0):
                b = 1.0
            return b + c
        a = torch.ones(1, dtype=torch.float)

        @_trace(torch.zeros(1, dtype=torch.float))
        def use(b):
            if False:
                return 10
            return foo(b - 1.0, a) + 1.0
        self.assertTrue('Dynamic' not in str(use.graph))
        self.assertEqual(3, use(torch.ones(1, dtype=torch.float)))
        self.assertEqual(2, use(torch.zeros(1, dtype=torch.float)))

    def test_trace_with_size(self):
        if False:
            return 10

        @_trace(torch.zeros(1, 1))
        def foo(x):
            if False:
                return 10
            return x + 1

        @torch.jit.script
        def bar(x):
            if False:
                i = 10
                return i + 15
            y = int(foo(x))
            if 1 == 1:
                y = 7
            return y + 1
        self.assertEqual(8, bar(torch.ones(1, 1)))

    def test_tracing_slicing(self):
        if False:
            return 10

        @_trace(torch.zeros(10))
        def foo_trace(x):
            if False:
                i = 10
                return i + 15
            return x[-5:-3]

        @torch.jit.script
        def foo_script(x):
            if False:
                while True:
                    i = 10
            return x[-5:-3]

        def foo(x):
            if False:
                return 10
            return x[-5:-3]
        a = torch.arange(0, 8)
        b = torch.arange(0, 20)
        self.assertEqual(foo_trace(a), foo_script(a))
        self.assertEqual(foo_trace(a), foo(a))
        self.assertNotEqual(foo_trace(a), foo_trace(b))

    def test_tracing_indexing(self):
        if False:
            print('Hello World!')

        @_trace(torch.zeros(10))
        def foo_trace(x):
            if False:
                print('Hello World!')
            return x[-2]

        @torch.jit.script
        def foo_script(x):
            if False:
                i = 10
                return i + 15
            return x[-2]

        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return x[-2]
        a = torch.arange(0, 8)
        b = torch.arange(0, 20)
        self.assertEqual(foo_script(a), foo_trace(a))
        self.assertEqual(foo_trace(a), foo(a))
        self.assertNotEqual(foo_trace(a), foo_trace(b))

    def test_trace_hierarchy(self):
        if False:
            return 10

        class AnotherScriptMod(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(1, 2, 3))

            @torch.jit.script_method
            def bar(self):
                if False:
                    print('Hello World!')
                return torch.zeros(4, 5)

        class SomeScriptMod(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.asm = AnotherScriptMod()

            @torch.jit.script_method
            def foo(self):
                if False:
                    return 10
                return torch.zeros(3, 4)

            @torch.jit.script_method
            def bar(self):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.zeros(4, 3)

        class TraceMe(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.ssm = SomeScriptMod()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.ssm.bar() + x
        orig = TraceMe()
        traced = torch.jit.trace(orig, (torch.rand(4, 3),))
        self.assertTrue(traced.ssm._c._has_method('foo'))
        self.assertTrue(hasattr(traced.ssm, 'foo'))
        imported = self.getExportImportCopy(traced)
        self.assertTrue(imported.ssm._c._has_method('foo'))
        self.assertTrue(hasattr(imported.ssm, 'foo'))
        self.assertTrue(imported.ssm.asm._c._has_method('bar'))
        self.assertTrue(hasattr(imported.ssm.asm, 'bar'))
        self.assertTrue(hasattr(imported.ssm.asm, 'param'))

    def test_trace_parameter(self):
        if False:
            print('Hello World!')

        class Param(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.register_parameter('bias', nn.Parameter(torch.empty(4, 4)))

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x

        class M3(torch.jit.ScriptModule):

            def __init__(self, model):
                if False:
                    return 10
                super().__init__()
                self.traced = torch.jit.trace(model, torch.rand(3, 3))

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.traced(x)

        class M2(nn.Module):

            def __init__(self, model):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.module = M3(model)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.module(x)

        class M1(torch.jit.ScriptModule):

            def __init__(self, model):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.traced = torch.jit.trace(M2(model), torch.rand(3, 3))

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    return 10
                return self.traced(x)
        with torch.jit.optimized_execution(False):
            module = M1(Param())
            f = io.BytesIO()
            torch.jit.save(module, f)

    @_tmp_donotuse_dont_inline_everything
    def test_call_script_fn_from_traced_module(self):
        if False:
            return 10

        @torch.jit.script
        def scripted_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.neg(x)

        class TracedModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return scripted_fn(torch.mm(x, self.param))
        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))
        FileCheck().check('aten::mm').check('name="scripted_fn"').check('prim::CallFunction').run(str(tm.graph))

    @_tmp_donotuse_dont_inline_everything
    def test_call_script_module_from_traced_module(self):
        if False:
            return 10

        class ScriptMod(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.param_foo = torch.nn.Parameter(torch.rand(5, 7))

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.mm(x, self.param_foo)

        class TracedModule(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))
                self.mod = ScriptMod()

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.mod(torch.mm(x, self.param)) + 1.0
        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))
        FileCheck().check('aten::mm').check('prim::CallMethod').check_same('forward').check('aten::add').run(str(tm.graph))

    @_tmp_donotuse_dont_inline_everything
    def test_call_traced_fn_from_script_fn(self):
        if False:
            print('Hello World!')

        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            if False:
                while True:
                    i = 10
            return torch.neg(x)

        @torch.jit.script
        def script_fn(x):
            if False:
                return 10
            return traced_fn(x) + 1
        FileCheck().check('prim::CallFunction').check('aten::add').run(str(script_fn.graph))

    def test_call_traced_mod_from_script_fn(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, 'Cannot call a ScriptModule that is not a submodule of the caller'):

            class TracedModule(torch.nn.Module):

                def forward(self, x):
                    if False:
                        return 10
                    return torch.mm(x, torch.zeros(4, 3))
            tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

            @torch.jit.script
            def script_fn(x):
                if False:
                    for i in range(10):
                        print('nop')
                return tm(x) + 1

    @_tmp_donotuse_dont_inline_everything
    def test_call_tracing_fn_from_script_module(self):
        if False:
            while True:
                i = 10

        @_trace(torch.rand(3, 3))
        def traced_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.neg(x)

        class ScriptMod(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return traced_fn(torch.mm(x, self.param))
        sm = ScriptMod()
        FileCheck().check('aten::mm').check('prim::CallFunction').run(str(sm.forward.graph))

    @_tmp_donotuse_dont_inline_everything
    def test_call_tracing_mod_from_script_module(self):
        if False:
            print('Hello World!')

        class TracedMod(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 5))

            def forward(self, x):
                if False:
                    return 10
                return torch.mm(x, self.param)

        class ScriptMod(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))
                self.tm = torch.jit.trace(TracedMod(), torch.rand(3, 3))

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.tm(torch.mm(x, self.param))
        sm = ScriptMod()
        FileCheck().check('aten::mm').check('prim::CallMethod').run(str(sm.graph))

    def test_script_inline_trace_multiple_args(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def forward(self, input, input2):
                if False:
                    i = 10
                    return i + 15
                return input + input2

        class M2(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.m = torch.jit.trace(M(), (torch.zeros(4, 3), torch.zeros(4, 3)))

            @torch.jit.script_method
            def forward(self, inp):
                if False:
                    print('Hello World!')
                return self.m(inp, inp)
        with torch.jit.optimized_execution(False):
            m2 = M2()
            m2(torch.zeros(4, 3))

    def test_trace_dict_mix_script(self):
        if False:
            for i in range(10):
                print('nop')

        class testB(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, feature_map: Dict[str, List[Tensor]]) -> Tensor:
                if False:
                    while True:
                        i = 10
                output = []
                for j in feature_map.values():
                    output.append(self.linear(j[0]))
                return torch.stack(output)

        class testA(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.b = torch.jit.script(testB())

            def forward(self, input_map: Dict[str, List[Tensor]]) -> Tensor:
                if False:
                    while True:
                        i = 10
                feature_map = {}
                for (i, j) in input_map.items():
                    feature_map[i] = [j[0]]
                return self.b(feature_map)
        input_map = {'1': [torch.rand(2, 2), torch.rand(2, 2)], '3': [torch.rand(2, 2), torch.rand(2, 2)]}
        model = testA()
        traced_model = torch.jit.trace(model, input_map)
        new_input_map = {'1': [torch.rand(2, 2), torch.randn(2, 2)], '3': [torch.rand(2, 2), torch.rand(2, 2)]}
        self.assertEqual(model(new_input_map), traced_model(new_input_map))

    def test_trace_script_returning_complex_dict(self):
        if False:
            return 10
        'Tracing over a script function returning a dictionary should work.\n        The dictionary can should be able to contain other containers (like a tuple) recursively.\n        '

        class ReturnsDict(torch.nn.Module):

            def forward(self, id_score_list: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                if False:
                    return 10
                v = id_score_list['1000']
                idx_keys = v[1] - 1500000
                weights = v[2]
                result = {'1000': (v[0], idx_keys, weights)}
                return result

        class ChecksDict(torch.nn.Module):

            def forward(self, input: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
                if False:
                    i = 10
                    return i + 15
                v = input['1000']
                return v[1] + 1

        class TestModule(torch.nn.Module):

            def __init__(self, checks_dict, returns_dict):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.checks_dict = checks_dict
                self.returns_dict = returns_dict

            def forward(self, input: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
                if False:
                    return 10
                foo = self.returns_dict(input)
                return self.checks_dict(foo)
        input1 = {'1000': (torch.tensor([0]), torch.tensor([], dtype=torch.int64), torch.tensor([]))}
        input2 = {'1000': (torch.tensor([0]), torch.tensor([1500000, 1500004], dtype=torch.int64), torch.tensor([2.0, 3.0]))}
        checks_dict = torch.jit.script(ChecksDict())
        returns_dict = torch.jit.script(ReturnsDict())
        eager_module = TestModule(checks_dict, returns_dict)
        traced_module = torch.jit.trace(eager_module, input1)
        self.assertEqual(traced_module(input1), eager_module(input1))
        self.assertEqual(traced_module(input2), eager_module(input2))

    def test_trace_returning_dict_with_tensor_tuples(self):
        if False:
            while True:
                i = 10
        'Tracing over a module returning a dictionary whose values are tuples of tensors\n        should work.\n        '

        class ReturnsDict(torch.nn.Module):

            def forward(self, k: torch.Tensor, v: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
                if False:
                    return 10
                x = 2 * k
                y = 3 * v
                result = {'imakey': (x, y)}
                return result

        class ReturnsBadDict(torch.nn.Module):

            def forward(self, k: torch.Tensor, v: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, float]]:
                if False:
                    while True:
                        i = 10
                x = 2 * k
                result = {'imakey': (x, 1)}
                return result
        mod = ReturnsDict()
        traced_module = torch.jit.trace(mod, [torch.ones(1), torch.ones(1)], strict=False)
        out = traced_module(torch.ones(1), torch.ones(1))
        expected = {'imakey': (torch.tensor([2.0]), torch.tensor([3.0]))}
        self.assertEqual(out, expected)
        with self.assertRaisesRegex(RuntimeError, 'cannot be understood by the tracer, only outputs matching'):
            mod = ReturnsBadDict()
            traced_module = torch.jit.trace(mod, [torch.ones(1), torch.ones(1)], strict=False)

    def test_trace_linear(self):
        if False:
            print('Hello World!')
        m = torch.nn.Linear(20, 20)
        inp = torch.rand([20, 20])
        self.checkTrace(m, (inp,))
        g = torch.jit.trace(m, (inp,)).graph
        FileCheck().check('aten::linear').run(g)

    def test_traced_module_implements_interface(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.interface
        class TestModuleInterface(nn.Module):

            def forward(self, first_arg: torch.Tensor, second_arg: torch.Tensor) -> torch.Tensor:
                if False:
                    i = 10
                    return i + 15
                pass
        make_global(TestModuleInterface)

        class TestModule(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, first_arg: torch.Tensor, second_arg: torch.Tensor) -> torch.Tensor:
                if False:
                    while True:
                        i = 10
                return self.conv(first_arg) + second_arg

        def fn_takes_interface(x: TestModuleInterface):
            if False:
                print('Hello World!')
            ones = torch.ones(1, 1, 3, 3)
            return x.forward(ones, ones)
        scripted_test_module = torch.jit.script(TestModule())
        self.checkScript(fn_takes_interface, (scripted_test_module,))

    def test_traced_module_contains_scripted_interface_types(self):
        if False:
            while True:
                i = 10

        class LeafModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(19))

            def forward(self, input: torch.Tensor):
                if False:
                    while True:
                        i = 10
                return input + self.weight

        class LowerModuleImpl(torch.nn.Module):

            def __init__(self) -> None:
                if False:
                    return 10
                super().__init__()
                self.leaf = LeafModule()

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                if False:
                    while True:
                        i = 10
                return self.leaf(input)

        @torch.jit.interface
        class LowerModuleInterface(torch.nn.Module):

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                if False:
                    return 10
                pass

        class MiddleModule(torch.nn.Module):
            lower: LowerModuleInterface

            def __init__(self, feature_processor_modules=None):
                if False:
                    print('Hello World!')
                super().__init__()
                self.lower = LowerModuleImpl()

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return self.lower(input)

        class WrapperModule(torch.nn.Module):

            def __init__(self, m):
                if False:
                    return 10
                super().__init__()
                self.middle = m

            def forward(self, input):
                if False:
                    return 10
                return self.middle(input)

        class TopModule(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                m = MiddleModule()
                m = torch.jit.script(m)
                self.sub1 = m
                self.sub2 = WrapperModule(m)

            def forward(self, input: torch.Tensor):
                if False:
                    i = 10
                    return i + 15
                return self.sub1(input) + self.sub2(input)
        top = TopModule()
        top_example_input = torch.ones(1)
        torch.jit.trace(top, top_example_input)

    def test_jit_trace_callfunction_return_shapes(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def inner_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.cat((x, x))

        def outer_fn(x, y):
            if False:
                print('Hello World!')
            return inner_fn(x + y).relu()
        (x, y) = [torch.rand((2, 2), dtype=torch.float) for _ in range(2)]
        fn_t = torch.jit.trace(outer_fn, (x, y))
        FileCheck().check('Float').check('4, 2').check('CallFunction').run(fn_t.graph)
        for n in fn_t.graph.nodes():
            if n.kind() == 'prim::CallFunction':
                self.assertTrue(n.output().isCompleteTensor())