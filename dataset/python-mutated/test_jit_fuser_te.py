import operator
import os
import unittest
import contextlib
import math
import torch
import torch.nn.functional as F
from torch.testing import FileCheck
from typing import List
import warnings
torch._C._jit_set_profiling_executor(True)
torch._C._get_graph_executor_optimize(True)
from torch.testing._internal.common_utils import run_tests, ProfilingMode, GRAPH_EXECUTOR, enable_profiling_mode_for_profiling_tests, slowTest, skipIfTorchDynamo, TEST_WITH_ASAN, IS_FBCODE
from torch.testing._internal.jit_utils import JitTestCase, RUN_CUDA, RUN_CUDA_HALF, RUN_CUDA_MULTI_GPU, warmup_backward, set_fusion_group_inlining, clone_inputs, get_traced_sample_variant_pairs, TensorExprTestOptions, NoTracerWarnContextManager
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_device_type import ops, onlyCPU, instantiate_device_type_tests, OpDTypes
from torch.testing._internal.common_jit import JitCommonTestCase
from torch.testing._internal.jit_metaprogramming_utils import create_traced_fn
from textwrap import dedent
from itertools import product, permutations, combinations
from test_jit import backward_graph, get_lstm_inputs, get_milstm_inputs, LSTMCellC, LSTMCellF, LSTMCellS, MiLSTMCell
from jit.test_fuser_common import TestFuserCommon
FUSION_GROUP = 'prim::TensorExprGroup'
LLVM_ENABLED = torch._C._llvm_enabled()
autograd_check_set = {'aten::__is__', 'prim::AutogradAllNonZero', 'prim::AutogradAllZero', 'prim::ListConstruct'}

def strip_profiling_nodes(nodes):
    if False:
        i = 10
        return i + 15
    profiling_opcodes = {'prim::BailoutTemplate', 'prim::BailOut'}
    return [n for n in nodes if n.kind() not in profiling_opcodes]

def warmup_forward(f, *args, profiling_count=2):
    if False:
        return 10
    for i in range(profiling_count):
        results = f(*args)
    return results

@contextlib.contextmanager
def texpr_reductions_enabled():
    if False:
        return 10
    old = torch._C._jit_set_texpr_reductions_enabled(True)
    try:
        yield
    finally:
        torch._C._jit_set_texpr_reductions_enabled(old)

@contextlib.contextmanager
def texpr_enable_strategy(strategy):
    if False:
        i = 10
        return i + 15
    old = torch._C._jit_set_fusion_strategy(strategy)
    try:
        yield
    finally:
        torch._C._jit_set_fusion_strategy(old)

@contextlib.contextmanager
def inline_fusion_groups():
    if False:
        return 10
    old_inlining = torch._C._debug_get_fusion_group_inlining()
    torch._C._debug_set_fusion_group_inlining(True)
    try:
        yield
    finally:
        torch._C._debug_set_fusion_group_inlining(old_inlining)

@skipIfTorchDynamo()
class TestTEFuser(JitTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.tensorexpr_options = TensorExprTestOptions()
        fusion_strategy = [('DYNAMIC', 20)] if self.dynamic_shapes else [('STATIC', 20)]
        self.old_fusion_strategy = torch._C._jit_set_fusion_strategy(fusion_strategy)
        self.devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']
        self.int_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.bool]
        self.fp_dtypes = [torch.float16, torch.float32, torch.float64, torch.bfloat16]
        self.dtypes = self.int_dtypes + self.fp_dtypes

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tensorexpr_options.restore()
        torch._C._jit_set_fusion_strategy(self.old_fusion_strategy)
        super().tearDown()

    def assertAllFused(self, graph, except_for=None):
        if False:
            for i in range(10):
                print('nop')
        except_for = except_for if except_for is not None else set()
        guards = ('prim::TypeCheck', 'prim::RequiresGradCheck', 'prim::TensorExprDynamicGuard')
        guard_found = False

        def autodiff_guard(node):
            if False:
                for i in range(10):
                    print('nop')
            if node.kind() != 'aten::all':
                return False
            inps = list(node.inputs())
            if len(inps) != 1 or inps[0].node().kind() != 'prim::ListConstruct':
                return False
            li_inps = list(inps[0].node().inputs())
            for li_inp in li_inps:
                if li_inp.node().kind() in ('prim::AutogradAllNonZero', 'prim::AutogradAllZero'):
                    return True
            return False

        def is_guard(node):
            if False:
                i = 10
                return i + 15
            return node.kind() in guards or autodiff_guard(node)
        for node in graph.block().nodes():
            if node.kind() == 'prim::Constant':
                continue
            if is_guard(node):
                self.assertFalse(guard_found)
                guard_found = True
                continue
            if node.kind() in except_for:
                continue
            if node.kind() == 'prim::If':
                self.assertTrue(is_guard(node.prev()))
                continue
            self.assertTrue(False, 'Found unexpected node:' + node.kind())
        self.assertTrue(guard_found)

    def assertLastGraphAllFused(self):
        if False:
            i = 10
            return i + 15
        self.assertAllFused(torch.jit.last_executed_optimized_graph())

    def findFusionGroups(self, graph):
        if False:
            while True:
                i = 10
        result = []
        for n in graph.nodes():
            if n.kind() == FUSION_GROUP:
                result.append(n.g('Subgraph'))
                continue
            for block in n.blocks():
                result += self.findFusionGroups(block)
        return result

    def test_typecheck(self):
        if False:
            i = 10
            return i + 15
        a = torch.ones(1)

        def fused_kernel(a, b):
            if False:
                while True:
                    i = 10
            return (a + b) * 2.0
        scripted = self.checkScript(fused_kernel, (a, a))
        graph = scripted.graph_for(a, a)
        fusion_groups = self.findFusionGroups(graph)
        self.assertEqual(len(fusion_groups), 1)
        a = torch.ones(2)
        self.assertEqual(scripted(a, a), fused_kernel(a, a))

    def test_sum_simple(self):
        if False:
            print('Hello World!')

        def func(x):
            if False:
                while True:
                    i = 10
            x2 = x * x
            return x2.sum()
        with texpr_reductions_enabled():
            a = torch.tensor(list(range(0, 15)), dtype=torch.float, device='cpu')
            a = a.reshape(5, 3)
            scripted = self.checkScript(func, (a,))
            self.assertLastGraphAllFused()

    def test_nop(self):
        if False:
            print('Hello World!')
        pass

    def test_sum_dim(self):
        if False:
            return 10

        def func(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.sum((0,)) * 2

        def func_neg(x):
            if False:
                print('Hello World!')
            return x.sum((-2,)) * 2
        with texpr_reductions_enabled():
            a = torch.tensor(list(range(0, 15)), dtype=torch.float, device='cpu')
            a = a.reshape(5, 3)
            scripted = self.checkScript(func, (a,))
            self.assertLastGraphAllFused()
            scripted = self.checkScript(func_neg, (a,))
            self.assertLastGraphAllFused()

    def test_sum_keepdim_cast(self):
        if False:
            i = 10
            return i + 15

        def func(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.sum((0,), keepdim=True, dtype=torch.double) * 2
        with texpr_reductions_enabled():
            a = torch.tensor(list(range(0, 15)), dtype=torch.float, device='cpu')
            a = a.reshape(5, 3)
            self.checkScript(func, (a,))
            self.assertLastGraphAllFused()

    def test_abs(self):
        if False:
            while True:
                i = 10
        for device in self.devices:

            def func(x):
                if False:
                    return 10
                return x.abs() * 2
            a = torch.randn(5, device=device)
            scripted = self.checkScript(func, (a,))
            self.assertLastGraphAllFused()

    def test_unsqueeze_size_calculation(self):
        if False:
            return 10
        for device in self.devices:

            def foo(b, d):
                if False:
                    print('Hello World!')
                x = d.unsqueeze(1)
                y = x * 42.0
                z = b + y
                r = z / 42.0
                return r
            inputs = (torch.rand(20, 28, device=device, requires_grad=True), torch.rand(20, device=device))
            scripted = self.checkScript(foo, inputs)
            self.assertAllFused(scripted.graph_for(*inputs))

    def test_zero_element_tensors(self):
        if False:
            i = 10
            return i + 15
        for device in self.devices:

            def decode(sin_t, cos_t):
                if False:
                    return 10
                theta = torch.atan2(sin_t.float(), cos_t.float())
                return theta
            sin = torch.zeros(0, device=device)
            cos = torch.zeros(0, device=device)
            inputs = [sin, cos]
            ge = self.checkScript(decode, inputs)

    def test_arg_configurations_smoke(self):
        if False:
            print('Hello World!')
        if self.dynamic_shapes:
            self.skipTest('TODO: chunk dynamic shapes')
        for device in self.devices:

            def f(x, y):
                if False:
                    while True:
                        i = 10
                (z1, z2) = (x + y).chunk(2, dim=1)
                return z1 * z2
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)
            traced_f = torch.jit.trace(f, (x, y))
            self.assertEqual(traced_f(x.t().contiguous(), y), traced_f(x.t(), y))

    def test_broadcast(self):
        if False:
            while True:
                i = 10
        for device in self.devices:

            def scaleshift(x, scale, shift):
                if False:
                    print('Hello World!')
                return x * scale + shift
            inputs = [torch.randn(4, 4, dtype=torch.float, device=device), torch.randn(4, dtype=torch.float, device=device), torch.randn(4, dtype=torch.float, device=device)]
            self.checkScript(scaleshift, inputs)

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    @unittest.skipIf(not RUN_CUDA_HALF, 'no half support')
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, 'no half support with profiling on')
    def test_cuda_half(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(4, 4, dtype=torch.half, device='cuda')
        y = torch.randn(4, 4, dtype=torch.half, device='cuda')
        funcs = [self.fn_test_comparison_gt_lt, self.fn_test_relu, self.fn_test_exp]
        inputs = (x.float(), y.float())
        fusion_inputs = (x, y)
        for fn in funcs:
            local_inputs = [t.clone().requires_grad_() for t in inputs]
            local_fusion_inputs = [t.clone().requires_grad_() for t in fusion_inputs]
            fusion = torch.jit.trace(fn, local_fusion_inputs, check_trace=False)
            outputs = fn(*local_inputs)
            fusion_outputs = fusion(*local_fusion_inputs)
            outputs_half = [t.half() for t in outputs]
            self.assertEqual(outputs_half, fusion_outputs)
            for (output, fusion_output) in zip(outputs_half, fusion_outputs):
                grads = torch.autograd.grad(output.float().sum(), local_inputs, allow_unused=True, retain_graph=True)
                fusion_grads = torch.autograd.grad(fusion_output.sum(), local_fusion_inputs, allow_unused=True, retain_graph=True)
                grads_half = [t.half() for t in grads]
                self.assertEqual(grads_half, fusion_grads)

    def test_checks_cat_inputs(self):
        if False:
            while True:
                i = 10
        with set_fusion_group_inlining(True):
            for device in self.devices:

                def f(x, y):
                    if False:
                        while True:
                            i = 10
                    return torch.cat([x + 2 * x + x ** 2, y + 4 * y + y ** 3], dim=0)
                x = torch.randn(2, 4, dtype=torch.float, device=device)
                y = torch.randn(1, 4, dtype=torch.float, device=device)
                scripted = self.checkScript(f, (x, y))
                self.assertEqual(scripted(x, y).shape, (3, 4))
                self.assertAllFused(scripted.graph_for(x, y))

    def test_chunk(self):
        if False:
            while True:
                i = 10
        if self.dynamic_shapes:
            self.skipTest('TODO: chunk dynamic shapes')
        for device in self.devices:

            def fn(x):
                if False:
                    return 10
                (a, b, c) = x.chunk(3, 1)
                return a * b + c
            inputs = [torch.randn(10, 6, dtype=torch.float, device=device)]
            self.checkScript(fn, inputs)
            self.assertLastGraphAllFused()

    def test_chunk_correctness(self):
        if False:
            for i in range(10):
                print('nop')
        if self.dynamic_shapes:
            self.skipTest('TODO: chunk dynamic shapes')
        for device in self.devices:

            def chunk_4_0(x):
                if False:
                    return 10
                (x0, x1, x2, x3) = x.chunk(4, 0)
                return x0 + x1 + x2 + x3

            def chunk_4_1(x):
                if False:
                    return 10
                (x0, x1, x2, x3) = x.chunk(4, 1)
                return x0 + x1 + x2 + x3

            def chunk_4_last(x):
                if False:
                    return 10
                (x0, x1, x2, x3) = x.chunk(4, 2)
                return x0 + x1 + x2 + x3
            fns = [chunk_4_0, chunk_4_1, chunk_4_last]
            tensors = [torch.randn(4, 4, 4, dtype=torch.float, device=device), torch.randn(12, 8, 16, dtype=torch.float, device=device), torch.randn(12, 8, 16, dtype=torch.float, device=device).transpose(1, 2)]
            for tensor in tensors:
                for fn in fns:
                    self.checkScript(fn, [tensor])
                    self.assertLastGraphAllFused()

    def test_chunk_distributes(self):
        if False:
            return 10
        if self.dynamic_shapes:
            self.skipTest('TODO: chunk dynamic shapes')
        if self.dynamic_shapes:
            self.skipTest('TODO: chunk dynamic shapes')
        for device in self.devices:

            def f(x, y):
                if False:
                    print('Hello World!')
                (z1, z2) = (x + y).chunk(2, dim=1)
                return z1 * z2
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)
            ge = self.checkTrace(f, (x, y))
            graph = ge.graph_for(x, y)
            FileCheck().check('with ' + FUSION_GROUP + '_').check_count('ConstantChunk', 1, exactly=True).run(str(graph))

    def test_chunk_motion_deduplicates_inputs(self):
        if False:
            while True:
                i = 10
        if self.dynamic_shapes:
            self.skipTest('TODO: chunk dynamic shapes')
        for device in self.devices:

            def func1(x):
                if False:
                    while True:
                        i = 10
                z = x * x
                (z0, z1) = z.chunk(2)
                return z0 * z1

            def func2(x):
                if False:
                    return 10
                z = x * x * x
                (z0, z1) = z.chunk(2)
                return z0 * z1
            inputs = [torch.tensor([1.1, 1.2], device=device, dtype=torch.float)]
            for func in [func1, func2]:
                self.checkScript(func, inputs)
                self.assertLastGraphAllFused()

    def test_chunk_multiple(self):
        if False:
            while True:
                i = 10
        if self.dynamic_shapes:
            self.skipTest('TODO: chunk dynamic shapes')
        for device in self.devices:

            def fn(s, x, y, z):
                if False:
                    return 10
                (z1, z2) = z.chunk(2, 2)
                (x1, x2, x3) = x.chunk(3, 1)
                (y1, y2) = y.chunk(2, 0)
                return s + x1 + x2 + x3 + y1 + y2 + z1 + z2
            inputs = [torch.randn(5, 2, 3, dtype=torch.float, device=device), torch.randn(5, 6, 3, dtype=torch.float, device=device), torch.randn(10, 2, 3, dtype=torch.float, device=device), torch.randn(5, 2, 6, dtype=torch.float, device=device)]
            ge = self.checkScript(fn, inputs)
            self.assertAllFused(ge.graph_for(*inputs))

    def test_minmax(self):
        if False:
            return 10
        for device in self.devices:

            def tmax(a, b):
                if False:
                    while True:
                        i = 10
                return torch.max(2 * a, b)

            def tmin(a, b):
                if False:
                    i = 10
                    return i + 15
                return torch.min(2 * a, b)
            a = torch.randn(4, 4, dtype=torch.float)
            b = torch.randn(4, 4, dtype=torch.float)
            nan = torch.tensor(float('nan'), dtype=torch.float)
            for (f, inputs, device) in product((tmax, tmin), ([a, b], [a, nan], [b, nan]), self.devices):
                inputs = [t.to(device) for t in inputs]
                s = self.checkScript(f, inputs)
                self.assertAllFused(s.graph_for(*inputs))

    def test_clamp(self):
        if False:
            i = 10
            return i + 15
        for device in self.devices:

            def func2(a, b):
                if False:
                    print('Hello World!')
                return torch.clamp(a + b, min=0, max=2)

            def funcInf(a, b):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.clamp(a + b, min=0, max=float('inf'))

            def funcNegInf(a, b):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.clamp(a + b, min=float('-inf'), max=0)

            def funcOptMin(a, b):
                if False:
                    print('Hello World!')
                return torch.clamp(a + b, max=2)

            def funcOptMax(a, b):
                if False:
                    return 10
                return torch.clamp(a + b, min=0)
            a = torch.randn(4, 4, dtype=torch.float, device=device, requires_grad=True)
            b = torch.randn(4, 4, dtype=torch.float, device=device)
            nan = torch.tensor(float('nan'), dtype=torch.float, device=device)
            funcs = (func2, funcInf, funcNegInf, funcOptMin, funcOptMax)
            for (f, inputs) in product(funcs, [[a, b], [a, nan]]):
                (inp1, inp2) = inputs
                s = self.checkScript(f, (inp1, inp2), profiling=ProfilingMode.PROFILING)
                self.assertAllFused(s.graph_for(inp1, inp2), except_for={'aten::size', 'aten::_size_if_not_equal'})
                c = s(inp1, inp2)
                with enable_profiling_mode_for_profiling_tests():
                    warmup_backward(c.sum())
                graph = backward_graph(s)
                self.assertAllFused(graph, except_for={'aten::Float', 'aten::_grad_sum_to_size'}.union(autograd_check_set))

    def test_clamp_double(self):
        if False:
            for i in range(10):
                print('nop')
        for device in self.devices:

            def clamp_double(x, eta: float):
                if False:
                    while True:
                        i = 10
                return 1 - x.clamp(eta, 1 - eta)
            x = torch.tensor([1.0, 1.0], dtype=torch.double, device=device)
            eta = 1e-09
            s = self.checkScript(clamp_double, (x, eta), profiling=ProfilingMode.PROFILING, atol=1e-10, rtol=1e-05)
            self.assertAllFused(s.graph_for(x, eta), except_for={'aten::sub'})

    def test_clamp_int(self):
        if False:
            for i in range(10):
                print('nop')
        for device in self.devices:

            def clamp_int(x, eta: int):
                if False:
                    for i in range(10):
                        print('nop')
                return x.clamp(0, eta)
            x = torch.tensor([1, 1], device=device)
            eta = 1 << 32
            s = self.checkScript(clamp_int, (x, eta), profiling=ProfilingMode.PROFILING)
            self.assertAllFused(s.graph_for(x, eta))

    def test_add_bool(self):
        if False:
            print('Hello World!')
        sizes = [(1,), (2,), (4, 4)]
        for (device, size) in product(self.devices, sizes):

            def f(x, y, z):
                if False:
                    while True:
                        i = 10
                return x + y + z
            x = torch.randint(0, 2, size, dtype=torch.bool, device=device)
            y = torch.randint(0, 2, size, dtype=torch.bool, device=device)
            z = torch.randint(0, 2, size, dtype=torch.bool, device=device)
            ge = self.checkTrace(f, (x, y, z), inputs_require_grads=False)
            self.assertAllFused(ge.graph_for(x, y, z))

    def test_mul_bool(self):
        if False:
            i = 10
            return i + 15
        for device in self.devices:

            def f(x, y, z):
                if False:
                    return 10
                return x * y * z
            x = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            y = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            z = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            ge = self.checkTrace(f, (x, y, z), inputs_require_grads=False)
            self.assertAllFused(ge.graph_for(x, y, z))

    def test_div_bool(self):
        if False:
            i = 10
            return i + 15
        for device in self.devices:

            def f(x, y, z):
                if False:
                    return 10
                return (x + y) / z
            x = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            y = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            z = torch.ones_like(x, dtype=torch.bool, device=device)
            ge = self.checkTrace(f, (x, y, z), inputs_require_grads=False)
            self.assertAllFused(ge.graph_for(x, y, z))

    def test_bitwise_ops(self):
        if False:
            i = 10
            return i + 15

        def apply(fn):
            if False:
                while True:
                    i = 10
            return lambda x, y, z: fn(fn(x, y), z)
        binary_ops = [operator.__and__, operator.__or__, operator.__xor__, operator.__lshift__, operator.__rshift__]
        devices = self.devices
        for (dtype, op, device) in product(self.int_dtypes, binary_ops, devices):
            try:
                x = self.data_for(dtype, device)
                y = self.data_for(dtype, device)
                z = self.data_for(dtype, device)
                fn = apply(op)
                ref = fn(x, y, z)
            except Exception:
                continue
            try:
                t = torch.jit.trace(fn, (x, y, z))
                self.assertEqual(ref, t(x, y, z))
                self.assertAllFused(t.graph_for(x, y, z))
            except Exception as e:
                raise RuntimeError(' '.join(['Failed:', str(dtype), op.__name__, device])) from e

    def test_minmax_int_ops(self):
        if False:
            for i in range(10):
                print('nop')

        def apply(fn):
            if False:
                while True:
                    i = 10
            return lambda x, y, z: fn(fn(x, y), z)
        binary_ops = [torch.min, torch.max]
        devices = self.devices
        for (dtype, op, device) in product(self.int_dtypes, binary_ops, devices):
            try:
                x = self.data_for(dtype, device)
                y = self.data_for(dtype, device)
                z = self.data_for(dtype, device)
                fn = apply(op)
                ref = fn(x, y, z)
            except Exception:
                continue
            try:
                t = torch.jit.trace(fn, (x, y, z))
                self.assertEqual(ref, t(x, y, z))
                self.assertAllFused(t.graph_for(x, y, z))
            except Exception as e:
                raise RuntimeError(' '.join(['Failed:', str(dtype), op.__name__, device])) from e

    def test_comparison_eq_ne(self):
        if False:
            print('Hello World!')
        for device in self.devices:

            def f(x, y):
                if False:
                    for i in range(10):
                        print('nop')
                mask = (x == 0).type_as(x)
                z = x * mask + y
                mask = (x != 0).type_as(x)
                z = z * mask + y
                return z
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)
            ge = self.checkTrace(f, (x, y))
            self.assertAllFused(ge.graph_for(x, y))

    @staticmethod
    def fn_test_comparison_gt_lt(x, y):
        if False:
            i = 10
            return i + 15
        mask = (x > 0).type_as(x)
        z = x * mask + y
        mask = (x < 0).type_as(x)
        z = z * mask + y
        return z

    def test_comparison_gt_lt(self):
        if False:
            print('Hello World!')
        for device in self.devices:
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)
            ge = self.checkTrace(self.fn_test_comparison_gt_lt, (x, y))
            self.assertAllFused(ge.graph_for(x, y))

    def test_comparison_ge_le(self):
        if False:
            while True:
                i = 10
        for device in self.devices:

            def f(x, y):
                if False:
                    for i in range(10):
                        print('nop')
                mask = (x >= 0).type_as(x)
                z = x * mask + y
                mask = (x <= 0).type_as(x)
                z = z * mask + y
                return z
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)
            ge = self.checkTrace(f, (x, y))
            self.assertAllFused(ge.graph_for(x, y))
            x.requires_grad_(True)
            y.requires_grad_(True)
            self.assertAllFused(ge.graph_for(x, y), except_for=('aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'))

    def test_addcmul(self):
        if False:
            while True:
                i = 10
        for device in self.devices:
            t = torch.randn(1, 4, dtype=torch.float, device=device)
            t1 = torch.randn(4, 1, dtype=torch.float, device=device)
            t2 = torch.randn(1, 4, dtype=torch.float, device=device)

            def foo(t, t1, t2):
                if False:
                    i = 10
                    return i + 15
                return t.addcmul(t + 1, t2, value=0.1)
            ge = self.checkTrace(foo, (t, t1, t2), allow_unused=True)
            graph = ge.graph_for(t, t1, t2)
            fusion_groups = self.findFusionGroups(graph)
            self.assertEqual(len(fusion_groups), 1)
            FileCheck().check('aten::add(').check('aten::addcmul(').run(str(fusion_groups[0]))

    def test_lerp(self):
        if False:
            i = 10
            return i + 15
        for device in self.devices:
            start = torch.randn(4, 1, dtype=torch.float, device=device)
            end = torch.randn(1, 4, dtype=torch.float, device=device)
            weight = torch.tensor(0.5, dtype=torch.float, device=device)

            def foo_weight_scalar(start, end):
                if False:
                    return 10
                return torch.lerp(start + 1, end, 0.5)

            def foo_weight_tensor(start, end):
                if False:
                    while True:
                        i = 10
                return torch.lerp(start + 1, end, weight)
            ge_weight_scalar = self.checkTrace(foo_weight_scalar, (start, end))
            graph = ge_weight_scalar.graph_for(start, end)
            self.assertAllFused(graph)

    def test_concat(self):
        if False:
            i = 10
            return i + 15
        with set_fusion_group_inlining(True):
            for device in self.devices:
                hx = torch.randn(3, 20, dtype=torch.float, device=device)
                cx = torch.randn(3, 20, dtype=torch.float, device=device)

                def foo(hx, cx):
                    if False:
                        print('Hello World!')
                    return torch.cat((hx + cx, hx * cx))
                ge = self.checkTrace(foo, (hx, cx))
                graph = ge.graph_for(hx, cx)
                self.assertAllFused(graph)

    def test_remove_output_used_only_in_size(self):
        if False:
            while True:
                i = 10
        for device in self.devices:

            def test_fuse(a, b):
                if False:
                    while True:
                        i = 10
                c = a + b
                d = c + b
                return d
            scripted_f = torch.jit.script(test_fuse)
            x = torch.ones(1, requires_grad=True, device=device)
            y = torch.ones(1, requires_grad=True, device=device)
            warmup_forward(scripted_f, x, y, profiling_count=3)
            g = scripted_f.graph_for(x, y)
            diff_nodes = g.findAllNodes('prim::DifferentiableGraph')
            self.assertEqual(len(diff_nodes), 1)
            g = diff_nodes[0].g('Subgraph')
            if_nodes = [n for n in g.nodes() if n.kind() == 'prim::If']
            self.assertEqual(len(if_nodes), 1)
            self.assertEqual(len(list(if_nodes[0].outputs())), 1)

    def test_concat_invariant(self):
        if False:
            while True:
                i = 10
        for device in self.devices:

            def fn(x, y, z):
                if False:
                    i = 10
                    return i + 15
                x1 = x + y
                y1 = x - y
                w = torch.cat([x1, y1])
                return w + z
            x = torch.randn(2, 2, dtype=torch.float, device=device)
            y = torch.randn(2, 2, dtype=torch.float, device=device)
            z = torch.randn(4, 2, dtype=torch.float, device=device)
            ge = self.checkTrace(fn, (x, y, z))
            graph = ge.graph_for(x, y, z)
            self.assertAllFused(graph, except_for={'aten::add'})

    @staticmethod
    def fn_test_exp(x, y):
        if False:
            while True:
                i = 10
        return (x + 0.5 * y).exp()

    def test_exp(self):
        if False:
            return 10
        for device in self.devices:
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)
            ge = self.checkTrace(self.fn_test_exp, (x, y))
            self.assertAllFused(ge.graph_for(x, y))

    def test_threshold(self):
        if False:
            print('Hello World!')
        for device in self.devices:

            def f(x):
                if False:
                    while True:
                        i = 10
                return torch.threshold(x, 0, -10) + x + x + x
            x = torch.tensor([-1, -0.5, 0, 1, 2, 3], device=device)
            scripted = self.checkScript(f, (x,))
            self.assertAllFused(scripted.graph_for(x))

    def test_scalar_arg(self):
        if False:
            while True:
                i = 10
        for device in self.devices:

            def fn_test_scalar_arg(x: torch.Tensor, p: float) -> torch.Tensor:
                if False:
                    print('Hello World!')
                return p * (x * x + x)
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            p = 3
            scripted = self.checkScript(fn_test_scalar_arg, (x, p))
            self.assertAllFused(scripted.graph_for(x, p))
            x.requires_grad_(True)

            def fn_test_scalar_arg_requires_grad(x: torch.Tensor, p: float) -> torch.Tensor:
                if False:
                    i = 10
                    return i + 15
                return p * (x * x + x)
            scripted = torch.jit.script(fn_test_scalar_arg_requires_grad)
            out = scripted(x, p)
            out = scripted(x, p)
            out = scripted(x, p)
            self.assertAllFused(scripted.graph_for(x, p), except_for=('aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, 'needs non-zero device')
    def test_fusion_reuse_multi_gpu(self):
        if False:
            print('Hello World!')

        def fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x * y * x * y
        inputs_cpu = [torch.randn(4, 4, dtype=torch.float), torch.randn(4, 4, dtype=torch.float)]
        inputs_cuda0 = [x.cuda(0) for x in inputs_cpu]
        inputs_cuda1 = [y.cuda(1) for y in inputs_cpu]
        ge = self.checkScript(fn, inputs_cpu)
        self.assertAllFused(ge.graph_for(*inputs_cpu))
        ge(*inputs_cuda0)
        ge(*inputs_cuda1)

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, 'needs non-zero device')
    def test_kernel_cache_multi_gpu(self):
        if False:
            return 10

        def not_fusible(x):
            if False:
                while True:
                    i = 10
            return x

        def fn(x, y, z):
            if False:
                for i in range(10):
                    print('nop')
            x_out = x * x * x * x * x
            y_out = y * y * y * y * y
            z_out = z * z * z * z * z
            return (not_fusible(x_out), not_fusible(y_out), not_fusible(z_out))
        inputs = [torch.randn(4, 4, dtype=torch.float), torch.randn(4, 4, dtype=torch.float, device='cuda:0'), torch.randn(4, 4, dtype=torch.float, device='cuda:1')]
        prev_cache_size = torch._C._jit_debug_fuser_num_cached_kernel_specs()
        ge = self.checkScript(fn, inputs)
        self.assertGraphContainsExactly(ge.graph_for(*inputs), FUSION_GROUP, 3, True)
        new_cache_size = torch._C._jit_debug_fuser_num_cached_kernel_specs()

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, 'needs non-zero device')
    def test_nonzero_device_cuda(self):
        if False:
            return 10
        device = 'cuda:' + str(1)
        x = torch.tensor([0.4], dtype=torch.float, device=device)
        y = torch.tensor([0.7], dtype=torch.float, device=device)

        def doit(x, y):
            if False:
                return 10
            return torch.sigmoid(torch.tanh(x * (x + y) + x))
        ge = self.checkTrace(doit, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    def test_lstm(self):
        if False:
            while True:
                i = 10
        for device in self.devices:
            inputs = get_lstm_inputs(device, training=True)
            module = self.checkScript(LSTMCellS, inputs)
            self.assertAllFused(module.graph_for(inputs), except_for={'prim::TupleConstruct'})

    def test_lstm_concat(self):
        if False:
            while True:
                i = 10
        with set_fusion_group_inlining(True):
            for device in self.devices:
                inputs = get_lstm_inputs(device)
                ge = self.checkTrace(LSTMCellC, inputs)
                graph = ge.graph_for(*inputs)
                except_nodes = {'prim::TupleConstruct', 'aten::linear'}
                if self.dynamic_shapes:
                    except_nodes = except_nodes.union({'aten::add', 'prim::ConstantChunk'})
                self.assertAllFused(ge.graph_for(*inputs), except_for=except_nodes)

    def test_lstm_gates_permutations(self):
        if False:
            print('Hello World!')
        for device in self.devices:
            choices = ['x.mm(w_ih.t())', 'hx.mm(w_hh.t())', 'b_ih', 'b_hh']
            template = dedent('\n            def cell(x, hx, cx, w_ih, w_hh, b_ih, b_hh):\n                gates = {} + {} + {} + {}\n                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)\n                return ingate * forgetgate * cellgate * outgate\n            ')
            for permutation in permutations(choices, len(choices)):
                code = template.format(*permutation)
                scope = {}
                exec(code, globals(), scope)
                cu = torch.jit.CompilationUnit(code)
                fusion_group_len = 2 if self.dynamic_shapes else 1
                inputs = get_lstm_inputs(device, training=False)
                self.assertEqual(cu.cell(*inputs), scope['cell'](*inputs))
                forward_graph = cu.cell.graph_for(*inputs)
                self.assertGraphContainsExactly(forward_graph, FUSION_GROUP, fusion_group_len)

    def test_lstm_traced(self):
        if False:
            while True:
                i = 10
        for device in self.devices:
            inputs = get_lstm_inputs(device)
            ge = self.checkTrace(LSTMCellF, inputs)
            graph = ge.graph_for(*inputs)
            fusion_groups = self.findFusionGroups(graph)
            fusion_group_len = 2 if self.dynamic_shapes else 1
            self.assertEqual(len(fusion_groups), fusion_group_len)
            f = FileCheck()
            if not self.dynamic_shapes:
                f.check('Chunk')
            f.check('aten::sigmoid').check('aten::tanh').run(str(fusion_groups[0 if not self.dynamic_shapes else 1]))

    def test_milstm(self):
        if False:
            for i in range(10):
                print('nop')
        if self.dynamic_shapes:
            self.skipTest("don't run conv with dynamic shapes")
        for device in self.devices:
            inputs = get_milstm_inputs(device, training=True)
            module = self.checkScript(MiLSTMCell, inputs)
            forward_graph = module.graph_for(*inputs)
            fusion_group_len = 2 if self.dynamic_shapes else 1
            self.assertGraphContainsExactly(forward_graph, FUSION_GROUP, fusion_group_len, consider_subgraphs=True)
            FileCheck().check('DifferentiableGraph').check('TupleConstruct').check_next('return').check(FUSION_GROUP).run(str(forward_graph))
            (hy, cy) = module(*inputs)
            warmup_backward((hy + cy).sum())

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    @unittest.skip('rand_like is not supported yet')
    def test_rand_cuda(self):
        if False:
            print('Hello World!')

        class M(torch.jit.ScriptModule):
            __constants__ = ['d']

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.d = torch.device('cuda')

            @torch.jit.script_method
            def create(self, x):
                if False:
                    print('Hello World!')
                return x * x + x + torch.rand_like(x)
        x = torch.zeros([3, 4, 5], dtype=torch.float, device='cuda')
        m = M()
        out1 = m.create(x)
        out2 = m.create(x)
        self.assertNotEqual(out1, out2)
        self.assertTrue(torch.all(out1 >= 0))
        self.assertTrue(torch.all(out1 < 1))
        self.assertTrue(torch.all(out2 >= 0))
        self.assertTrue(torch.all(out2 < 1))
        self.assertAllFused(m.create.graph_for(x))

    @staticmethod
    def fn_test_relu(x, y):
        if False:
            for i in range(10):
                print('nop')
        return F.relu(x + 0.5 * y)

    def test_relu(self):
        if False:
            return 10
        for device in self.devices:
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)
            ge = self.checkTrace(self.fn_test_relu, (x, y))
            self.assertAllFused(ge.graph_for(x, y))

    def test_erf(self):
        if False:
            print('Hello World!')
        for device in self.devices:
            if device == 'cpu':
                continue

            def fn_test_erf(x):
                if False:
                    i = 10
                    return i + 15
                return F.relu(torch.erf(x) - torch.erfc(x))
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            ge = self.checkScript(fn_test_erf, (x,), profiling=ProfilingMode.PROFILING)
            self.assertAllFused(ge.graph_for(x))
            x.requires_grad_(True)
            ge = self.checkScript(fn_test_erf, (x,), profiling=ProfilingMode.PROFILING)
            self.assertAllFused(ge.graph_for(x), except_for=('aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    @unittest.skip('rand_like is not supported yet')
    def test_rand_broadcast_cuda(self):
        if False:
            i = 10
            return i + 15

        def fn_test_rand(x, y):
            if False:
                while True:
                    i = 10
            r = torch.rand_like(y)
            return r * x + x

        def fn_test_rand2(x, y):
            if False:
                i = 10
                return i + 15
            r = torch.rand_like(y)
            return r * x * x
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
        script_f = torch.jit.script(fn_test_rand)
        warmup_forward(script_f, x, y)
        out = script_f(x, y)
        self.assertAllFused(script_f.graph_for(x, y))
        x.requires_grad_(True)
        out = script_f(x, y)
        self.assertAllFused(script_f.graph_for(x, y), except_for=('aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'))
        x = torch.ones(4, 4, dtype=torch.float, device='cuda')
        y = torch.ones(4, dtype=torch.float, device='cuda')
        script_f = torch.jit.script(fn_test_rand2)
        warmup_forward(script_f, x, y)
        out = script_f(x, y)
        self.assertEqual(out[0, :] + torch.zeros(4, 4, device='cuda'), out)

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    @unittest.skip('rand_like is not supported yet')
    def test_rand_diamond(self):
        if False:
            return 10

        def fn_test_diamond(x, y):
            if False:
                for i in range(10):
                    print('nop')
            r = torch.rand_like(y)
            a = x + r
            b = y - r
            return a + b
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
        script_f = torch.jit.script(fn_test_diamond)
        warmup_forward(script_f, x, y)
        out = script_f(x, y)
        self.assertEqual(out, x + y)

    def test_scalar(self):
        if False:
            i = 10
            return i + 15

        def fn(x, y):
            if False:
                print('Hello World!')
            return 2 * x + y
        x = torch.tensor(0.1, dtype=torch.float, device='cpu')
        y = torch.tensor(1, dtype=torch.float, device='cpu')
        ge = self.checkScript(fn, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    def test_inlined_optimized_graph(self):
        if False:
            return 10

        @torch.jit.script
        def foo(x):
            if False:
                while True:
                    i = 10
            return torch.relu(x + x)
        for _ in range(3):
            foo(torch.rand([4, 4]))
        for _ in range(3):
            foo(torch.rand([10]))
        for _ in range(3):
            foo(torch.rand([2, 2, 2]))
        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check_count('prim::If', 1, exactly=True).check('prim::TensorExpr').run(g)
        torch._C._jit_pass_inline(g)
        f = FileCheck()
        for _ in range(3):
            f.check('prim::If').check('prim::TensorExpr')
        f.run(g)

    def test_small_constant(self):
        if False:
            for i in range(10):
                print('nop')
        for device in self.devices:

            def fn_test_small_constant(x, y):
                if False:
                    return 10
                return (1e-08 * x + 5e-09 * y) * 100000000.0
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)
            ge = self.checkTrace(fn_test_small_constant, (x, y))
            self.assertAllFused(ge.graph_for(x, y))

    def test_tensor_scalar_ops(self):
        if False:
            print('Hello World!')
        for device in self.devices:

            def should_fuse(x):
                if False:
                    print('Hello World!')
                z = 3.0
                y = x + z
                return x * y

            def should_fuse_scalar(x, z):
                if False:
                    i = 10
                    return i + 15
                y = x + int(z)
                return x * y
            inputs = [torch.randn(2, 2, dtype=torch.float, device=device)]
            ge = self.checkScript(should_fuse, inputs)
            graph = ge.graph_for(*inputs)
            fusion_groups = self.findFusionGroups(graph)
            self.assertEqual(len(fusion_groups), 1)
            FileCheck().check('aten::add').check('aten::mul').run(str(fusion_groups[0]))
            inputs = [torch.randn(2, 2, dtype=torch.float, device=device), torch.tensor(3.0, dtype=torch.float, device=device)]
            ge = self.checkScript(should_fuse_scalar, inputs)
            inputs = [torch.randn(2, 2, dtype=torch.float, device=device), torch.tensor(7.0, dtype=torch.float, device=device)]
            self.assertEqual(ge(*inputs), should_fuse_scalar(*inputs))
            self.assertGraphContainsExactly(ge.graph_for(*inputs), FUSION_GROUP, 1, consider_subgraphs=True)

    def test_where_and_typing(self):
        if False:
            print('Hello World!')
        for device in self.devices:

            def f(x, y):
                if False:
                    i = 10
                    return i + 15
                mask = x > y
                res = torch.where(mask, x, y)
                return (mask, res)
            x = torch.randn(4, 4, dtype=torch.double, device=device)
            y = torch.randn(4, 4, dtype=torch.double, device=device)
            script_f = self.checkScript(f, (x, y))
            self.assertAllFused(script_f.graph_for(x, y), except_for={'prim::TupleConstruct'})

    def test_disabled(self):
        if False:
            return 10
        old_cpu_fuser_state = torch._C._jit_can_fuse_on_cpu()
        torch._C._jit_override_can_fuse_on_cpu(False)

        def fn(a):
            if False:
                for i in range(10):
                    print('nop')
            return a ** 2 + a
        x = torch.randn(4, dtype=torch.float, device='cpu')
        s = self.checkScript(fn, (x,))
        g = s.graph_for(x)
        self.assertEqual(len(self.findFusionGroups(g)), 0)
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuser_state)

    def data_for(self, dtype, device='cuda', size=None):
        if False:
            i = 10
            return i + 15
        if size is None:
            v = torch.arange(1, 3, dtype=torch.float, device=device)
        else:
            v = torch.rand(*size, device=device)
        if dtype == torch.bool:
            return v > 2
        elif dtype in [torch.qint8, torch.quint8, torch.qint32]:
            return torch.quantize_per_tensor(v, 0.1, 1, dtype=dtype)
        else:
            return v.to(dtype)

    def test_torch_to(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.to(torch.float)
        foo(torch.tensor([3.0], dtype=torch.float))
        foo(torch.tensor([3.0], dtype=torch.float))
        FileCheck().check_not('TensorExpr').run(torch.jit.last_executed_optimized_graph())

        @torch.jit.script
        def foo(x, dtype: int):
            if False:
                while True:
                    i = 10
            return x.to(dtype)
        foo(torch.tensor([3.0], dtype=torch.float), torch.int)
        foo(torch.tensor([3.0], dtype=torch.float), torch.int)
        FileCheck().check_not('TensorExpr').run(torch.jit.last_executed_optimized_graph())

        @torch.jit.script
        def foo(x, dtype: int):
            if False:
                print('Hello World!')
            return x.to(pin_memory=True)
        foo(torch.tensor([3.0], dtype=torch.float), torch.int)
        foo(torch.tensor([3.0], dtype=torch.float), torch.int)
        FileCheck().check_not('TensorExpr').run(torch.jit.last_executed_optimized_graph())
        if torch.cuda.is_available():

            @torch.jit.script
            def foo(x):
                if False:
                    while True:
                        i = 10
                return x.to(device='cuda')
            foo(torch.tensor([3.0], dtype=torch.float))
            foo(torch.tensor([3.0], dtype=torch.float))
            FileCheck().check_not('TensorExpr').run(torch.jit.last_executed_optimized_graph())
        sizes = [(1, 4), (4, 4)]
        dtypes = [torch.bool, torch.int, torch.float16, torch.float32, torch.float64]

        class MyMod(torch.nn.Module):

            def __init__(self, dtype):
                if False:
                    return 10
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x.to(self.dtype)
        bad_dtypes = []
        for (dtype, output_dtype, device, size) in product(dtypes, dtypes, self.devices, sizes):
            if dtype in [torch.float16, torch.bfloat16] and device == 'cpu':
                continue
            if dtype == output_dtype:
                continue
            x = self.data_for(dtype, device, size=size)
            mod = MyMod(output_dtype)
            ref = mod.forward(x)
            mod = torch.jit.freeze(torch.jit.script(mod.eval()))
            warmup_forward(mod.forward, x)
            self.assertEqual(ref, mod.forward(x))
            self.assertLastGraphAllFused()

    @unittest.skip('Temporarily disabled')
    def test_masked_fill(self):
        if False:
            i = 10
            return i + 15
        dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.float32, torch.float64, torch.bool]
        sizes = [(2,), (4, 4)]
        for (self_dtype, device, scalar_val, size) in product(dtypes, self.devices, [0.4, 3], sizes):
            input_v = self.data_for(self_dtype, device, size=size)
            mask = self.data_for(torch.bool, device, size=size)

            def fn(input_v, mask):
                if False:
                    return 10
                return torch.masked_fill(input_v, mask, scalar_val)
            ref = fn(input_v, mask)
            try:
                t = torch.jit.trace(fn, (input_v, mask))
                torch.testing.assert_close(ref, t(input_v, mask))
                self.assertLastGraphAllFused()
            except Exception as e:
                raise RuntimeError(' '.join(['Failed:', str(self_dtype), op.__name__, device, str(size)])) from e

    def test_isnan(self):
        if False:
            print('Hello World!')
        x = torch.rand([4])
        x[0] = float('nan')
        inputs = [x, torch.tensor([float('nan'), 0.5])]
        dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64, torch.bool]
        for (inp, device, dtype) in product(inputs, self.devices, dtypes):
            if dtype in [torch.float16, torch.bfloat16] and device == 'cpu':
                continue
            inp = inp.to(device=device, dtype=dtype)
            try:
                f = torch.jit.trace(lambda x: x.isnan(), (inp,))
                warmup_forward(f, inp)
                self.assertEqual(f(inp), inp.isnan())
                self.assertLastGraphAllFused()
            except Exception as e:
                raise RuntimeError(' '.join(['Failed:', str(dtype), 'isnan', device])) from e

    def test_gelu(self):
        if False:
            for i in range(10):
                print('nop')

        def apply(fn):
            if False:
                return 10
            return lambda x, approximate: fn(x, approximate)
        unary_ops = [F.gelu]
        sizes = [(1,), (2,), (4, 4)]
        for (dtype, op, device, size) in product(self.dtypes, unary_ops, self.devices, sizes):
            if dtype in [torch.float16, torch.bfloat16] and device == 'cpu':
                continue
            try:
                x = self.data_for(dtype, device, size=size)
                cond = self.data_for(torch.bool, device)
                fn = apply(op)
                ref = fn(x, cond)
            except Exception:
                continue
            try:
                t = torch.jit.trace(fn, (x, cond))
                torch.testing.assert_close(ref, t(x, cond))
                self.assertAllFused(t.graph_for(x, cond))
            except Exception as e:
                raise RuntimeError(' '.join(['Failed:', str(dtype), op.__name__, device, str(size)])) from e

    def test_unary_ops(self):
        if False:
            while True:
                i = 10
        with torch._jit_internal._disable_emit_hooks():

            def apply(fn):
                if False:
                    while True:
                        i = 10
                return lambda x: fn(x)
            unary_ops = [torch.lgamma, torch.sigmoid, torch.reciprocal, torch.neg, torch.relu, F.relu6, torch.log, torch.log10, torch.log1p, torch.log2, torch.exp, torch.expm1, torch.erf, torch.erfc, torch.cos, torch.sin, torch.tan, torch.acos, torch.asin, torch.cosh, torch.sinh, torch.atan, torch.tanh, F.hardtanh, F.hardsigmoid, F.hardswish, F.softplus, F.silu, F.mish, F.elu, torch.sqrt, torch.rsqrt, torch.abs, torch.frac, F.leaky_relu, lambda x: torch.threshold(x, 0, -10)]
            gpu_only = {torch.erf, torch.erfc}
            sizes = [(1,), (2,), (4, 4)]
            for (dtype, op, device, size) in product(self.dtypes, unary_ops, self.devices, sizes):
                if dtype in [torch.float16, torch.bfloat16] and device == 'cpu':
                    continue
                if dtype == torch.bfloat16 and op == torch.round:
                    continue
                if op in gpu_only and device == 'cpu':
                    continue
                try:
                    x = self.data_for(dtype, device, size=size)
                    fn = apply(op)
                    ref = fn(x)
                except Exception:
                    continue
                try:
                    t = torch.jit.trace(fn, (x,))
                    torch.testing.assert_close(ref, t(x))
                    self.assertAllFused(t.graph_for(x))
                except Exception as e:
                    raise RuntimeError(' '.join(['Failed:', str(dtype), op.__name__, device, str(size)])) from e

    def test_binary_ops(self):
        if False:
            for i in range(10):
                print('nop')

        def apply(fn):
            if False:
                i = 10
                return i + 15
            return lambda x, y: fn(x, y)
        binary_ops = [operator.__and__, operator.__or__, operator.__xor__, torch.add, torch.sub, torch.mul, torch.min, torch.max, lambda x, y: torch.lerp(x, y, 0.5), torch.atan2, torch.div, torch.eq, torch.ne, torch.ge, torch.gt, torch.lt, torch.fmod, torch.remainder, lambda x, y: y.type_as(x)]
        fp_only = [torch.fmod, torch.remainder]
        devices = self.devices
        for (dtype, op, device) in product(self.dtypes, binary_ops, devices):
            if dtype in [torch.float16, torch.bfloat16] and device == 'cpu':
                continue
            try:
                x = self.data_for(dtype, device)
                y = self.data_for(dtype, device)
                fn = apply(op)
                ref = fn(x, y)
            except Exception:
                continue
            try:
                t = torch.jit.trace(fn, (x, y))
                self.assertEqual(ref, t(x, y))
                if op not in fp_only or dtype.is_floating_point:
                    self.assertAllFused(t.graph_for(x, y))
            except Exception as e:
                raise RuntimeError(' '.join(['Failed:', str(dtype), op.__name__, device])) from e

    def test_binary_scalar_ops(self):
        if False:
            while True:
                i = 10

        def apply(fn):
            if False:
                i = 10
                return i + 15
            return lambda x, y: fn(x, y)
        ir_template = '\n        graph(%x : {dtype_x}, %y : {dtype_y}):\n          %z = {op}(%x, %y)\n          return (%z)'
        binary_ops = ['aten::mul', 'aten::add', 'aten::sub', 'aten::div', 'aten::lt', 'aten::le', 'aten::eq', 'aten::ne', 'aten::gt', 'aten::ge', 'aten::__or__', 'aten::__xor__', 'aten::__and__', 'aten::__lshift__', 'aten::__rshift__']
        dtypes = ['int', 'float', 'bool']
        values = {'int': [10, 3], 'float': [12.34, 2.78], 'bool': [True, False]}
        devices = self.devices
        for (dtype_x, dtype_y, op, device) in product(dtypes, dtypes, binary_ops, devices):
            code = ir_template.format(**locals())
            try:
                graph = torch._C.parse_ir(code)
                for (x, y) in product(values[dtype_x], values[dtype_y]):
                    ref = torch._C._jit_interpret_graph(graph, (x, y))
            except Exception:
                continue
            try:
                k = torch._C._te.TensorExprKernel(graph)
            except Exception as e:
                raise RuntimeError(' '.join(['Compilation failed:', device, str(code)])) from e
            for (x, y) in product(values[dtype_x], values[dtype_y]):
                ref = torch._C._jit_interpret_graph(graph, (x, y))
                try:
                    res = k.run((x, y))
                    self.assertEqual(ref, res)
                except Exception as e:
                    raise RuntimeError(' '.join(['Failed at runtime:', device, str(x), str(y), str(code)])) from e

    def test_matmul(self):
        if False:
            for i in range(10):
                print('nop')
        if self.dynamic_shapes:
            self.skipTest("don't run conv with dynamic shapes")

        def fn(x, y):
            if False:
                i = 10
                return i + 15
            return torch.matmul(x, y)
        devices = ['cpu']
        sizes = [[[128, 128], [128, 128]], [[10, 10], [10, 10]], [[1, 16], [16, 128]], [[128], [128]], [[128], [128, 128]], [[3], [3]], [[3, 4], [4]], [[10, 3, 4], [4]], [[10, 3, 4], [10, 4, 5]], [[10, 3, 4], [4, 5]]]
        skip_is_fused_check_sizes = ['[[128], [128]]', '[[128], [128, 128]]', '[[3], [3]]', '[[3, 4], [4]]', '[[10, 3, 4], [4]]', '[[10, 3, 4], [10, 4, 5]]', '[[10, 3, 4], [4, 5]]']
        for (dtype, size, device) in product(self.dtypes, sizes, devices):
            if dtype in [torch.float16, torch.bfloat16] and device == 'cpu':
                continue
            try:
                (size_x, size_y) = size
                x = self.data_for(dtype, device, size=size_x)
                y = self.data_for(dtype, device, size=size_y)
                ref = fn(x, y)
            except Exception as e:
                continue
            try:
                t = torch.jit.trace(fn, (x, y))
                t(x, y)
                self.assertEqual(ref, t(x, y))
                if str(size) not in skip_is_fused_check_sizes:
                    self.assertAllFused(t.graph_for(x, y))
            except Exception as e:
                raise RuntimeError(' '.join(['Failed:', str(dtype), device])) from e

    def test_binary_tensor_scalar_ops(self):
        if False:
            while True:
                i = 10
        with torch._jit_internal._disable_emit_hooks():

            def apply_with_scalar(fn, scalar):
                if False:
                    while True:
                        i = 10
                return lambda x: fn(x, scalar)
            binary_ops = [operator.__and__, operator.__or__, operator.__xor__, torch.add, torch.sub, torch.mul, torch.eq, torch.ne, torch.ge, torch.lt, torch.gt]
            devices = self.devices
            scalars = [1.5, 3, 0, -2.0, -1]
            for (dtype, op, device, scalar) in product(self.dtypes, binary_ops, devices, scalars):
                if dtype in [torch.float16, torch.bfloat16] and device == 'cpu':
                    continue
                try:
                    x = self.data_for(dtype, device)
                    fn = apply_with_scalar(op, scalar)
                    ref = fn(x)
                except Exception:
                    continue
                try:
                    t = torch.jit.trace(fn, x)
                    self.assertEqual(ref, t(x))
                    self.assertAllFused(t.graph_for(x))
                except Exception as e:
                    raise RuntimeError(' '.join(['Failed:', str(dtype), op.__name__, device])) from e

    def test_binary_div_ops(self):
        if False:
            i = 10
            return i + 15

        def apply_with_scalar(fn, scalar):
            if False:
                i = 10
                return i + 15
            return lambda x: fn(x, scalar)
        binary_ops = [torch.div, torch.remainder, torch.fmod]
        devices = self.devices
        scalars = [1.5, 3, -2.0, -1]
        for (dtype, op, device, scalar) in product(self.dtypes, binary_ops, devices, scalars):
            if dtype in [torch.float16, torch.bfloat16] and device == 'cpu':
                continue
            try:
                x = self.data_for(dtype, device)
                fn = apply_with_scalar(op, scalar)
                ref = fn(x)
            except Exception:
                continue
            try:
                t = torch.jit.trace(fn, x)
                self.assertEqual(ref, t(x))
            except Exception as e:
                raise RuntimeError(f'Failed: {dtype} {op.__name__} {device} {scalar}') from e

    def test_binary_pow(self):
        if False:
            return 10

        def apply_with_scalar(fn, scalar):
            if False:
                print('Hello World!')
            return lambda x: fn(x, scalar)
        dtypes = [torch.float32, torch.float64]
        binary_ops = [torch.pow]
        scalars = [1.5, 3, 0, -2.0, -1]
        for (dtype, op, device, scalar) in product(dtypes, binary_ops, self.devices, scalars):
            if dtype in [torch.float16, torch.bfloat16] and device == 'cpu':
                continue
            try:
                x = self.data_for(dtype, device)
                fn = apply_with_scalar(op, scalar)
                ref = fn(x)
            except Exception:
                continue
            try:
                t = torch.jit.trace(fn, x)
                self.assertEqual(ref, t(x))
                self.assertAllFused(t.graph_for(x))
            except Exception as e:
                raise RuntimeError(' '.join(['Failed:', str(dtype), op.__name__, device])) from e

    def test_ternary_ops(self):
        if False:
            return 10

        def apply(fn):
            if False:
                i = 10
                return i + 15
            return lambda x, y, z: fn(x, y, z)
        ternary_ops = [torch.lerp, torch.addcmul]
        devices = self.devices
        for (dtype, op, device) in product(self.dtypes, ternary_ops, devices):
            if dtype in [torch.float16, torch.bfloat16] and device == 'cpu':
                continue
            try:
                x = self.data_for(dtype, device)
                y = self.data_for(dtype, device)
                z = self.data_for(dtype, device)
                fn = apply(op)
                ref = fn(x, y, z)
            except Exception:
                continue
            try:
                t = torch.jit.trace(fn, (x, y, z))
                self.assertEqual(ref, t(x, y, z))
                self.assertAllFused(t.graph_for(x, y, z))
            except Exception as e:
                raise RuntimeError(' '.join(['Failed:', str(dtype), op.__name__, device])) from e

    def test_ternary_norm_ops(self):
        if False:
            print('Hello World!')

        def apply(fn):
            if False:
                i = 10
                return i + 15
            return lambda x, y, z: fn(x, y, z)
        ternary_ops = [F.batch_norm]
        devices = self.devices
        for (dtype, op, device) in product(self.dtypes, ternary_ops, devices):
            if dtype in [torch.float16, torch.bfloat16] and device == 'cpu':
                continue
            try:
                x = self.data_for(dtype, device, size=[5, 3, 128, 128])
                y = self.data_for(dtype, device, size=[3])
                z = self.data_for(dtype, device, size=[3])
                fn = apply(op)
                ref = fn(x, y, z)
            except Exception:
                continue
            try:
                t = torch.jit.trace(fn, (x, y, z))
                self.assertEqual(ref, t(x, y, z))
                self.assertAllFused(t.graph_for(x, y, z))
            except Exception as e:
                raise RuntimeError(' '.join(['Failed:', str(dtype), op.__name__, device])) from e

    @unittest.skip("FIXME: fuser doesn't include ListConstruct nodes to the group causing a failure")
    def test_list_ops(self):
        if False:
            i = 10
            return i + 15

        def apply(fn):
            if False:
                while True:
                    i = 10
            return lambda x, y, z: fn([x * x, y * y, z * z])
        devices = self.devices
        list_ops = [torch.cat]
        for (dtype, op, device) in product(self.dtypes, list_ops, devices):
            if dtype in [torch.float16, torch.bfloat16] and device == 'cpu':
                continue
            try:
                x = self.data_for(dtype, device, size=[5, 4, 1, 7])
                y = self.data_for(dtype, device, size=[5, 4, 1, 7])
                z = self.data_for(dtype, device, size=[5, 4, 1, 7])
                fn = apply(op)
                ref = fn(x, y, z)
            except Exception:
                continue
            try:
                t = torch.jit.trace(fn, (x, y, z))
                self.assertEqual(ref, t(x, y, z))
                self.assertAllFused(t.graph_for(x, y, z))
            except Exception as e:
                raise RuntimeError(' '.join(['Failed:', str(dtype), op.__name__, device])) from e

    def test_where_ops(self):
        if False:
            i = 10
            return i + 15

        def apply(fn):
            if False:
                return 10
            return lambda cond, x, y: fn(cond, x, y)
        ops = [torch.where, lambda cond, x, y: torch.where(cond, x, 3.1415), lambda cond, x, y: torch.where(cond, 42, y)]
        devices = self.devices
        for (dtype, op, device) in product(self.dtypes, ops, devices):
            if dtype in [torch.float16, torch.bfloat16] and device == 'cpu':
                continue
            try:
                cond = self.data_for(torch.bool, device)
                x = self.data_for(dtype, device)
                y = self.data_for(dtype, device)
                fn = apply(op)
                ref = fn(cond, x, y)
            except Exception:
                continue
            try:
                t = torch.jit.trace(fn, (cond, x, y))
                self.assertEqual(ref, t(cond, x, y))
                self.assertAllFused(t.graph_for(cond, x, y))
            except Exception as e:
                raise RuntimeError(' '.join(['Failed:', str(dtype), op.__name__, device])) from e

    def test_unsupported_dtypes(self):
        if False:
            return 10
        for device in self.devices:

            def fn(x):
                if False:
                    print('Hello World!')
                return x * x + x
            unsupported_dtypes = [torch.uint8, torch.complex32, torch.complex64, torch.complex128, torch.qint8, torch.quint8, torch.qint32]
            for dtype in unsupported_dtypes:
                try:
                    x = self.data_for(dtype, device)
                    ref = fn(x)
                except Exception:
                    continue
                t = torch.jit.trace(fn, (x,))
                self.assertEqual(ref, t(x))
                self.assertEqual(len(self.findFusionGroups(t.graph_for(x))), 0)

    def test_superslomo(self):
        if False:
            for i in range(10):
                print('nop')
        devices = self.devices.copy()
        if not LLVM_ENABLED:
            devices.remove('cpu')
        for device in devices:

            def eager(t0, t1, t2, t3, t4):
                if False:
                    i = 10
                    return i + 15
                t5 = torch.mul(t0, t4)
                t6 = torch.mul(t2, t3)
                t7 = torch.mul(t6, t1)
                t9 = torch.add(t5, t7)
                t11 = torch.add(t0, t6)
                ft_p = torch.div(t9, t11)
                return (ft_p, t11, t9, t6)
            t0 = torch.rand(1, 6, 352, 352, device=device).transpose(0, 1)
            t1 = torch.rand(6, 3, 352, 352, device=device)
            t2 = torch.rand(6, device=device)[None, None, None, :].permute(3, 0, 1, 2)
            t3 = torch.rand(6, 1, 352, 352, device=device)
            t4 = torch.rand(6, 3, 352, 352, device=device)
            inputs = [t0, t1, t2, t3, t4]
            script = torch.jit.script(eager)
            for _ in range(4):
                for pair in zip(script(*inputs), eager(*inputs)):
                    (test, ref) = pair
                    torch.testing.assert_close(test, ref)
                    self.assertAllFused(script.graph_for(*inputs), except_for={'prim::TupleConstruct'})

    def test_sub_gt_and(self):
        if False:
            for i in range(10):
                print('nop')
        for device in self.devices:

            def eager(t1, t2, t3, t4, t: float):
                if False:
                    i = 10
                    return i + 15
                w = t1 - t2
                h = t3 - t4
                k = (w > t) & (h > t)
                assert k.dtype == torch.bool
                if t > 0.5:
                    return k + 1
                return w
            t = torch.rand(8, dtype=torch.float, device=device)
            scripted = self.checkScript(eager, (t, t, t, t, 0.1))

    def test_chunk_mul_one(self):
        if False:
            while True:
                i = 10
        if self.dynamic_shapes:
            self.skipTest('TODO: chunk dynamic shapes')
        for device in self.devices:

            def eager(x):
                if False:
                    i = 10
                    return i + 15
                (z, y, w) = torch.chunk(x, 3, -1)
                return (z * 3, y, w)
            x = torch.rand(64, 1, 3072, dtype=torch.float, device=device)
            (z, y, w) = eager(x)
            script = self.checkScript(eager, (x,))

    def test_eq_unsqueeze_type_as(self):
        if False:
            return 10
        for device in self.devices:

            def eager(a, b):
                if False:
                    return 10
                mask = b == 1
                mask = torch.unsqueeze(mask, -1)
                x = mask.type_as(a)
                return (x, mask)
            a = torch.rand(1, 64, 1024, device=device, dtype=torch.float)
            b = torch.randint(-2, 2, (1, 64), device=device, dtype=torch.long)
            script = self.checkScript(eager, (a, b))

    def test_neg_pow(self):
        if False:
            while True:
                i = 10

        def eager_tt(a: torch.Tensor, b: torch.Tensor):
            if False:
                while True:
                    i = 10
            return torch.neg(torch.pow(a, b))

        def eager_ts(a: torch.Tensor, b: float):
            if False:
                i = 10
                return i + 15
            return torch.neg(torch.pow(a, b))

        def eager_st(a: float, b: torch.Tensor):
            if False:
                i = 10
                return i + 15
            return torch.neg(torch.pow(a, b))
        a = torch.rand(1, dtype=torch.float)
        b = torch.rand(1, dtype=torch.float)
        s = b.item()
        script = self.checkScript(eager_tt, (a, b))
        script = self.checkScript(eager_ts, (a, s))
        script = self.checkScript(eager_st, (s, b))

    @unittest.skipIf(not LLVM_ENABLED, 'Too slow to run with the TE interpreter')
    def test_conv2d_depthwise(self):
        if False:
            for i in range(10):
                print('nop')
        if self.dynamic_shapes:
            self.skipTest("don't run conv with dynamic shapes")

        def eager(input, weight, bias):
            if False:
                for i in range(10):
                    print('nop')
            return torch.conv2d(input, weight, bias, stride=1, padding=1, groups=72)
        input = torch.rand((1, 72, 56, 56), dtype=torch.float)
        weight = torch.rand((72, 1, 3, 3), dtype=torch.float)
        bias = torch.rand(72, dtype=torch.float)
        script = self.checkScript(eager, (input, weight, bias))
        self.assertAllFused(script.graph_for(input, weight, bias))

    def test_conv2d(self):
        if False:
            return 10
        if self.dynamic_shapes:
            self.skipTest("don't run conv with dynamic shapes")

        def eager(input, weight, bias):
            if False:
                i = 10
                return i + 15
            return torch.conv2d(input, weight, bias, stride=1, padding=1, groups=1)
        input = torch.rand((1, 64, 56, 56), dtype=torch.float)
        weight = torch.rand((64, 64, 3, 3), dtype=torch.float)
        bias = torch.rand(64, dtype=torch.float)
        script = self.checkScript(eager, (input, weight, bias))
        FileCheck().check_not('TensorExpr').run(torch.jit.last_executed_optimized_graph())

    def test_type_as_cat(self):
        if False:
            while True:
                i = 10
        with inline_fusion_groups():

            def eager(x, y):
                if False:
                    while True:
                        i = 10
                return torch.cat((x, y.type_as(x)), dim=1)
            dtypes = self.dtypes.copy()
            dtypes.remove(torch.float16)
            dtypes.remove(torch.bfloat16)
            for (dtype1, dtype2) in product(dtypes, dtypes):
                x = torch.randint(2, (1, 13)).to(dtype1)
                zero = torch.tensor([[0]]).to(dtype2)
                one = torch.tensor([[1]]).to(dtype2)
                script = torch.jit.trace(eager, (x, zero))
                for _ in range(3):
                    torch.testing.assert_close(script(x, zero), eager(x, zero))
                    torch.testing.assert_close(script(x, one), eager(x, one))
                self.assertAllFused(script.graph_for(x, one))

    def test_to_device(self):
        if False:
            i = 10
            return i + 15

        def eager(x):
            if False:
                print('Hello World!')
            return x.to(device='cpu').relu()
        x = torch.rand(8)
        script = self.checkScript(eager, (x,))
        self.assertAllFused(script.graph_for(x))

    def test_dims(self):
        if False:
            while True:
                i = 10

        def eager(x, y):
            if False:
                print('Hello World!')
            return x / (y + 0.0001)
        x = torch.linspace(-1, 1, 768, dtype=torch.float32).as_strided((1, 1, 768), (768, 1, 1))
        y = torch.tensor([[[2.0]]], dtype=torch.float32)
        script = self.checkScript(eager, (x, y))
        self.assertAllFused(script.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_channels_last_dims_dynamic(self):
        if False:
            for i in range(10):
                print('nop')

        def eager(x, y):
            if False:
                i = 10
                return i + 15
            return x + (y + 0.0001)
        indices = [0, 1, 2, 3]
        sets = []
        for i in range(0, len(indices) + 1):
            for subset in combinations(indices, i):
                sets.append(subset)
        for set in sets:
            size = [2, 3, 4, 5]
            for index in set:
                size[index] = 1
            inp = torch.rand(size).to(memory_format=torch.channels_last).cuda()
            with texpr_enable_strategy([('DYNAMIC', 20)]):
                foo_s = torch.jit.trace(eager, (inp, inp))
                for _ in range(3):
                    out = foo_s(inp, inp)
                out_eager = eager(inp, inp)
                self.assertEqual(out_eager, out)
                self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
                g = torch.jit.last_executed_optimized_graph()
                FileCheck().check('TensorExpr').run(g)

    def test_exhaust_specializations(self):
        if False:
            return 10
        with texpr_enable_strategy([('STATIC', 1)]):

            @torch.jit.script
            def foo(x):
                if False:
                    i = 10
                    return i + 15
                return x + x + x
            for _ in range(3):
                foo(torch.rand([2, 2]))
            for _ in range(3):
                foo(torch.rand([4, 4, 4]))
            g = torch.jit.last_executed_optimized_graph()
            torch._C._jit_pass_inline(g)
            FileCheck().check_count('TensorExpr', 2, exactly=True).run(g)

    def test_unsqueeze_var_dim(self):
        if False:
            i = 10
            return i + 15

        def eager(x, y, z: int):
            if False:
                return 10
            return x * torch.unsqueeze(y, dim=z)
        x = torch.rand(4, 4, 64).permute(1, 0, 2)
        y = torch.rand(4, 4)
        z = 2
        script = self.checkScript(eager, (x, y, z))

    def _test_fwd_bwd(self, fn):
        if False:
            for i in range(10):
                print('nop')
        x = torch.arange(-10, 10, dtype=torch.float32, requires_grad=True)
        xs = torch.arange(-10, 10, dtype=torch.float32, requires_grad=True)
        script = torch.jit.script(fn)
        for i in range(11):
            y = fn(x)
            g0 = torch.rand_like(y)
            y.backward(g0)
            ys = script(xs)
            ys.backward(g0)
            with torch.no_grad():
                x -= 0.1 * x.grad
                xs -= 0.1 * xs.grad
                x.grad = None
                xs.grad = None
        torch.testing.assert_close(y, ys)

    def test_relu_fwd_bwd(self):
        if False:
            while True:
                i = 10

        def eager(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.relu(x * 1.01)
        self._test_fwd_bwd(eager)

    def test_hardswish_fwd_bwd(self):
        if False:
            for i in range(10):
                print('nop')

        def eager(x):
            if False:
                for i in range(10):
                    print('nop')
            return F.hardswish(x) * 1.01
        self._test_fwd_bwd(eager)

    def test_hardsigmoid_fwd_bwd(self):
        if False:
            for i in range(10):
                print('nop')

        def eager(x):
            if False:
                return 10
            return F.hardsigmoid(x) * 1.01
        self._test_fwd_bwd(eager)

    def test_cat_graph_opt(self):
        if False:
            print('Hello World!')

        def foo(x, y, z):
            if False:
                return 10
            return torch.log(torch.cat([x, y, z]))
        self.checkScript(foo, (torch.rand([5, 5]), torch.rand([2, 5]), torch.rand([1, 5])))
        self.assertLastGraphAllFused()

    def test_dynamic_cat(self):
        if False:
            i = 10
            return i + 15
        with inline_fusion_groups():

            @torch.jit.script
            def repro(xs: List[torch.Tensor], ys: List[torch.Tensor], zs: List[torch.Tensor]):
                if False:
                    return 10
                return [torch.cat([x, torch.cat([y, z], dim=-1)], dim=-1) for (x, y, z) in zip(xs, ys, zs)]
            for _ in range(3):
                N = 3
                xs = [torch.ones(21) for _ in range(N)]
                ys = [torch.ones(N - i) for i in range(N)]
                zs = [torch.ones(i) for i in range(N)]
                repro(xs, ys, zs)

    def test_scalar_only_inputs(self):
        if False:
            print('Hello World!')

        def eager(b: float):
            if False:
                print('Hello World!')
            a = torch.ones(1)
            return a * b
        script = self.checkScript(eager, (1.0,))

    def test_cat_2k_args(self):
        if False:
            print('Hello World!')
        with inline_fusion_groups():

            def eager(x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.relu(torch.cat([x for _ in range(2000)]))
            x = torch.randn(1)
            trace = self.checkTrace(eager, (x,))
            fusion_groups = self.findFusionGroups(trace.graph_for(x))
            self.assertEqual(len(fusion_groups), 0)

    def test_adaptive_avg_pool2d(self):
        if False:
            print('Hello World!')
        with inline_fusion_groups():

            def foo1(x):
                if False:
                    return 10
                return torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))

            def foo2(x):
                if False:
                    while True:
                        i = 10
                return torch.nn.functional.adaptive_avg_pool2d(x, 2)
            x = torch.randn(4, 4, 4)
            for foo in [foo1, foo2]:
                f = torch.jit.trace(foo, (x,))
                kernel = torch._C._te.TensorExprKernel(f.graph)
                correct_val = f(x)
                self.assertEqual(kernel.run((x,)), correct_val)

    def test_unrolled_cat(self):
        if False:
            print('Hello World!')
        with inline_fusion_groups():

            def eager(x):
                if False:
                    return 10
                ret = torch.empty(0)
                for i in range(x.shape[0]):
                    ret = torch.cat([ret, x[i].relu()])
                return ret
            script = torch.jit.script(eager)
            x = torch.ones(1, 1)
            for _ in range(3):
                script(x)
            torch.testing.assert_close(eager(x), script(x))
            x = torch.ones((8, 1))
            torch.testing.assert_close(eager(x), script(x))

    @unittest.skipIf(TEST_WITH_ASAN, 'takes 10+ minutes on asan')
    def test_batch_norm(self):
        if False:
            i = 10
            return i + 15

        def test(fn, args):
            if False:
                for i in range(10):
                    print('nop')
            trace = torch.jit.trace(fn, args)
            self.assertAllFused(trace.graph_for(*args))
            torch.testing.assert_close(fn(*args), trace(*args), equal_nan=True)

        def bn(i, x):
            if False:
                while True:
                    i = 10
            return torch.batch_norm(i, x, x, x, x, False, 0.1, 0.0001, False).relu()

        def bn_no_weight(i, x):
            if False:
                while True:
                    i = 10
            return torch.batch_norm(i, None, x, x, x, False, 0.1, 0.0001, False).relu()

        def bn_no_bias(i, x):
            if False:
                return 10
            return torch.batch_norm(i, x, None, x, x, False, 0.1, 0.0001, False).relu()

        def bn_neither(i, x):
            if False:
                i = 10
                return i + 15
            return torch.batch_norm(i, None, None, x, x, False, 0.1, 0.0001, False).relu()
        for device in self.devices:
            i = torch.randn(4, 16, 32, 40, device=device)
            x = torch.randn(16, device=device)
            for fn in [bn, bn_no_weight, bn_no_bias, bn_neither]:
                test(fn, (i, x))

    def test_profiler(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def test(x, y, z):
            if False:
                while True:
                    i = 10
            return x * y + z
        args = [torch.randn(4) for _ in range(3)]
        with torch.autograd.profiler.profile() as prof:
            for _ in range(3):
                test(*args)
        self.assertIn('fused_mul_add', prof.table())

    def test_skip_grad_in_check(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def foo(x):
            if False:
                i = 10
                return i + 15
            return (x + 2) / 2
        inp = torch.rand([4, 4])
        for _ in range(3):
            foo(inp)
        inp.requires_grad_(True)
        with torch.inference_mode():
            for _ in range(3):
                foo(inp)
        g = torch.jit.last_executed_optimized_graph()
        torch._C._jit_pass_inline(g)
        torch._C._jit_pass_inline(g)
        FileCheck().check_count('prim::If', 1, exactly=True).run(g)

    def test_dynamic_shapes(self):
        if False:
            while True:
                i = 10
        from functools import partial
        n = 10
        gen_tensor = (lambda n: R(1, n), lambda n: R(n, n), lambda n: R(n, n).transpose(0, 1), lambda n: R(n + 1, n + 1, 2)[:n, n, 0], lambda n: R(n, n, 2)[:, :, 0], lambda n: R(n, n + 1, n + 2, n + 3).to(memory_format=torch.channels_last))
        with texpr_enable_strategy([('DYNAMIC', 20)]):

            def foo(x, y, z):
                if False:
                    return 10
                return torch.sigmoid(torch.tanh(x))
            foo.__disable_jit_function_caching__ = True

            def fi(x, y, z):
                if False:
                    while True:
                        i = 10
                return torch.tanh(x + y)
            fi.__disable_jit_function_caching__ = True

            def fum(x, y, z):
                if False:
                    while True:
                        i = 10
                return torch.tanh(x + y) + z
            fum.__disable_jit_function_caching__ = True
            funcs = [foo, fi, fum]
            with inline_fusion_groups():
                for device in self.devices:
                    I = partial(torch.randint, 0, 100, device=device)
                    R = partial(torch.randn, device=device)
                    for (i, func) in enumerate(funcs):
                        num_args = i + 1
                        for (j, gen) in enumerate(gen_tensor):
                            inps = (gen(n), gen(n), gen(n))
                            func_s = torch.jit.trace(func, inps, check_trace=False)
                            torch._C._jit_pass_erase_shape_information(func_s.graph)
                            for _ in range(2):
                                (x, y, z) = (gen(n), gen(n), gen(n))
                                func_s(x, y, z)
                            for incr in range(3):
                                func_s(*[gen(n + 1) for _ in range(3)])
                            g = torch.jit.last_executed_optimized_graph()
                            torch._C._jit_pass_inline(g)
                            torch._C._jit_pass_dce(g)
                            FileCheck().check_count('TensorExprDynamicGuard', 1, exactly=True).run(g)
                            self.assertEqual(func(*inps), func_s(*inps))
                    gen = gen_tensor[0]
                    inps = (gen(n), gen(n), gen(n))
                    foo_s = torch.jit.trace(foo, inps)
                    torch._C._jit_pass_erase_shape_information(foo_s.graph)
                    g_prev = None
                    for gen in gen_tensor:
                        for i in range(3):
                            foo_s(*[gen(n + i) for _ in range(3)])
                            inps = (gen(n), gen(n), gen(n))
                            self.assertEqual(foo_s(*inps), foo(*inps))
                    g = torch.jit.last_executed_optimized_graph()
                    torch._C._jit_pass_inline(g)
                    torch._C._jit_pass_dce(g)
                    FileCheck().check_count('TensorExprDynamicGuard', len(gen_tensor), exactly=True).run(g)

    @unittest.skipIf(not RUN_CUDA, 'half-precision NNC fusion requires CUDA')
    def test_autocast_up(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                print('Hello World!')
            y = x._autocast_to_full_precision(True, True)
            z = torch.exp(y)
            return z
        x = torch.rand((2, 2), dtype=torch.half, device='cuda')
        scr = torch.jit.script(f)
        scr(x)
        scr(x)
        self.assertLastGraphAllFused()

    @unittest.skipIf(not RUN_CUDA, 'half-precision NNC fusion requires CUDA')
    def test_autocast_down(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                while True:
                    i = 10
            y = torch.sigmoid(x)
            z = y._autocast_to_reduced_precision(True, True, torch.half, torch.half)
            return z
        x = torch.rand((2, 2), dtype=torch.float, device='cuda')
        scr = torch.jit.script(f)
        scr(x)
        scr(x)
        self.assertLastGraphAllFused()

    @unittest.skipIf(not LLVM_ENABLED, 'Compiles with TensorExprKernel')
    def test_to_dtype(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                print('Hello World!')
            y = torch.sigmoid(x)
            z = y._autocast_to_reduced_precision(True, True, torch.half, torch.bfloat16)
            h = z._autocast_to_full_precision(True, True)
            i = h.to(dtype=torch.bfloat16)
            j = i.to(dtype=torch.float32)
            return j
        x = torch.rand((2, 2), dtype=torch.float32)
        scr = torch.jit.trace(f, x)
        scr(x)
        scr(x)
        self.assertLastGraphAllFused()
        self.assertEqual(f(x), scr(x), atol=0.004, rtol=0.004)
        bf_x = torch.rand((2, 2), dtype=torch.bfloat16)
        bf_scr = torch.jit.trace(f, bf_x)
        bf_scr(bf_x)
        bf_scr(bf_x)
        graph = bf_scr.graph_for(bf_x)
        fusion_groups = self.findFusionGroups(graph)
        self.assertEqual(len(fusion_groups), 2)
        self.assertEqual(f(bf_x), bf_scr(bf_x), atol=0.004, rtol=0.004)

    def test_with_strict_fusion(self):
        if False:
            for i in range(10):
                print('nop')

        def success(x):
            if False:
                i = 10
                return i + 15
            with torch.jit.strict_fusion():
                return x + x + x
        scripted = self.checkScript(success, (torch.rand([4]),))
        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check_not('aten::add').check('prim::TensorExprGroup').run(g)

        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            with torch.jit.strict_fusion():
                return x + x + torch.rand([4]) + 3
        with self.assertRaises(Exception) as error_out:
            foo_s = torch.jit.script(foo)
            foo_s(torch.rand([4]))
            foo_s(torch.rand([4]))
            print(torch.jit.last_executed_optimized_graph())
        fc = FileCheck().check('Found unfused operators')
        fc.check('aten::rand(SymInt[] size')
        fc.check('torch.rand([4]').run(str(error_out.exception))
        with warnings.catch_warnings(record=True) as warns:
            foo(torch.rand([4]))
        FileCheck().check('Only works in script mode').run(str(warns[0]))

        def test_autodiff(x):
            if False:
                print('Hello World!')
            with torch.jit.strict_fusion():
                return torch.rand([4]) + x + x + x
        foo_s = torch.jit.script(test_autodiff)
        inp = torch.rand([4], requires_grad=True)
        with self.assertRaises(Exception) as error_out:
            for _ in range(3):
                foo_s(inp)
        f = FileCheck().check('unfused operators').check('aten::rand')
        f.run(str(error_out.exception))

        def test_separate_fusions(x, y):
            if False:
                for i in range(10):
                    print('nop')
            with torch.jit.strict_fusion():
                return (x + x + x, y + y + y)
        inp = torch.rand([4], requires_grad=True)
        with self.assertRaises(Exception) as error_out:
            for _ in range(3):
                foo_s = torch.jit.script(test_separate_fusions)
                foo_s(inp, inp)
        f = FileCheck().check('Found multiple fusions')
        f.run(str(error_out.exception))

    def test_constant_chunk_shapes(self):
        if False:
            return 10
        if self.dynamic_shapes:
            self.skipTest('TODO: chunk dynamic shapes')
        for device in self.devices:

            def f(x, y):
                if False:
                    for i in range(10):
                        print('nop')
                r = torch.tensor(4)
                (z1, z2) = (x + y + r).chunk(2, dim=1)
                return z1 * z2
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)
            ge = self.checkTrace(f, (x, y))
            graph = ge.graph_for(x, y)
            FileCheck().check('with ' + FUSION_GROUP + '_').check_count('ConstantChunk', 1, exactly=True).run(str(graph))
            f_traced = torch.jit.trace(f, (x, y))
            for i in range(4):
                res = f_traced(x, y)
            self.assertEqual(res, f(x, y))

    @unittest.skipIf(not RUN_CUDA_HALF, 'half-precision NNC fusion requires CUDA')
    def test_pow_multiple_dtype(self):
        if False:
            print('Hello World!')

        def fn(p: torch.Tensor, gamma: float=2.0) -> torch.Tensor:
            if False:
                while True:
                    i = 10
            p = torch.sigmoid(p)
            result = p ** gamma
            return result
        x = torch.rand((2, 2), dtype=torch.half, device='cuda')
        ref = fn(x)
        script_fn = torch.jit.script(fn)
        for i in range(4):
            res = script_fn(x)
        self.assertEqual(ref, res)

class TestTEFuserStatic(TestTEFuser):
    dynamic_shapes = False

class TestTEFuserDynamic(TestTEFuser):
    dynamic_shapes = True
del TestTEFuser
works_list = ['__radd__', '__rdiv__', '__rmul__', '__rmod__', 'abs', 'acos', 'add', 'addcmul', 'addmm.decomposed', 'asin', 'atan', 'atan2', 'ceil', 'clamp', 'clamp.scalar', 'contiguous', 'cos', 'cosh', 'div.no_rounding_mode', 'div.true_rounding', 'div.floor_rounding', 'div.trunc_rounding', 'eq', 'erf', 'erfc', 'exp', 'expand', 'expand_as', 'expm1', 'floor', 'fmod', 'fmod.autodiffed', 'ge', 'gt', 'isnan', 'le', 'lerp', 'lgamma', 'log', 'log10', 'log1p', 'log2', 'lt', 'masked_fill', 'max.binary', 'mean', 'min.binary', 'mm', 'mul', 'ne', 'neg', 'nn.functional.hardshrink', 'nn.functional.hardsigmoid', 'nn.functional.hardswish', 'nn.functional.softplus', 'nn.functional.hardtanh', 'nn.functional.leaky_relu', 'nn.functional.relu', 'nn.functional.relu6', 'nn.functional.softsign', 'nn.functional.tanhshrink', 'nn.functional.threshold', 'permute', 'pow', 'reciprocal', 'remainder', 'remainder.autodiffed', 'reshape', 'reshape_as', 'round', 'rsub', 'rsub.rsub_tensor', 'rsqrt', 'sigmoid', 'sign', 'sin', 'sinh', 'sqrt', 'sub', 'sum', 't', 'tan', 'tanh', 'transpose', 'true_divide', 'trunc', 'unsqueeze', 'view', 'view_as', 'where', 'bool', 'byte', 'char', 'double', 'float', 'half', 'int', 'long', 'short', 'bool.channels_last', 'byte.channels_last', 'char.channels_last', 'double.channels_last', 'float.channels_last', 'half.channels_last', 'int.channels_last', 'long.channels_last', 'short.channels_last']
known_failures = ['__rmatmul__', 'frac', 'matmul']
skip_ops = ['conj']

def get_name(op):
    if False:
        while True:
            i = 10
    l = [op.name]
    if op.variant_test_name != '':
        l.append(op.variant_test_name)
    return '.'.join(l)

@skipIfTorchDynamo()
class TestNNCOpInfoParent(JitCommonTestCase):
    pass

class TestNNCOpInfo(TestNNCOpInfoParent):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestNNCOpInfoParent, self).setUp()
        self.tensorexpr_options = TensorExprTestOptions()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tensorexpr_options.restore()
        super(TestNNCOpInfoParent, self).tearDown()

    def te_compile(self, device, dtype, op):
        if False:
            print('Hello World!')
        if op.name in skip_ops:
            return
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in sample_inputs_itr:
            arg_values = [sample_input.input] + list(sample_input.args)
            kwarg_values = sample_input.kwargs
            param_names = []
            param_values = []
            fx_args = []
            for (idx, v) in enumerate(arg_values):
                if isinstance(v, torch.Tensor):
                    param_names.append(f'arg_{idx}')
                    param_values.append(v)
                    fx_args.append(param_names[-1])
                else:
                    fx_args.append(f'{repr(v)}')
            for (k, v) in kwarg_values.items():
                if isinstance(v, torch.Tensor):
                    param_names.append(k)
                    param_values.append(v)
                    fx_args.append(f'{k} = {k}')
                else:
                    fx_args.append(f'{k} = {repr(v)}')
            code = f"\ndef f({', '.join(param_names)}):\n    return op.op({', '.join(fx_args)})"
            g = {'torch': torch, 'inf': math.inf, 'op': op}
            exec(code, g)
            f = g['f']
            f.__module__ = 'test'
            out = f(*param_values)
            ts_g = torch.jit.trace(f, param_values)
            kernel = torch._C._te.TensorExprKernel(ts_g.graph)
            correct_val = f(*param_values)
            self.assertEqual(kernel.run(tuple(param_values)), correct_val)
            self.assertEqual(kernel.fallback(tuple(param_values)), correct_val)

    @onlyCPU
    @skipIfTorchDynamo('TorchDynamo fails here for unknown reasons')
    @unittest.skipIf(not LLVM_ENABLED, 'Compiles with TensorExprKernel')
    @ops([op for op in op_db if get_name(op) in works_list], allowed_dtypes=(torch.float,))
    def test_working(self, device, dtype, op):
        if False:
            while True:
                i = 10
        self.te_compile(device, dtype, op)

    @onlyCPU
    @unittest.skipIf(not LLVM_ENABLED, 'Compiles with TensorExprKernel')
    @ops([op for op in op_db if get_name(op) in known_failures], allowed_dtypes=(torch.float,))
    def test_failures(self, device, dtype, op):
        if False:
            while True:
                i = 10
        try:
            self.te_compile(device, dtype, op)
        except Exception as e:
            pass
        else:
            raise RuntimeError('Expected test to fail. If it now works, move op into works_list')

    @onlyCPU
    @unittest.skipIf(not LLVM_ENABLED, 'Compiles with TensorExprKernel')
    @ops([op for op in op_db if get_name(op) not in works_list + known_failures], allowed_dtypes=(torch.float,))
    def test_unsupported(self, device, dtype, op):
        if False:
            while True:
                i = 10
        if get_name(op) in skip_ops:
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', TracerWarning)
                self.te_compile(device, dtype, op)
        except Exception as e:
            pass
        else:
            raise RuntimeError('Expected test to fail. If it now works, move op into works_list')

    @slowTest
    @onlyCPU
    @ops(op_db, dtypes=OpDTypes.supported)
    def test_nnc_correctness(self, device, dtype, op):
        if False:
            while True:
                i = 10
        if not op.supports_tracing:
            self.skipTest('Requires tracing support')
        with NoTracerWarnContextManager() as no_warn:
            variant_sample_pairs = get_traced_sample_variant_pairs(device, dtype, op)
            for (variant, sample) in variant_sample_pairs:
                trace = create_traced_fn(self, variant, cache_traced_fn=True)
                ref = variant(*clone_inputs((sample.input, *sample.args)), **sample.kwargs)
                trace(*clone_inputs((sample.input, *sample.args)), **sample.kwargs)
                val = trace(*clone_inputs((sample.input, *sample.args)), **sample.kwargs)
                atol = 0.2 if dtype == torch.bfloat16 else 1e-05
                rtol = 0.2 if dtype == torch.bfloat16 else 1e-05
                self.assertEqual(ref, val, atol=atol, rtol=rtol)
            torch.jit._state._python_cu.drop_all_functions()
only_for = 'cuda' if IS_FBCODE else ('cpu', 'cuda')
instantiate_device_type_tests(TestNNCOpInfo, globals(), only_for=only_for)

@skipIfTorchDynamo()
class TestLoopnestRandomizationParent(JitTestCase):
    pass

class TestLoopnestRandomization(TestLoopnestRandomizationParent):

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestLoopnestRandomizationParent, self).setUp()
        self.old_cpu_fuser_state = torch._C._jit_can_fuse_on_cpu()
        self.old_must_use_cpu_state = torch._C._jit_get_te_must_use_llvm_cpu()
        self.old_gpu_fuser_state = torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        self.old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        self.old_profiling_mode = torch._C._get_graph_executor_optimize(True)
        self.old_fusion_inlining = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        self.texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
        torch._C._jit_set_texpr_fuser_enabled(True)
        self.old_te_must_use_llvm_cpu = torch._C._jit_get_te_must_use_llvm_cpu()
        torch._C._jit_set_te_must_use_llvm_cpu(False)
        os.environ['PYTORCH_TENSOREXPR_RANDOM_TRANSFORM_SEED'] = '1'

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        torch._C._jit_set_profiling_executor(self.old_profiling_executor)
        torch._C._get_graph_executor_optimize(self.old_profiling_mode)
        torch._C._jit_override_can_fuse_on_gpu(self.old_gpu_fuser_state)
        torch._C._jit_override_can_fuse_on_cpu(self.old_cpu_fuser_state)
        torch._C._jit_set_te_must_use_llvm_cpu(self.old_must_use_cpu_state)
        torch._C._debug_set_fusion_group_inlining(self.old_fusion_inlining)
        torch._C._jit_set_texpr_fuser_enabled(self.texpr_fuser_state)
        torch._C._jit_set_te_must_use_llvm_cpu(self.old_te_must_use_llvm_cpu)
        os.environ['PYTORCH_TENSOREXPR_RANDOM_TRANSFORM_SEED'] = '0'
        super(TestLoopnestRandomizationParent, self).tearDown()

    @onlyCPU
    @unittest.skipIf(not LLVM_ENABLED, 'Compiles with TensorExprKernel')
    def test_relu(self, device):
        if False:
            print('Hello World!')

        def fn_test_relu(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return F.relu(x + 0.5 * y)
        x = torch.randn(4, 4, dtype=torch.float, device=device)
        y = torch.randn(4, 4, dtype=torch.float, device=device)
        fn = fn_test_relu
        traced_fn = torch.jit.trace(fn, (x, y))
        ref = fn(x, y)
        res = traced_fn(x, y)
        assert torch.allclose(ref, res)
instantiate_device_type_tests(TestLoopnestRandomization, globals(), only_for='cpu')
if __name__ == '__main__':
    run_tests()