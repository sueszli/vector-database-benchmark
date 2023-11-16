import unittest
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import FileCheck
from unittest import skipIf
from torch.testing._internal.common_utils import run_tests, IS_SANDCASTLE, ProfilingMode, GRAPH_EXECUTOR, enable_profiling_mode_for_profiling_tests, IS_WINDOWS, TemporaryDirectoryName, shell
from torch.testing._internal.jit_utils import JitTestCase, enable_cpu_fuser, _inline_everything, RUN_CUDA, RUN_CUDA_HALF, RUN_CUDA_MULTI_GPU, warmup_backward
from textwrap import dedent
from itertools import product, permutations
from torch.testing._internal.common_cuda import with_tf32_off
from test_jit import backward_graph, all_backward_graphs, get_lstm_inputs, get_milstm_inputs, LSTMCellC, LSTMCellF, LSTMCellS, MiLSTMCell
if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)

def strip_profiling_nodes(nodes):
    if False:
        while True:
            i = 10
    profiling_opcodes = {'prim::BailoutTemplate', 'prim::BailOut'}
    return [n for n in nodes if n.kind() not in profiling_opcodes]

def warmup_forward(f, *args):
    if False:
        i = 10
        return i + 15
    profiling_count = 2
    for i in range(profiling_count):
        results = f(*args)
    return results

@skipIf(GRAPH_EXECUTOR == ProfilingMode.LEGACY, 'skip due to SIGIOT failures, #67646')
class TestFuser(JitTestCase):

    def assertAllFused(self, graph, except_for=()):
        if False:
            return 10
        diff_graphs = [n for n in graph.nodes() if n.kind() == 'prim::DifferentiableGraph']
        if len(diff_graphs) > 0:
            self.assertEqual(len(diff_graphs), 1)
            graph = diff_graphs[0].g('Subgraph')
        allowed_nodes = {'prim::Constant', 'prim::FusionGroup', 'prim::BailoutTemplate', 'prim::BailOut', 'prim::TupleConstruct'} | set(except_for)
        self.assertTrue(all((node.kind() in allowed_nodes for node in graph.nodes())), f'got {graph}')
        self.assertTrue([node.kind() for node in graph.nodes()].count('prim::FusionGroup') == 1)

    def _test_fused_abs(self, device='cpu'):
        if False:
            return 10

        def func(x):
            if False:
                while True:
                    i = 10
            return x.abs() * 2
        a = torch.randn(5, device=device)
        scripted = self.checkScript(func, (a,))
        self.assertAllFused(scripted.graph_for(a))

    @unittest.skipIf(IS_SANDCASTLE, 'NYI: fuser CPU support for Sandcastle')
    @enable_cpu_fuser
    def test_abs_cpu(self):
        if False:
            while True:
                i = 10
        self._test_fused_abs()

    @unittest.skipIf(not IS_WINDOWS, 'This is meant to be Windows-specific')
    @unittest.skipIf(IS_SANDCASTLE, 'NYI: fuser CPU support for Sandcastle')
    @enable_cpu_fuser
    def test_abs_cpu_unicode_temp_dir(self):
        if False:
            for i in range(10):
                print('nop')
        with TemporaryDirectoryName(suffix='中文') as dname:
            shell_env = os.environ.copy()
            shell_env['TMP'] = dname
            cmd = [sys.executable, os.path.basename(__file__), type(self).__name__ + '.test_abs_cpu']
            legacy_jit_flag = '--jit-executor=legacy'
            for v in sys.argv:
                if v == legacy_jit_flag:
                    cmd.append(legacy_jit_flag)
            return_code = shell(cmd, cwd=os.path.dirname(__file__), env=shell_env)
            self.assertEqual(return_code, 0)

    @unittest.skipIf(not RUN_CUDA, 'requires CUDA')
    def test_abs_cuda(self):
        if False:
            i = 10
            return i + 15
        self._test_fused_abs(device='cuda')

    @unittest.skipIf(not RUN_CUDA, 'requires CUDA')
    def test_zero_element_tensors(self):
        if False:
            return 10

        def decode(sin_t, cos_t):
            if False:
                while True:
                    i = 10
            theta = torch.atan2(sin_t.float(), cos_t.float())
            return theta
        sin = torch.zeros(0, device='cuda')
        cos = torch.zeros(0, device='cuda')
        inputs = [sin, cos]
        ge = self.checkScript(decode, inputs)

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_arg_configurations_smoke_cuda(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x, y):
            if False:
                print('Hello World!')
            (z1, z2) = (x + y).chunk(2, dim=1)
            return z1 * z2
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
        traced_f = torch.jit.trace(f, (x, y))
        self.assertEqual(traced_f(x.t().contiguous(), y), traced_f(x.t(), y))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_broadcast_cuda(self):
        if False:
            i = 10
            return i + 15

        def scaleshift(x, scale, shift):
            if False:
                while True:
                    i = 10
            return x * scale + shift
        inputs = [torch.randn(4, 4, dtype=torch.float, device='cuda'), torch.randn(4, dtype=torch.float, device='cuda'), torch.randn(4, dtype=torch.float, device='cuda')]
        ge = self.checkTrace(scaleshift, inputs)
        self.assertAllFused(ge.graph_for(*inputs))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, 'no bfloat support with profiling on')
    def test_cuda_bfloat16(self):
        if False:
            while True:
                i = 10

        def foo(x, y):
            if False:
                i = 10
                return i + 15
            return (x + y).relu()
        m = torch.jit.script(foo)
        x = torch.randn(65536).cuda().bfloat16()
        y = torch.randn_like(x)
        self.assertAllFused(m.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    @unittest.skipIf(not RUN_CUDA_HALF, 'no half support')
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, 'no half support with profiling on')
    def test_cuda_half(self):
        if False:
            print('Hello World!')
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

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_checks_cat_inputs(self):
        if False:
            print('Hello World!')

        def f(x, y):
            if False:
                i = 10
                return i + 15
            return torch.cat([x + 2 * x + x ** 2, y + 4 * y + y ** 3], dim=0)
        x = torch.randn(2, 4, dtype=torch.float, device='cuda')
        y = torch.randn(1, 4, dtype=torch.float, device='cuda')
        scripted = self.checkScript(f, (x, y))
        self.assertAllFused(scripted.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, 'No CUDA')
    def test_remainder_cuda(self):
        if False:
            while True:
                i = 10

        def cuda_rem(x, y):
            if False:
                return 10
            return 1 + torch.remainder(x, y) - 1
        a = torch.rand([512], dtype=torch.float).cuda()
        b = torch.rand([512], dtype=torch.float).cuda()
        inputs = [a, b]
        ge = self.checkScript(cuda_rem, inputs)
        graph = ge.graph_for(*inputs)
        self.assertAllFused(graph)

    @unittest.skipIf(not RUN_CUDA, 'No CUDA')
    def test_chunk_cuda(self):
        if False:
            i = 10
            return i + 15

        def fn(x):
            if False:
                return 10
            (a, b, c) = x.chunk(3, 1)
            return a * b + c
        inputs = [torch.randn(10, 6, dtype=torch.float, device='cuda')]
        ge = self.checkScript(fn, inputs)
        graph = ge.graph_for(*inputs)
        self.assertAllFused(graph)
        FileCheck().check('prim::ConstantChunk[chunks=3, dim=1]').run(str(graph))

    @staticmethod
    def _test_chunk_correctness(self, device='cpu'):
        if False:
            i = 10
            return i + 15

        def chunk_4_0(x):
            if False:
                print('Hello World!')
            (x0, x1, x2, x3) = x.chunk(4, 0)
            return x0 + x1 + x2 + x3

        def chunk_4_1(x):
            if False:
                print('Hello World!')
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

    @unittest.skipIf(IS_SANDCASTLE, 'NYI: fuser CPU support for Sandcastle')
    @enable_cpu_fuser
    def test_chunk_correctness(self):
        if False:
            return 10
        return self._test_chunk_correctness(self, 'cpu')

    @unittest.skipIf(not RUN_CUDA, 'No CUDA')
    def test_chunk_correctness_cuda(self):
        if False:
            i = 10
            return i + 15
        return self._test_chunk_correctness(self, 'cuda')

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_chunk_distributes_cuda(self):
        if False:
            print('Hello World!')

        def f(x, y):
            if False:
                print('Hello World!')
            (z1, z2) = (x + y).chunk(2, dim=1)
            return z1 * z2
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
        ge = self.checkTrace(f, (x, y))
        graph = ge.graph_for(x, y)
        FileCheck().check('broadcast_tensors').check('with prim::FusionGroup_').check_count('ConstantChunk', 2, exactly=True).run(str(graph))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_chunk_motion_deduplicates_inputs(self):
        if False:
            while True:
                i = 10

        def func1(x):
            if False:
                i = 10
                return i + 15
            z = x * x
            (z0, z1) = z.chunk(2)
            return z0 * z1

        def func2(x):
            if False:
                for i in range(10):
                    print('nop')
            z = x * x * x
            (z0, z1) = z.chunk(2)
            return z0 * z1
        inputs = [torch.tensor([1.1, 1.2], device='cuda', dtype=torch.float)]
        for func in [func1, func2]:
            module = self.checkScript(func, inputs)
            forward_graph = module.graph_for(*inputs)
            self.assertGraphContainsExactly(forward_graph, 'prim::FusionGroup', 1)
            fusion_group = list(forward_graph.nodes())[-1]
            self.assertEqual(len(list(fusion_group.inputs())), 1)

    @unittest.skipIf(not RUN_CUDA, 'No CUDA')
    def test_chunk_multiple_cuda(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(s, x, y, z):
            if False:
                for i in range(10):
                    print('nop')
            (z1, z2) = z.chunk(2, 2)
            (x1, x2, x3) = x.chunk(3, 1)
            (y1, y2) = y.chunk(2, 0)
            return s + x1 + x2 + x3 + y1 + y2 + z1 + z2
        inputs = [torch.randn(5, 2, 3, dtype=torch.float, device='cuda'), torch.randn(5, 6, 3, dtype=torch.float, device='cuda'), torch.randn(10, 2, 3, dtype=torch.float, device='cuda'), torch.randn(5, 2, 6, dtype=torch.float, device='cuda')]
        ge = self.checkScript(fn, inputs)
        self.assertAllFused(ge.graph_for(*inputs))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_minmax(self):
        if False:
            i = 10
            return i + 15

        def tmax(a, b):
            if False:
                i = 10
                return i + 15
            return torch.max(2 * a, b)

        def tmin(a, b):
            if False:
                print('Hello World!')
            return torch.min(2 * a, b)
        a = torch.randn(4, 4, dtype=torch.float, device='cuda')
        b = torch.randn(4, 4, dtype=torch.float, device='cuda')
        nan = torch.tensor(float('nan'), dtype=torch.float, device='cuda')
        for (f, inputs) in product((tmax, tmin), ([a, b], [a, nan], [b, nan])):
            s = self.checkScript(f, inputs)
            self.assertAllFused(s.graph_for(*inputs))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_clamp(self):
        if False:
            print('Hello World!')

        def func2(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return torch.clamp(a + b, min=0, max=2)

        def funcInf(a, b):
            if False:
                return 10
            return torch.clamp(a + b, min=0, max=float('inf'))

        def funcOptMin(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return torch.clamp(a + b, max=2)

        def funcOptMax(a, b):
            if False:
                return 10
            return torch.clamp(a + b, min=0)
        a = torch.randn(4, 4, dtype=torch.float, device='cuda', requires_grad=True)
        b = torch.randn(4, 4, dtype=torch.float, device='cuda')
        nan = torch.tensor(float('nan'), dtype=torch.float, device='cuda')
        funcs = (func2, funcInf, funcOptMin, funcOptMax)
        for (f, inputs) in product(funcs, [[a, b], [a, nan]]):
            f.__disable_jit_function_caching__ = True
            (inp1, inp2) = inputs
            s = self.checkScript(f, (inp1, inp2), profiling=ProfilingMode.PROFILING)
            self.assertAllFused(s.graph_for(inp1, inp2), except_for={'aten::size', 'aten::_size_if_not_equal'})
            c = s(inp1, inp2)
            with enable_profiling_mode_for_profiling_tests():
                warmup_backward(c.sum())
            graph = backward_graph(s)
            self.assertAllFused(graph, except_for={'aten::Float', 'aten::_grad_sum_to_size'})

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, 'no half support with profiling on')
    def test_dropout(self):
        if False:
            return 10

        def func(x):
            if False:
                return 10
            x = torch.nn.functional.dropout(x)
            return torch.nn.functional.relu(x)
        a = torch.randn(4, 4, dtype=torch.float, device='cuda', requires_grad=True)
        s = torch.jit.script(func)
        c = s(a)
        c = s(a)
        warmup_backward(c.sum())
        graph = backward_graph(s, skip_check=True)
        self.assertAllFused(graph, except_for={'aten::div', 'prim::Constant'})

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_comparison_eq_ne(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x, y):
            if False:
                i = 10
                return i + 15
            mask = (x == 0).type_as(x)
            z = x * mask + y
            mask = (x != 0).type_as(x)
            z = z * mask + y
            return z
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
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

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_comparison_gt_lt_cuda(self):
        if False:
            return 10
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
        ge = self.checkTrace(self.fn_test_comparison_gt_lt, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_comparison_ge_le_cuda(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x, y):
            if False:
                i = 10
                return i + 15
            mask = (x >= 0).type_as(x)
            z = x * mask + y
            mask = (x <= 0).type_as(x)
            z = z * mask + y
            return z
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
        ge = self.checkTrace(f, (x, y))
        self.assertAllFused(ge.graph_for(x, y))
        x.requires_grad_(True)
        y.requires_grad_(True)
        self.assertAllFused(ge.graph_for(x, y), except_for=('aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_addcmul_cuda(self):
        if False:
            return 10
        t = torch.randn(1, 4, dtype=torch.float, device='cuda')
        t1 = torch.randn(4, 1, dtype=torch.float, device='cuda')
        t2 = torch.randn(1, 4, dtype=torch.float, device='cuda')

        def foo(t, t1, t2):
            if False:
                for i in range(10):
                    print('nop')
            return t.addcmul(t + 1, t2, value=0.1)
        ge = self.checkTrace(foo, (t, t1, t2), allow_unused=True)
        graph = ge.graph_for(t, t1, t2)
        self.assertAllFused(graph)

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_lerp(self):
        if False:
            for i in range(10):
                print('nop')
        start = torch.randn(4, 1, dtype=torch.float, device='cuda')
        end = torch.randn(1, 4, dtype=torch.float, device='cuda')
        weight = torch.tensor(0.5, dtype=torch.float, device='cuda')

        def foo_weight_scalar(start, end):
            if False:
                for i in range(10):
                    print('nop')
            return torch.lerp(start + 1, end, 0.5)

        def foo_weight_tensor(start, end):
            if False:
                return 10
            return torch.lerp(start + 1, end, weight)
        ge_weight_scalar = self.checkTrace(foo_weight_scalar, (start, end))
        graph = ge_weight_scalar.graph_for(start, end)
        self.assertAllFused(graph)
        ge_weight_tensor = self.checkTrace(foo_weight_tensor, (start, end))
        graph = ge_weight_tensor.graph_for(start, end)
        self.assertAllFused(graph)

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_concat_cuda(self):
        if False:
            return 10
        hx = torch.randn(3, 20, dtype=torch.float, device='cuda')
        cx = torch.randn(3, 20, dtype=torch.float, device='cuda')

        def foo(hx, cx):
            if False:
                print('Hello World!')
            return torch.cat((hx + cx, hx * cx))
        ge = self.checkTrace(foo, (hx, cx))
        graph = ge.graph_for(hx, cx)
        self.assertAllFused(graph)
        FileCheck().check('FusedConcat').check_next('return').run(str(graph))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_concat_invariant_cuda(self):
        if False:
            print('Hello World!')

        def fn(x, y, z):
            if False:
                for i in range(10):
                    print('nop')
            x1 = x + y
            y1 = x - y
            w = torch.cat([x1, y1])
            return w + z
        x = torch.randn(2, 2, dtype=torch.float, device='cuda')
        y = torch.randn(2, 2, dtype=torch.float, device='cuda')
        z = torch.randn(4, 2, dtype=torch.float, device='cuda')
        ge = self.checkTrace(fn, (x, y, z))
        graph = ge.graph_for(x, y, z)
        self.assertAllFused(graph, except_for={'aten::add'})
        FileCheck().check('FusedConcat').check_next('return').run(str(graph))

    @staticmethod
    def fn_test_exp(x, y):
        if False:
            return 10
        return (x + 0.5 * y).exp()

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_exp_cuda(self):
        if False:
            print('Hello World!')
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
        ge = self.checkTrace(self.fn_test_exp, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, 'broken with profiling on')
    @torch._jit_internal._disable_emit_hooks_decorator
    @_inline_everything
    def test_fuse_decompose_normalization(self):
        if False:
            for i in range(10):
                print('nop')

        class ResLike(torch.jit.ScriptModule):

            def __init__(self, norm_module):
                if False:
                    print('Hello World!')
                super().__init__()
                self.nm = norm_module

            @torch.jit.script_method
            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return y + torch.relu(self.nm(x))

        def test_norm_decompose(nm, in_opt_graph, not_in_opt_graph, in_fusegraph):
            if False:
                i = 10
                return i + 15
            model = ResLike(nm).cuda()
            model_noopt = ResLike(nm).cuda()
            model_noopt.load_state_dict(model.state_dict())
            x = torch.randn(2, 16, 8, 8, device='cuda')
            y = torch.randn(2, 16, 8, 8, device='cuda')
            with torch.no_grad():
                out = model(x, y)
                graph = model.graph_for(x, y)
                rep = str(graph)
                with torch.jit.optimized_execution(False):
                    out_noopt = model_noopt(x, y)
                    rep_noopt = str(model_noopt.graph_for(x, y))
                self.assertEqual(out, out_noopt, atol=3e-05)
            for node_in_graph in in_opt_graph:
                self.assertIn(node_in_graph, rep)
            for node_not_in_graph in not_in_opt_graph:
                self.assertNotIn(node_not_in_graph, rep)
                self.assertIn(node_not_in_graph, rep_noopt)
            fusion_groups = [node for node in graph.nodes() if node.kind() == 'prim::FusionGroup']
            self.assertEqual(len(fusion_groups), 1)
            fused_graph = str(fusion_groups[0].g('Subgraph'))
            for node_in_fusegraph in in_fusegraph:
                self.assertIn(node_in_fusegraph, fused_graph)
        bm = nn.BatchNorm2d(16)
        test_norm_decompose(bm, ['aten::batch_norm_update_stats'], ['aten::batch_norm('], ['aten::sqrt'])
        lm = nn.LayerNorm(8)
        test_norm_decompose(lm, ['aten::batch_norm_stats'], ['aten::layer_norm('], ['aten::sub', 'aten::mul', 'aten::add'])

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_threshold(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.threshold(x, 0, -10) + x + x + x
        x = torch.tensor([-1, -0.5, 0, 1, 2, 3], device='cuda')
        scripted = self.checkScript(f, (x,))
        self.assertAllFused(scripted.graph_for(x))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_scalar_arg_cuda(self):
        if False:
            while True:
                i = 10

        def fn_test_scalar_arg(x: torch.Tensor, p: float) -> torch.Tensor:
            if False:
                i = 10
                return i + 15
            return p * (x * x + x)
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        p = 3
        scripted = self.checkScript(fn_test_scalar_arg, (x, p))
        self.assertAllFused(scripted.graph_for(x, p))
        x.requires_grad_(True)

        def fn_test_scalar_arg_requires_grad(x: torch.Tensor, p: float) -> torch.Tensor:
            if False:
                print('Hello World!')
            return p * (x * x + x)
        scripted = torch.jit.script(fn_test_scalar_arg_requires_grad)
        out = scripted(x, p)
        self.assertAllFused(scripted.graph_for(x, p), except_for=('aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'))

    @unittest.skipIf(IS_SANDCASTLE, 'NYI: fuser CPU support for Sandcastle')
    @unittest.skip("deduplicating introduces aliasing in backward graph's outputs")
    @enable_cpu_fuser
    def test_fuser_deduplication(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return torch.sigmoid(x + y)
        b = torch.randn(5, 5, requires_grad=True)
        a = torch.randn(5, 5, requires_grad=True)
        s = self.checkScript(f, (a, b))
        self.assertAllFused(s.graph_for(a, b), except_for={'aten::size', 'aten::_size_if_not_equal', 'prim::BroadcastSizes'})
        c = s(a, b)
        results = warmup_backward(c.sum(), [a, b])
        (ga2, gb2) = results.pop()
        graph = backward_graph(s)
        self.assertAllFused(graph)
        self.assertEqual(ga2.data_ptr(), gb2.data_ptr())

    @unittest.skipIf(IS_SANDCASTLE, 'NYI: fuser CPU support for Sandcastle')
    @enable_cpu_fuser
    @unittest.skip('temporarily disabled because fusion was restricted in fixing #22833')
    def test_fuser_iou(self):
        if False:
            print('Hello World!')

        def iou(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2):
            if False:
                for i in range(10):
                    print('nop')
            ltx = torch.max(b1x1, b2x1)
            lty = torch.max(b1y1, b2y1)
            rbx = torch.min(b1x2, b2x2)
            rby = torch.min(b1y2, b2y2)
            w = (rbx - ltx).clamp(min=0, max=float('inf'))
            h = (rby - lty).clamp(min=0, max=float('inf'))
            inter = w * h
            area1 = (b1x2 - b1x1) * (b1y2 - b1y2)
            area2 = (b2x2 - b2x1) * (b2y2 - b2y2)
            iou = inter / (area1 + area2 - inter)
            return iou
        box1 = torch.randn(5, 4, requires_grad=True)
        box2 = torch.randn(5, 4, requires_grad=True)
        b1x1 = box1[:, 0].unsqueeze(1)
        b1y1 = box1[:, 1].unsqueeze(1)
        b1x2 = box1[:, 2].unsqueeze(1)
        b1y2 = box1[:, 3].unsqueeze(1)
        b2x1 = box2[:, 0].unsqueeze(0)
        b2y1 = box2[:, 1].unsqueeze(0)
        b2x2 = box2[:, 2].unsqueeze(0)
        b2y2 = box2[:, 3].unsqueeze(0)
        s = self.checkScript(iou, (b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2))
        self.assertAllFused(s.graph_for(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2), except_for={'aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'})
        with enable_profiling_mode_for_profiling_tests(True):
            c = s(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2)
            warmup_backward(c.sum(), [b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2])
            graph = backward_graph(s)
            self.assertAllFused(graph, except_for={'aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'})

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, 'needs non-zero device')
    @enable_cpu_fuser
    def test_fusion_reuse_multi_gpu(self):
        if False:
            print('Hello World!')

        def fn(x, y):
            if False:
                print('Hello World!')
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
    @enable_cpu_fuser
    def test_kernel_cache_multi_gpu(self):
        if False:
            while True:
                i = 10

        def not_fusible(x):
            if False:
                for i in range(10):
                    print('nop')
            return x

        def fn(x, y, z):
            if False:
                while True:
                    i = 10
            x_out = x * x * x * x * x
            y_out = y * y * y * y * y
            z_out = z * z * z * z * z
            return (not_fusible(x_out), not_fusible(y_out), not_fusible(z_out))
        inputs = [torch.randn(4, 4, dtype=torch.float), torch.randn(4, 4, dtype=torch.float, device='cuda:0'), torch.randn(4, 4, dtype=torch.float, device='cuda:1')]
        prev_cache_size = torch._C._jit_debug_fuser_num_cached_kernel_specs()
        ge = self.checkScript(fn, inputs)
        self.assertGraphContainsExactly(ge.graph_for(*inputs), 'prim::FusionGroup', 3, True)
        new_cache_size = torch._C._jit_debug_fuser_num_cached_kernel_specs()
        self.assertEqual(new_cache_size - prev_cache_size, 1)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, 'needs non-zero device')
    def test_nonzero_device_cuda(self):
        if False:
            while True:
                i = 10
        device = 'cuda:' + str(1)
        x = torch.tensor([0.4], dtype=torch.float, device=device)
        y = torch.tensor([0.7], dtype=torch.float, device=device)

        def doit(x, y):
            if False:
                print('Hello World!')
            return torch.sigmoid(torch.tanh(x * (x + y) + x))
        ge = self.checkTrace(doit, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_lstm_cuda(self):
        if False:
            for i in range(10):
                print('nop')
        inputs = get_lstm_inputs('cuda', training=True)
        module = self.checkScript(LSTMCellS, inputs)
        return
        forward_graph = module.graph_for(*inputs)
        self.assertGraphContainsExactly(forward_graph, 'prim::FusionGroup', 1, consider_subgraphs=True)
        self.assertTrue(len(strip_profiling_nodes(forward_graph.nodes())) == 2)
        FileCheck().check('DifferentiableGraph').check_next('TupleConstruct').check_next('return').run(str(forward_graph))
        with enable_profiling_mode_for_profiling_tests(True):
            (hy, cy) = module(*inputs)
            warmup_backward((hy + cy).sum())
            backward = backward_graph(module)
        self.assertAllFused(backward, except_for=('aten::t', 'aten::mm', 'aten::_grad_sum_to_size'))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    @with_tf32_off
    def test_lstm_concat_cuda(self):
        if False:
            return 10
        inputs = get_lstm_inputs('cuda')
        ge = self.checkTrace(LSTMCellC, inputs)
        graph = ge.graph_for(*inputs)
        FileCheck().check('FusedConcat').check_next('return').run(str(graph))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_lstm_gates_permutations_cuda(self):
        if False:
            for i in range(10):
                print('nop')
        choices = ['x.mm(w_ih.t())', 'hx.mm(w_hh.t())', 'b_ih', 'b_hh']
        template = dedent('\n        def cell(x, hx, cx, w_ih, w_hh, b_ih, b_hh):\n            gates = {} + {} + {} + {}\n            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)\n            return ingate * forgetgate * cellgate * outgate\n        ')
        for permutation in permutations(choices, len(choices)):
            code = template.format(*permutation)
            scope = {}
            exec(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)
            inputs = get_lstm_inputs('cuda', training=False)
            self.assertEqual(cu.cell(*inputs), scope['cell'](*inputs))
            forward_graph = cu.cell.graph_for(*inputs)
            self.assertGraphContainsExactly(forward_graph, 'prim::FusionGroup', 1)

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    @with_tf32_off
    def test_lstm_traced_cuda(self):
        if False:
            while True:
                i = 10
        inputs = get_lstm_inputs('cuda')
        ge = self.checkTrace(LSTMCellF, inputs)
        graph = ge.graph_for(*inputs)
        FileCheck().check_not('Chunk').check_not('aten::sigmoid').check_not('aten::tanh').check('FusionGroup').check_next('TupleConstruct').check_next('return').check_not('FusionGroup_2').run(str(graph))

    @unittest.skipIf(IS_SANDCASTLE, 'NYI: fuser CPU support for Sandcastle')
    @unittest.skip('Test is flaky, see https://github.com/pytorch/pytorch/issues/8746')
    @enable_cpu_fuser
    def test_lstm_traced_cpu(self):
        if False:
            print('Hello World!')
        inputs = get_lstm_inputs('cpu')
        try:
            ge = self.checkTrace(LSTMCellF, inputs)
            graph = ge.graph_for(*inputs)
            FileCheck.check('FusionGroup').run(str(graph))
        except RuntimeError as e:
            if 'Failed to compile' in e.args[0]:
                warnings.warn('CPU fuser test has failed! This is not a hard failure, because the kernels sometimes trigger bugs in compilers (most notably GCC 7.2).')
                raise unittest.SkipTest('Failed to compile') from e
            else:
                raise

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_milstm_cuda(self):
        if False:
            print('Hello World!')
        inputs = get_milstm_inputs('cuda', training=True)
        module = self.checkScript(MiLSTMCell, inputs)
        forward_graph = module.graph_for(*inputs)
        self.assertGraphContainsExactly(forward_graph, 'prim::FusionGroup', 1, consider_subgraphs=True)
        FileCheck().check('DifferentiableGraph').check_next('TupleConstruct').check_next('return').check('FusionGroup').run(str(forward_graph))
        (hy, cy) = module(*inputs)
        warmup_backward((hy + cy).sum())

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.LEGACY, 'borked on the legacy executor')
    def test_rand_cuda(self):
        if False:
            while True:
                i = 10

        class M(torch.jit.ScriptModule):
            __constants__ = ['d']

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.d = torch.device('cuda')

            @torch.jit.script_method
            def create(self, x):
                if False:
                    return 10
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
            return 10
        return F.relu(x + 0.5 * y)

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_relu_cuda(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
        ge = self.checkTrace(self.fn_test_relu, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_erf_cuda(self):
        if False:
            print('Hello World!')

        def fn_test_erf(x):
            if False:
                i = 10
                return i + 15
            return F.relu(torch.erf(x) - torch.erfc(x))
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        ge = self.checkTrace(fn_test_erf, (x,))
        self.assertAllFused(ge.graph_for(x))
        x.requires_grad_(True)
        ge = self.checkTrace(fn_test_erf, (x,))
        self.assertAllFused(ge.graph_for(x), except_for=('aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.LEGACY, 'borked on the legacy executor')
    def test_rand_broadcast_cuda(self):
        if False:
            for i in range(10):
                print('nop')

        def fn_test_rand(x, y):
            if False:
                print('Hello World!')
            r = torch.rand_like(y)
            return r * x + x
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
        script_f = torch.jit.script(fn_test_rand)
        out = script_f(x, y)
        self.assertAllFused(script_f.graph_for(x, y))
        x.requires_grad_(True)
        out = script_f(x, y)
        self.assertAllFused(script_f.graph_for(x, y), except_for=('aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'))
        x = torch.ones(4, 4, dtype=torch.float, device='cuda')
        y = torch.ones(4, dtype=torch.float, device='cuda')
        out = script_f(x, y)
        self.assertEqual(out[0], out[1])

    @unittest.skipIf(IS_SANDCASTLE, 'NYI: fuser CPU support for Sandcastle')
    @enable_cpu_fuser
    def test_scalar(self):
        if False:
            print('Hello World!')

        def fn(x, y):
            if False:
                while True:
                    i = 10
            return 2 * x + y
        x = torch.tensor(0.1, dtype=torch.float, device='cpu')
        y = torch.tensor(1, dtype=torch.float, device='cpu')
        ge = self.checkScript(fn, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_small_constant_cuda(self):
        if False:
            while True:
                i = 10

        def fn_test_small_constant(x, y):
            if False:
                while True:
                    i = 10
            return (1e-08 * x + 5e-09 * y) * 100000000.0
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
        ge = self.checkTrace(fn_test_small_constant, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    def test_tensor_scalar_ops_cuda(self):
        if False:
            print('Hello World!')

        def should_fuse(x):
            if False:
                return 10
            z = 3.0
            y = x + z
            return x * y

        def should_not_fuse(x, z):
            if False:
                for i in range(10):
                    print('nop')
            y = x + int(z)
            return x * y
        inputs = [torch.randn(2, 2, dtype=torch.float, device='cuda')]
        ge = self.checkScript(should_fuse, inputs)
        self.assertAllFused(ge.graph_for(*inputs))
        inputs = [torch.randn(2, 2, dtype=torch.float, device='cuda'), torch.tensor(3.0, dtype=torch.float, device='cuda')]
        ge = self.checkScript(should_not_fuse, inputs)
        self.assertGraphContainsExactly(ge.graph_for(*inputs), 'prim::FusionGroup', 0, consider_subgraphs=True)

    @unittest.skipIf(IS_SANDCASTLE, 'NYI: fuser CPU support for Sandcastle')
    @enable_cpu_fuser
    def test_where_and_typing(self):
        if False:
            return 10

        def f(x, y):
            if False:
                return 10
            mask = x > y
            res = torch.where(mask, x, y)
            return (mask, res)
        x = torch.randn(4, 4, dtype=torch.double)
        y = torch.randn(4, 4, dtype=torch.double)
        script_f = self.checkScript(f, (x, y))
        self.assertAllFused(script_f.graph_for(x, y), except_for={'prim::TupleConstruct'})

    @unittest.skipIf(not RUN_CUDA, 'fuser requires CUDA')
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, 'no half support with profiling on')
    def test_grad_sum_to_size_elimination(self):
        if False:
            return 10

        def my_broadcasted_cell(a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            return a + b + c
        s1 = torch.randn(5, 1, requires_grad=True, device='cuda')
        s2 = torch.randn(5, 5, requires_grad=True, device='cuda')
        module = self.checkScript(my_broadcasted_cell, (s1, s1, s1), profiling=ProfilingMode.PROFILING)
        forward_graph = module.graph_for(s1, s1, s1)
        self.assertAllFused(forward_graph, except_for=('aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'))
        old_plans = set()
        for i in range(3):
            args = (s2 if i < 1 else s1, s2 if i < 2 else s1, s2)
            args = [a.detach_().requires_grad_() for a in args]
            module = self.checkScript(my_broadcasted_cell, args, profiling=ProfilingMode.PROFILING)
            res = module(s2 if i < 1 else s1, s2 if i < 2 else s1, s2)
            warmup_backward(res.sum(), args)
            grads = torch.autograd.grad(res.sum(), args)
            for (inp, gr) in zip(args, grads):
                self.assertEqual(inp.shape, gr.shape)
            backward = None
            for g in all_backward_graphs(module):
                if str(g) not in old_plans:
                    assert backward is None
                    backward = g
                    old_plans.add(str(backward))
            num_grads = 1 if i > 0 else 0
            self.assertEqual(len([n for n in backward.nodes() if n.kind() == 'aten::_grad_sum_to_size']), num_grads)
if __name__ == '__main__':
    run_tests()