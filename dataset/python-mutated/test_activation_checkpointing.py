import functools
import unittest
from importlib import import_module
import torch
import torch._dynamo.config
import torch._dynamo.test_case
import torch._functorch.config
import torch.utils.checkpoint
from functorch.compile import min_cut_rematerialization_partition
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import CompileCounterWithBackend
from torch._higher_order_ops.wrap import tag_activation_checkpoint
from torch.testing._internal.common_utils import IS_WINDOWS
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.utils.checkpoint import checkpoint, context_fn_gen
requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, 'requires cuda')

def count_ops(gm, args, freq=None, freq_ge=None, op=None, freqs=None, freqs_ge=None, ops=None):
    if False:
        return 10
    assert (freq or freq_ge) and op or ((freqs or freqs_ge) and ops)
    if op:
        ops = [op]
    if freq:
        freqs = [freq]
    if freq_ge:
        freqs_ge = [freq_ge]
    if freqs:
        for (op, freq) in zip(ops, freqs):
            actual_count = [node.target for node in gm.graph.nodes].count(op)
            assert actual_count == freq, f'In graph {gm}, expected {op} to have occurred {freq} times in the graph, but got {actual_count}.'
    else:
        assert freqs_ge is not None
        for (op, freq_ge) in zip(ops, freqs_ge):
            actual_count = [node.target for node in gm.graph.nodes].count(op)
            assert actual_count >= freq_ge, f'In graph {gm}, expected {op} to have occurred at least {freq_ge} times in the graph, but got {actual_count}.'
    return gm

class _InvalidContext:

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        pass

def _invalid_context_gen():
    if False:
        while True:
            i = 10
    return (_InvalidContext(), _InvalidContext())

def find_first_node(gm, func):
    if False:
        for i in range(10):
            print('nop')
    for node in gm.graph.nodes:
        if node.target is func:
            return node
    return None

def op_count(gm):
    if False:
        for i in range(10):
            print('nop')
    result = 0
    for node in gm.graph.nodes:
        if 'call' in node.op:
            result += 1
    return result

def _get_custom_policy(no_recompute_list=None):
    if False:
        return 10

    def _custom_policy(mode, func, *args, **kwargs):
        if False:
            return 10
        return func in no_recompute_list
    return _custom_policy

class ActivationCheckpointingViaTagsTests(torch._dynamo.test_case.TestCase):

    def _validate(self, fn, backend, *args, skip_check=False, fullgraph=True):
        if False:
            print('Hello World!')
        cloned_args = []
        for arg in args:
            cloned_args.append(arg.clone().detach().requires_grad_(arg.requires_grad))
        torch.manual_seed(0)
        expected = fn(*args)
        expected.sum().backward()
        torch.manual_seed(0)
        result = torch.compile(fn, fullgraph=fullgraph, backend=backend)(*cloned_args)
        result.sum().backward()
        if not skip_check:
            self.assertEqual(result, expected, msg='Output mismatch between torch.compile and eager versions')
            for (arg, cloned_arg) in zip(args, cloned_args):
                self.assertEqual(arg.grad, cloned_arg.grad, msg='Gradient mismatch between torch.compile and eager versions')

    @requires_cuda()
    def test_tags_function(self):
        if False:
            while True:
                i = 10

        def gn(x, y):
            if False:
                print('Hello World!')
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            if False:
                print('Hello World!')
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y)
        x = torch.randn(4, 4, device='cuda', requires_grad=True)
        y = torch.randn(4, 4, device='cuda', requires_grad=True)
        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq=3, op=torch.ops.aten.mm.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    def test_tags_function_via_global_checkpoint(self):
        if False:
            i = 10
            return i + 15

        def gn(x, y):
            if False:
                print('Hello World!')
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return checkpoint(gn, torch.sin(x), y)
        x = torch.randn(4, 4, device='cuda', requires_grad=True)
        y = torch.randn(4, 4, device='cuda', requires_grad=True)
        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq=3, op=torch.ops.aten.mm.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    def test_tags_function_with_kwargs(self):
        if False:
            print('Hello World!')

        def gn(x, y):
            if False:
                print('Hello World!')
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            if False:
                print('Hello World!')
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y, use_reentrant=True, preserve_rng_state=False)
        x = torch.randn(4, 4, device='cuda', requires_grad=True)
        y = torch.randn(4, 4, device='cuda', requires_grad=True)
        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq=3, op=torch.ops.aten.mm.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    def test_tags_multiple_checkpoints(self):
        if False:
            while True:
                i = 10

        def gn(x, y):
            if False:
                while True:
                    i = 10
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            if False:
                print('Hello World!')
            x = torch.sin(x)
            z = torch.utils.checkpoint.checkpoint(gn, x, y)
            x = torch.sin(z)
            z = torch.utils.checkpoint.checkpoint(gn, x, y)
            return z
        x = torch.randn(4, 4, device='cuda', requires_grad=True)
        y = torch.randn(4, 4, device='cuda', requires_grad=True)
        fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq=6, op=torch.ops.aten.mm.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    def test_tags_module(self):
        if False:
            print('Hello World!')

        class MockModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.sigmoid(self.linear(x))
        mod = MockModule().cuda()

        def fn(x):
            if False:
                i = 10
                return i + 15
            return torch.utils.checkpoint.checkpoint(mod, torch.sin(x))
        x = torch.randn(10, 10, device='cuda', requires_grad=True)
        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.sigmoid.default)
        bw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.sigmoid.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x)

    @requires_cuda()
    def test_tags_decomps(self):
        if False:
            while True:
                i = 10

        class MockModule(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                if False:
                    return 10
                return torch.nn.functional.gelu(self.linear(x))
        mod = MockModule().cuda()

        def fn(x):
            if False:
                i = 10
                return i + 15
            return torch.utils.checkpoint.checkpoint(mod, torch.sin(x))
        x = torch.randn(10, 10, device='cuda', requires_grad=True)
        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.erf.default)
        bw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.erf.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler, decompositions=lambda : import_module('torch._inductor.compile_fx').select_decomp_table())
        self._validate(fn, backend, x)

    @requires_cuda()
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_recomputed_rand(self):
        if False:
            while True:
                i = 10

        def gn(x, y):
            if False:
                print('Hello World!')
            return torch.sigmoid(torch.rand_like(x) * y) * x

        def fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            x = torch.sin(x)
            x = torch.utils.checkpoint.checkpoint(gn, x, y)
            x = torch.sin(x)
            z = torch.utils.checkpoint.checkpoint(gn, x, y)
            return z
        x = torch.randn(4, 4, device='cuda', requires_grad=True)
        y = torch.randn(4, 4, device='cuda', requires_grad=True)
        backend = 'inductor'
        self._validate(fn, backend, x, y)

    @requires_cuda()
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_rand(self):
        if False:
            print('Hello World!')

        def gn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            x = torch.mm(x, y)
            x = torch.mm(x, y)
            return x

        def fn(x, y):
            if False:
                while True:
                    i = 10
            x = torch.sin(x)
            x = torch.utils.checkpoint.checkpoint(gn, x, y)
            x = torch.sin(x)
            return x
        x = torch.randn(4, 4, device='cuda', requires_grad=True)
        y = torch.randn(4, 4, device='cuda', requires_grad=True)
        backend = 'inductor'
        self._validate(fn, backend, x, y)

    @requires_cuda()
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_dropout(self):
        if False:
            while True:
                i = 10

        class MockModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.dropout = torch.nn.Dropout(0.2)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.dropout(self.linear(x))
        mod = MockModule().cuda()

        def fn(x):
            if False:
                i = 10
                return i + 15
            return torch.utils.checkpoint.checkpoint(mod, x)
        x = torch.randn(10, 10, device='cuda', requires_grad=True)
        backend = 'inductor'
        self._validate(fn, backend, x, skip_check=True)

    @requires_cuda()
    def test_fallback(self):
        if False:
            print('Hello World!')

        def gn(x, y):
            if False:
                i = 10
                return i + 15
            torch._dynamo.graph_break()
            a = torch.sigmoid(torch.matmul(x, y))
            torch._dynamo.graph_break()
            return torch.cos(a)

        def fn(x, y):
            if False:
                return 10
            return torch.cos(checkpoint(gn, torch.sin(x), y, use_reentrant=False))
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        args = (x, y)
        backend = 'aot_eager'
        cnt = CompileCounterWithBackend(backend)
        expected = fn(*args)
        result = torch.compile(fn, backend=cnt)(*args)
        self.assertEqual(result, expected)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(len(cnt.graphs), 2)

    @requires_cuda()
    def test_kwargs(self):
        if False:
            while True:
                i = 10

        def gn(x, y, z=None):
            if False:
                i = 10
                return i + 15
            a = torch.matmul(x, y)
            if z is not None:
                return torch.matmul(a, z)
            return a

        def fn(x, y, z):
            if False:
                i = 10
                return i + 15
            return torch.cos(checkpoint(gn, x, y, use_reentrant=False, z=z))
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        z = torch.randn(4, 4, requires_grad=True)
        args = (x, y, z)
        backend = 'aot_eager'
        cnt = CompileCounterWithBackend(backend)
        expected = fn(*args)
        result = torch.compile(fn, backend=cnt)(*args)
        self.assertEqual(result, expected)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(cnt.graphs), 1)
        wrap_node = find_first_node(cnt.graphs[0], tag_activation_checkpoint)
        self.assertEqual(len(wrap_node.args), 4)
        body_function = getattr(cnt.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)

    @requires_cuda()
    def test_symints_location(self):
        if False:
            for i in range(10):
                print('nop')

        def gn(x, y):
            if False:
                i = 10
                return i + 15
            return torch.matmul(x, torch.nn.functional.dropout(y, 0.5))

        def fn(x, y):
            if False:
                print('Hello World!')
            return torch.utils.checkpoint.checkpoint(gn, x, y)
        backend = 'aot_eager'
        cnt = CompileCounterWithBackend(backend)
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        args = (x, y)
        expected = fn(*args)
        result = opt_fn(*args)
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)
        args = (x, y)
        expected = fn(*args)
        result = opt_fn(*args)
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(len(cnt.graphs), 2)
        wrap_node = find_first_node(cnt.graphs[0], tag_activation_checkpoint)
        self.assertEqual(len(wrap_node.args), 3)

    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @torch._dynamo.config.patch('_experimental_support_context_fn_in_torch_utils_checkpoint', True)
    def test_compile_selective_checkpoint_gemm_only(self):
        if False:
            while True:
                i = 10

        def selective_checkpointing_context_fn():
            if False:
                for i in range(10):
                    print('nop')
            no_recompute_list = [torch.ops.aten.mm.default]
            return context_fn_gen(_get_custom_policy(no_recompute_list=no_recompute_list))

        def gn(x, y):
            if False:
                while True:
                    i = 10
            return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

        def fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y, use_reentrant=False, context_fn=selective_checkpointing_context_fn)
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq=4, op=torch.ops.aten.mm.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler, partition_fn=min_cut_rematerialization_partition)
        self._validate(fn, backend, x, y)

    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @torch._dynamo.config.patch('_experimental_support_context_fn_in_torch_utils_checkpoint', True)
    def test_compile_selective_checkpoint_custom_rule(self):
        if False:
            return 10

        def _get_custom_policy(meta):
            if False:
                for i in range(10):
                    print('nop')
            no_recompute_list = [torch.ops.aten.mm.default]

            def _custom_policy(mode, func, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                mm_count_key = f'{mode}_mm_count'
                if mm_count_key not in meta:
                    meta[mm_count_key] = 0
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                return func in no_recompute_list and (not (func == torch.ops.aten.mm.default and meta[mm_count_key] == 2))
            return _custom_policy

        def selective_checkpointing_context_fn():
            if False:
                return 10
            meta = {}
            return context_fn_gen(_get_custom_policy(meta))

        def gn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return torch.sigmoid(torch.sigmoid(torch.matmul(torch.matmul(x, y) * y, y) * y))

        def fn(x, y):
            if False:
                print('Hello World!')
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y, use_reentrant=False, context_fn=selective_checkpointing_context_fn)
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq_ge=4, op=torch.ops.aten.mm.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler, partition_fn=min_cut_rematerialization_partition)
        self._validate(fn, backend, x, y)

    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @torch._dynamo.config.patch('_experimental_support_context_fn_in_torch_utils_checkpoint', True)
    def test_compile_selective_checkpoint_outplace_op(self):
        if False:
            i = 10
            return i + 15

        def selective_checkpointing_context_fn():
            if False:
                print('Hello World!')
            no_recompute_list = [torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default]
            return context_fn_gen(_get_custom_policy(no_recompute_list=no_recompute_list))

        def gn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return torch.sigmoid(torch.selu(torch.matmul(torch.matmul(x, y), y))).relu()

        def fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y, use_reentrant=False, context_fn=selective_checkpointing_context_fn)
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        fw_compiler = functools.partial(count_ops, freqs=[2, 1], ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default])
        bw_compiler = functools.partial(count_ops, freqs=[4, 0], ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default])
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler, partition_fn=min_cut_rematerialization_partition)
        self._validate(fn, backend, x, y)

    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @unittest.skip('In-place op support in selective checkpointing + torch.compile requires TorchDispatchMode + torch.compile work to complete')
    @torch._dynamo.config.patch('_experimental_support_context_fn_in_torch_utils_checkpoint', True)
    def test_compile_selective_checkpoint_inplace_op(self):
        if False:
            while True:
                i = 10

        def selective_checkpointing_context_fn():
            if False:
                while True:
                    i = 10
            no_recompute_list = [torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default]
            return context_fn_gen(_get_custom_policy(no_recompute_list=no_recompute_list))

        def gn(x, y):
            if False:
                while True:
                    i = 10
            return torch.sigmoid(torch.selu_(torch.matmul(torch.matmul(x, y), y))).relu_()

        def fn(x, y):
            if False:
                while True:
                    i = 10
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y, use_reentrant=False, context_fn=selective_checkpointing_context_fn)
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        fw_compiler = functools.partial(count_ops, freqs=[2, 1], ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default])
        bw_compiler = functools.partial(count_ops, freqs=[4, 0], ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default])
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler, partition_fn=min_cut_rematerialization_partition)
        self._validate(fn, backend, x, y)

    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @torch._dynamo.config.patch('_experimental_support_context_fn_in_torch_utils_checkpoint', True)
    def test_compile_selective_checkpoint_random_op(self):
        if False:
            for i in range(10):
                print('nop')

        def selective_checkpointing_context_fn():
            if False:
                while True:
                    i = 10
            no_recompute_list = [torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default]
            return context_fn_gen(_get_custom_policy(no_recompute_list=no_recompute_list))

        def gn(x, y):
            if False:
                print('Hello World!')
            return torch.sigmoid(torch.matmul(torch.matmul(torch.bernoulli(torch.sigmoid(x)), y), y))

        def fn(x, y):
            if False:
                i = 10
                return i + 15
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y, use_reentrant=False, context_fn=selective_checkpointing_context_fn)
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        fw_compiler = functools.partial(count_ops, freqs=[2, 2], ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default])
        bw_compiler = functools.partial(count_ops, freqs=[4, 0], ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default])
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler, partition_fn=min_cut_rematerialization_partition)
        self._validate(fn, backend, x, y)

    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @torch._dynamo.config.patch('_experimental_support_context_fn_in_torch_utils_checkpoint', True)
    def test_compile_selective_checkpoint_invalid_context(self):
        if False:
            return 10

        def gn(x, y):
            if False:
                return 10
            return torch.sigmoid(torch.matmul(x, y)) * y

        def fn(x, y):
            if False:
                while True:
                    i = 10
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y, use_reentrant=False, context_fn=_invalid_context_gen)
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq_ge=2, op=torch.ops.aten.mm.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler, partition_fn=min_cut_rematerialization_partition)
        with self.assertRaisesRegex(Exception, 'must generate a tuple of two `TorchDispatchMode`s'):
            self._validate(fn, backend, x, y)

    @requires_cuda()
    def test_autocast_flash_attention(self):
        if False:
            return 10

        def fn(primals_1, primals_2, primals_3):
            if False:
                print('Hello World!')
            return torch.ops.aten._scaled_dot_product_efficient_attention.default(primals_1, primals_2, primals_3, None, True, scale=0.17677669529663687)[0]

        def gn(*args):
            if False:
                i = 10
                return i + 15
            return torch.utils.checkpoint.checkpoint(fn, *args)
        with torch.cuda.amp.autocast():
            x = torch.randn(4, 2, 16, 32, device='cuda', requires_grad=True)
            y = torch.randn(4, 2, 16, 32, device='cuda', requires_grad=True)
            z = torch.randn(4, 2, 16, 32, device='cuda', requires_grad=True)
            args = (x, y, z)
            torch.manual_seed(0)
            ref = gn(*args)
            opt_gn = torch.compile(gn)
            torch.manual_seed(0)
            res = opt_gn(*args)
            self.assertEqual(ref, res)

    @requires_cuda()
    def test_error_msg(self):
        if False:
            while True:
                i = 10

        class MockModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = torch.sin(x)
                torch._dynamo.graph_break()
                x = torch.cos(x)
                return x
        mod = MockModule().cuda()

        def fn(x):
            if False:
                while True:
                    i = 10
            return torch.utils.checkpoint.checkpoint(mod, x)
        x = torch.randn(4, 4).cuda()
        opt_fn = torch.compile(fn, fullgraph=True)
        with self.assertRaisesRegex(RuntimeError, 'while introspecting torch.utils.checkpoint.checkpoint, we were unable to trace function `NNModuleVariable`'):
            opt_fn(x)

    @requires_cuda()
    def test_list_inputs(self):
        if False:
            i = 10
            return i + 15

        class MockModule(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()

            def forward(self, x, ys):
                if False:
                    while True:
                        i = 10
                a = torch.sin(x)
                b = torch.cos(ys[0])
                c = torch.cos(ys[1])
                return (x, [b, c])
        mod = MockModule().cuda()

        def fn(x, ys):
            if False:
                while True:
                    i = 10
            return torch.utils.checkpoint.checkpoint(mod, x, ys)
        x = torch.randn(4, 4).cuda()
        y = torch.randn(4, 4).cuda()
        z = torch.randn(4, 4).cuda()
        ref = fn(x, [y, z])
        opt_fn = torch.compile(fn, backend='eager', fullgraph=True)
        res = opt_fn(x, [y, z])
        self.assertEqual(ref, res)
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    run_tests()