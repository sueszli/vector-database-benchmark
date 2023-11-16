import functools
import unittest
from unittest.mock import patch
import torch
from torch._C import FileCheck
import torch.distributed._functional_collectives as _functional_collectives
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.utils import same
from torch._dynamo.testing import CompileCounter
from torch.distributed.distributed_c10d import GroupMember
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_distributed import DynamoDistributedSingleProcTestCase, DynamoDistributedMultiProcTestCase, _dynamo_dist_per_rank_init, requires_nccl, skip_if_lt_x_gpu
from torch._inductor.compile_fx import compile_fx as inductor_compile_fx
from torch.utils._triton import has_triton
from torch._inductor.utils import run_and_get_triton_code
import torch._dynamo.logging

def _tolist_with_constrain_as_size(tensor):
    if False:
        return 10
    lst = tensor.tolist()
    for elem in lst:
        torch._constrain_as_size(elem)
    return lst

@requires_nccl()
class TestCollectivesMultiProc(DynamoDistributedMultiProcTestCase):
    """
    Run correctness checks in multi-proc runner, mark with minimum # GPUs to run under
    """

    def get_world_trs(self):
        if False:
            for i in range(10):
                print('nop')
        return {'tag': '', 'ranks': list(range(self.world_size)), 'group_size': self.world_size}

    @property
    def world_size(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 2

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    def test_broadcast_inductor(self):
        if False:
            return 10
        '\n        Testing if broadcast works correctly when using inductor\n        '

        def example(tensor, src, *, tag, ranks, group_size):
            if False:
                while True:
                    i = 10
            res = torch.ops.c10d_functional.broadcast(tensor, src, tag, ranks, group_size)
            res = torch.ops.c10d_functional.wait_tensor(res)
            return res

        def compile(func, example_inputs):
            if False:
                return 10
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            example = functools.partial(example, **self.get_world_trs())
            t = torch.randn(4, 4, device='cuda')
            inputs = (t if self.rank == 0 else torch.zeros(4, 4, device='cuda'), 0)
            eager_out = example(*inputs)
            self.assertTrue(same(t, eager_out))
            compiled_func = compile(example, inputs)
            compiled_out = compiled_func(*inputs)
            self.assertTrue(same(eager_out, compiled_out))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    def test_allreduce_inductor(self):
        if False:
            while True:
                i = 10
        '\n        This is matmul/cat/allreduce is a pattern we aim to optimize.\n        '

        def matmul_cat_col(a, b, c, d, e, f, *, tag, ranks, group_size):
            if False:
                while True:
                    i = 10
            x = torch.matmul(a, b)
            y = torch.matmul(c, d)
            z = torch.cat((x, y))
            ar = torch.ops.c10d_functional.all_reduce(z, 'sum', tag, ranks, group_size)
            g = torch.matmul(e, f)
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            out = torch.add(ar, g.repeat(2, 1))
            return (out,)

        def compile(func, example_inputs):
            if False:
                while True:
                    i = 10
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            matmul_cat_col = functools.partial(matmul_cat_col, **self.get_world_trs())
            inputs = (torch.ones(4, 4, device='cuda') + self.rank,) * 6
            eager_out = matmul_cat_col(*inputs)
            compiled_matmul_cat_col = compile(matmul_cat_col, inputs)
            inductor_out = compiled_matmul_cat_col(*inputs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    def test_c10d_functional_tagged_pt2_compliant(self):
        if False:
            return 10
        op = torch.ops._c10d_functional.all_reduce.default
        self.assertIn(torch.Tag.pt2_compliant_tag, op.tags)
        op = torch.ops.c10d_functional.all_reduce.default
        self.assertIn(torch.Tag.pt2_compliant_tag, op.tags)

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    def test_eager_allreduce_inductor_wait(self):
        if False:
            print('Hello World!')

        def eager_func(a, b, c, d, *, tag, ranks, group_size):
            if False:
                print('Hello World!')
            x = torch.matmul(a, b)
            y = torch.matmul(c, d)
            z = torch.cat((x, y))
            ar = torch.ops.c10d_functional.all_reduce(z, 'sum', tag, ranks, group_size)
            return ar

        def inductor_func(ar, e, f):
            if False:
                return 10
            g = torch.matmul(e, f)
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            out = torch.add(ar, g.repeat(2, 1))
            return (out,)

        def compile(func, example_inputs):
            if False:
                return 10
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            eager_func = functools.partial(eager_func, **self.get_world_trs())
            eager_inputs = (torch.ones(4, 4, device='cuda') + self.rank,) * 4
            inductor_inputs = (torch.ones(4, 4, device='cuda') + self.rank,) * 2
            eager_out = inductor_func(eager_func(*eager_inputs), *inductor_inputs)
            compiled_inductor_func = compile(inductor_func, [eager_func(*eager_inputs)] + list(inductor_inputs))
            inductor_out = compiled_inductor_func(eager_func(*eager_inputs), *inductor_inputs)
            print(f'eager_out, {eager_out}')
            print(f'inductor_out, {inductor_out}')
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    def test_inductor_allreduce_eager_wait(self):
        if False:
            while True:
                i = 10

        def inductor_func(a, b, c, d, *, tag, ranks, group_size):
            if False:
                for i in range(10):
                    print('nop')
            x = torch.matmul(a, b)
            y = torch.matmul(c, d)
            z = torch.cat((x, y))
            ar = torch.ops.c10d_functional.all_reduce(z, 'sum', tag, ranks, group_size)
            return ar

        def eager_func(ar, e, f):
            if False:
                return 10
            g = torch.matmul(e, f)
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            out = torch.add(ar, g.repeat(2, 1))
            return (out,)

        def compile(func, example_inputs):
            if False:
                i = 10
                return i + 15
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inductor_func = functools.partial(inductor_func, **self.get_world_trs())
            inductor_inputs = (torch.ones(4, 4, device='cuda') + self.rank,) * 4
            eager_inputs = (torch.ones(4, 4, device='cuda') + self.rank,) * 2
            eager_out = eager_func(inductor_func(*inductor_inputs), *eager_inputs)
            compiled_inductor_func = compile(inductor_func, inductor_inputs)
            inductor_out = eager_func(compiled_inductor_func(*inductor_inputs), *eager_inputs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, 'allow_buffer_reuse', True)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    def test_allreduce_input_buffer_reuse(self):
        if False:
            i = 10
            return i + 15

        def func(a, *, tag, ranks, group_size):
            if False:
                print('Hello World!')
            ar = _functional_collectives.all_reduce(a, 'sum', ranks, tag)
            c = torch.relu(a)
            d = torch.matmul(c, c)
            e = d + ar
            return (e,)
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = torch.ones(4, 4, device='cuda') + self.rank
            compiled = torch.compile(func)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, 'allow_buffer_reuse', True)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    def test_allgather_output_buffer_reuse(self):
        if False:
            return 10

        class Model(torch.nn.Module):

            def __init__(self, *args, **kwargs) -> None:
                if False:
                    return 10
                super().__init__(*args, **kwargs)
                self.emb = torch.nn.Embedding(4, 4)

            def forward(self, x, world_size, tag, ranks, group_size):
                if False:
                    for i in range(10):
                        print('nop')
                y = self.emb(x)
                last_dim = y.dim() - 1
                res = _functional_collectives.all_gather_tensor(y, 0, ranks, tag)
                out = torch.cat(torch.chunk(res, world_size, dim=0), dim=last_dim)
                return out
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model = Model().cuda()
            model_compiled = torch.compile(model)
            inp = torch.tensor([[2, 1, 3, 0]], dtype=torch.long, device='cuda')
            out = model_compiled(inp, self.world_size, **self.get_world_trs())
            correct = model(inp, self.world_size, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    def test_allgather_contiguous_input(self):
        if False:
            return 10

        class Model(torch.nn.Module):

            def __init__(self, *args, **kwargs) -> None:
                if False:
                    while True:
                        i = 10
                super().__init__(*args, **kwargs)
                self.emb = torch.nn.Embedding(4, 4)

            def forward(self, x, world_size, tag, ranks, group_size):
                if False:
                    return 10
                y = self.emb(x)
                last_dim = y.dim() - 1
                y = y.transpose_(0, last_dim).contiguous()
                res = _functional_collectives.all_gather_tensor(y, 0, ranks, tag)
                out = y.transpose_(0, last_dim).contiguous()
                return out
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model = Model().cuda()
            model_compiled = torch.compile(model)
            inp = torch.tensor([[2, 1, 3, 0]], dtype=torch.long, device='cuda')
            out = model_compiled(inp, self.world_size, **self.get_world_trs())
            correct = model(inp, self.world_size, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    def test_allgather_into_tensor_inductor(self):
        if False:
            return 10
        '\n        This is matmul/cat/allreduce is a pattern we aim to optimize.\n        '

        def example(a, b, *, tag, ranks, group_size):
            if False:
                i = 10
                return i + 15
            c = torch.matmul(a, b)
            ag = torch.ops.c10d_functional.all_gather_into_tensor(c, tag, ranks, group_size)
            ag = torch.ops.c10d_functional.wait_tensor(ag)
            return (ag,)

        def compile(func, example_inputs):
            if False:
                print('Hello World!')
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            example = functools.partial(example, **self.get_world_trs())
            inputs = (torch.ones(4, 4, device='cuda') + self.rank,) * 2
            eager_out = example(*inputs)
            compiled_matmul_cat_col = compile(example, inputs)
            inductor_out = compiled_matmul_cat_col(*inputs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    def test_reduce_scatter_tensor_inductor(self):
        if False:
            for i in range(10):
                print('nop')

        def example(a, b, *, tag, ranks, group_size):
            if False:
                for i in range(10):
                    print('nop')
            c = torch.matmul(a, b)
            ag = torch.ops.c10d_functional.reduce_scatter_tensor(c, 'sum', tag, ranks, group_size)
            ag = torch.ops.c10d_functional.wait_tensor(ag)
            return (ag,)

        def compile(func, example_inputs):
            if False:
                while True:
                    i = 10
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            example = functools.partial(example, **self.get_world_trs())
            inputs = (torch.ones(4, 4, device='cuda') + self.rank,) * 2
            eager_out = example(*inputs)
            compiled_fn = compile(example, inputs)
            inductor_out = compiled_fn(*inputs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._dynamo.config, 'capture_scalar_outputs', True)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    def test_all_to_all_single_inductor(self):
        if False:
            for i in range(10):
                print('nop')

        def example(inp, input_split_sizes_tensor, output_split_sizes_tensor, *, tag, ranks, group_size):
            if False:
                i = 10
                return i + 15
            input_split_sizes = _tolist_with_constrain_as_size(input_split_sizes_tensor)
            output_split_sizes = _tolist_with_constrain_as_size(output_split_sizes_tensor)
            a2a = torch.ops.c10d_functional.all_to_all_single(inp, output_split_sizes, input_split_sizes, tag, ranks, group_size)
            a2a = torch.ops.c10d_functional.wait_tensor(a2a)
            out = a2a / a2a.sum(dim=0)
            return out
        with _dynamo_dist_per_rank_init(self.rank, self.world_size), torch._dynamo.config.patch(dynamic_shapes=True, capture_dynamic_output_shape_ops=True, capture_scalar_outputs=True):
            row = self.world_size * (self.rank + 1) * (self.world_size + 1) / 2
            input_split_sizes_tensor = torch.tensor([(i + 1) * (self.rank + 1) for i in range(self.world_size)], dtype=torch.int64)
            output_split_sizes_tensor = torch.tensor([(i + 1) * (self.rank + 1) for i in range(self.world_size)], dtype=torch.int64)
            inputs = (torch.ones(int(row), 5, device='cuda') * (self.rank + 1), input_split_sizes_tensor, output_split_sizes_tensor)
            trs = self.get_world_trs()
            compiled_fn = torch.compile(example, fullgraph=True, dynamic=True)
            code = run_and_get_triton_code(compiled_fn, *inputs, **trs)
            FileCheck().check_regex('all_to_all_single\\(buf\\d+\\[0\\], buf\\d+_inputs\\[0\\], output_split_sizes=\\[i\\d+, i\\d+\\], input_split_sizes=\\[i\\d+, i\\d+\\]').run(code)
            eager_out = example(*inputs, **trs)
            inductor_out = compiled_fn(*inputs, **trs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._dynamo.config, 'capture_scalar_outputs', True)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    def test_all_to_all_single_inductor_output_split_sizes_none(self):
        if False:
            print('Hello World!')

        def example(inp, input_split_sizes_tensor, *, tag, ranks, group_size):
            if False:
                return 10
            input_split_sizes = _tolist_with_constrain_as_size(input_split_sizes_tensor)
            a2a = torch.ops.c10d_functional.all_to_all_single(inp, None, input_split_sizes, tag, ranks, group_size)
            a2a = torch.ops.c10d_functional.wait_tensor(a2a)
            out = a2a / a2a.sum(dim=0)
            return out
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            input_split_sizes_tensor = torch.tensor([1] * self.world_size, dtype=torch.int64)
            inputs = (torch.ones(self.world_size, self.world_size, device='cuda') * (self.rank + 1), input_split_sizes_tensor)
            trs = self.get_world_trs()
            compiled_fn = torch.compile(example, fullgraph=True, dynamic=True)
            code = run_and_get_triton_code(compiled_fn, *inputs, **trs)
            FileCheck().check_regex('all_to_all_single\\(buf\\d+\\[0\\], buf\\d+_inputs\\[0\\], output_split_sizes=None, input_split_sizes=\\[i\\d+, i\\d+\\]').run(code)
            eager_out = example(*inputs, **trs)
            inductor_out = compiled_fn(*inputs, **trs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._dynamo.config, 'capture_scalar_outputs', True)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    def test_all_to_all_single_inductor_input_split_sizes_none(self):
        if False:
            while True:
                i = 10

        def example(inp, output_split_sizes_tensor, *, tag, ranks, group_size):
            if False:
                while True:
                    i = 10
            output_split_sizes = _tolist_with_constrain_as_size(output_split_sizes_tensor)
            a2a = torch.ops.c10d_functional.all_to_all_single(inp, output_split_sizes, None, tag, ranks, group_size)
            a2a = torch.ops.c10d_functional.wait_tensor(a2a)
            out = a2a / a2a.sum(dim=0)
            return out
        with _dynamo_dist_per_rank_init(self.rank, self.world_size), torch._dynamo.config.patch(dynamic_shapes=True, capture_dynamic_output_shape_ops=True, capture_scalar_outputs=True):
            output_split_sizes_tensor = torch.tensor([1] * self.world_size, dtype=torch.int64)
            inputs = (torch.ones(self.world_size, self.world_size, device='cuda') * (self.rank + 1), output_split_sizes_tensor)
            trs = self.get_world_trs()
            compiled_fn = torch.compile(example, fullgraph=True, dynamic=True)
            code = run_and_get_triton_code(compiled_fn, *inputs, **trs)
            FileCheck().check_regex('all_to_all_single\\(buf\\d+\\[0\\], buf\\d+_inputs\\[0\\], output_split_sizes=\\[i\\d+, i\\d+\\], input_split_sizes=None').run(code)
            eager_out = example(*inputs, **trs)
            inductor_out = compiled_fn(*inputs, **trs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    def test_all_to_all_single_inductor_split_sizes_none(self):
        if False:
            return 10

        def example(inp, *, tag, ranks, group_size):
            if False:
                for i in range(10):
                    print('nop')
            a2a = torch.ops.c10d_functional.all_to_all_single(inp, None, None, tag, ranks, group_size)
            a2a = torch.ops.c10d_functional.wait_tensor(a2a)
            out = a2a / a2a.sum(dim=0)
            return out
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = (torch.ones(self.world_size, self.world_size, device='cuda') * (self.rank + 1),)
            trs = self.get_world_trs()
            compiled_fn = torch.compile(example, fullgraph=True, dynamic=True)
            code = run_and_get_triton_code(compiled_fn, *inputs, **trs)
            FileCheck().check_regex('all_to_all_single\\(buf\\d+\\[0\\], buf\\d+_inputs\\[0\\], output_split_sizes=None, input_split_sizes=None').run(code)
            eager_out = example(*inputs, **trs)
            inductor_out = compiled_fn(*inputs, **trs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

@requires_nccl()
class TestCollectivesInductor(DynamoDistributedSingleProcTestCase):
    """
    Prefer single-proc test runner for basic tests as it is easier to work with.
    """

    def get_world_trs(self, world_size=1):
        if False:
            return 10
        return {'tag': '', 'ranks': list(range(world_size)), 'group_size': world_size}

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    def test_inductor_single_op(self):
        if False:
            while True:
                i = 10
        torch._inductor.config.debug = True

        def func(inp, *, tag, ranks, group_size):
            if False:
                i = 10
                return i + 15
            ar = torch.ops.c10d_functional.all_reduce(inp, 'sum', tag, ranks, group_size)
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            return ar
        inputs = torch.ones(4, 4, device='cuda')
        compiled = torch.compile(func)
        out = compiled(inputs, **self.get_world_trs())
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        FileCheck().check('buf0 = empty(').check('buf0.copy_(arg0_1)').check('buf1 = buf0').check('buf1_work = dist.all_reduce(buf1').check('fun_col_impl._register_tensor_work(buf1, buf1_work)').check('buf0 = _wait_tensor(buf0)').check('return (buf0, )').run(code)
        correct = func(inputs, **self.get_world_trs())
        self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    def test_inductor_steal_buffer(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        it's ok and optimal if inductor allreduce mutates the buffer of an intermediate\n        that isn't going to be used again\n        "
        torch._inductor.config.debug = True

        def func(inp, *, tag, ranks, group_size):
            if False:
                for i in range(10):
                    print('nop')
            x = inp + 1
            ar = torch.ops.c10d_functional.all_reduce(x, 'sum', tag, ranks, group_size)
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            other = torch.ones_like(inp) + 22
            return (ar, other)
        inputs = torch.ones(4, 4, device='cuda')
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        FileCheck().check('buf1 = buf0; del buf0  # reuse').check_not('buf1.copy_(').check('buf2 = buf1').check('buf2_work = dist.all_reduce(buf2').check('fun_col_impl._register_tensor_work(buf2, buf2_work)').check('buf1 = _wait_tensor(buf1)').check('buf4 = buf1').check('buf5 = empty').check('return (buf1, buf5').run(code)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @patch.object(torch._inductor.config.triton, 'descriptive_names', False)
    def test_inductor_doesnt_mutate_shared(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        make sure that an intermediate that's going to be reuse isn't mutated unless copied\n        "
        torch._inductor.config.debug = True

        def func(inp, *, tag, ranks, group_size):
            if False:
                print('Hello World!')
            x = inp + 1
            ar = torch.ops.c10d_functional.all_reduce(x, 'sum', tag, ranks, group_size)
            y = x + 2
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            other = torch.ones_like(inp) + 22
            return (ar, y, other)
        inputs = torch.ones(4, 4, device='cuda')
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        FileCheck().check('buf0 = empty(').check('buf5 = empty(').check('triton_poi__0.run(arg0_1, buf0, buf5').check_not('copy_(').check('buf1 = buf0; del buf0  # reuse').check('buf2 = buf1').check('buf2_work = dist.all_reduce(buf2').check('fun_col_impl._register_tensor_work(buf2, buf2_work)').check('buf1 = _wait_tensor(buf1)').check('buf4 = buf1').check('return (buf1, buf5, buf6').run(code)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        self.assertTrue(same(out, correct))

    def test_dynamo_trace_allreduce(self):
        if False:
            return 10

        def func(inp, *, tag, ranks, group_size):
            if False:
                i = 10
                return i + 15
            ar = _functional_collectives.all_reduce(inp, 'sum', ranks, tag)
            return ar
        inputs = torch.ones(4, 4, device='cuda')
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 2)
        self.assertTrue(same(out, correct))

    def test_dynamo_trace_all_gather_tensor(self):
        if False:
            while True:
                i = 10

        def func(inp, *, tag, ranks, group_size):
            if False:
                for i in range(10):
                    print('nop')
            ar = _functional_collectives.all_gather_tensor(inp, 0, ranks, tag)
            return ar
        inputs = torch.ones(4, 4, device='cuda')
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 2)
        self.assertTrue(same(out, correct))

    def test_dynamo_trace_all_gather_tensor_pg(self):
        if False:
            return 10

        def func(inp, *, pg):
            if False:
                while True:
                    i = 10
            ar = _functional_collectives.all_gather_tensor(inp, 0, pg)
            return ar
        inputs = torch.ones(4, 4, device=self.device)
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter, fullgraph=True)
        out = compiled(inputs, pg=GroupMember.WORLD)
        correct = func(inputs, pg=GroupMember.WORLD)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 2)
        self.assertTrue(same(out, correct))

    def test_dynamo_rewrite_dist_all_gather(self):
        if False:
            return 10

        def func(inp, out, *, pg):
            if False:
                return 10
            torch.distributed.all_gather_into_tensor(out, inp, pg)
        local_size = [4, 4]
        global_size = local_size
        inputs = torch.ones(local_size, device=self.device)
        outputs = torch.empty(global_size, device=self.device)
        correct_outputs = torch.empty(global_size, device=self.device)
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter, fullgraph=True)
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 1
        assert counter.op_count == 3
        assert same(outputs, correct_outputs)

    def test_dynamo_rewrite_dist_reduce_scatter(self):
        if False:
            return 10

        def func(inp, out, *, pg):
            if False:
                print('Hello World!')
            torch.distributed.reduce_scatter_tensor(out, inp, group=pg)
        local_size = [4, 4]
        global_size = local_size
        inputs = torch.ones(local_size, device=self.device)
        outputs = torch.empty(global_size, device=self.device)
        correct_outputs = torch.empty(global_size, device=self.device)
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter, fullgraph=True)
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 1
        assert counter.op_count == 3
        assert same(outputs, correct_outputs)

    def test_dynamo_graphbreaks_unsupported_async_op(self):
        if False:
            return 10

        def func(inp, out, *, pg):
            if False:
                i = 10
                return i + 15
            work = torch.distributed.reduce_scatter_tensor(out, inp, group=pg, async_op=True)
            work.wait()
        local_size = [4, 4]
        global_size = local_size
        inputs = torch.ones(local_size, device=self.device)
        outputs = torch.empty(global_size, device=self.device)
        correct_outputs = torch.empty(global_size, device=self.device)
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter)
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 0
        assert counter.op_count == 0
        assert same(outputs, correct_outputs)

    def test_dynamo_pg_var(self):
        if False:
            return 10

        def func(inp, *, pg):
            if False:
                for i in range(10):
                    print('nop')
            x = pg.rank() + 1 % pg.size()
            return inp + x
        local_size = [4, 4]
        inputs = torch.ones(local_size, device=self.device)
        correct_outputs = torch.empty(local_size, device=self.device)
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter, fullgraph=True)
        outputs = compiled(inputs, pg=GroupMember.WORLD)
        correct_outputs = func(inputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 1
        assert counter.op_count == 1
        assert same(outputs, correct_outputs)

    def test_dynamo_trace_reduce_scatter_tensor(self):
        if False:
            i = 10
            return i + 15

        def func(inp, *, tag, ranks, group_size):
            if False:
                while True:
                    i = 10
            ar = _functional_collectives.reduce_scatter_tensor(inp, 'sum', 0, ranks, tag)
            return ar
        inputs = torch.ones(4, 4, device='cuda')
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 2)
        self.assertTrue(same(out, correct))

    def test_dynamo_trace_allgather_coalesced(self):
        if False:
            while True:
                i = 10

        def func(inp, *, tag, ranks, group_size):
            if False:
                for i in range(10):
                    print('nop')
            ar = torch.ops.c10d_functional.all_gather_into_tensor_coalesced(inp, tag, ranks, group_size)
            return ar
        inputs = [torch.ones(4, 4, device='cuda'), torch.ones(6, 6, device='cuda')]
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        assert counter.frame_count == 1
        assert counter.op_count == 3
        assert same(out, correct)

    def test_backwards(self):
        if False:
            i = 10
            return i + 15
        "\n        It's probably not that common to need backwards support for collectives.\n\n        However, I wanted to at least see if it was possible to support it as a design goal.\n        "

        def func(inp, *, tag, ranks, group_size):
            if False:
                i = 10
                return i + 15
            ar = _functional_collectives.all_reduce(inp, 'sum', ranks, tag)
            return ar
        input = torch.ones(4, 4, device='cuda', requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, 'element 0 of tensors does not require grad and does not have a grad_fn'):
            compiled = torch.compile(func, backend='aot_eager')
            out = compiled(input, **self.get_world_trs())
            out.sum().backward()
            correct_input = input.clone().detach().requires_grad_()
            correct = func(correct_input, **self.get_world_trs())
            correct.sum().backward()
            self.assertTrue(same(out, correct))
            self.assertTrue(same(input.grad, correct_input.grad))

    def test_meta(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.rand((2, 3, 4), device='meta')
        out = torch.ops.c10d_functional.all_reduce(x, 'sum', **self.get_world_trs())
        self.assertEqual(x.size(), out.size())

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @patch.object(torch._inductor.config.triton, 'descriptive_names', False)
    def test_inductor_all_gather_coalesced(self):
        if False:
            while True:
                i = 10
        "\n        make sure that an intermediate that's going to be reuse isn't mutated unless copied\n        "
        torch._inductor.config.debug = True

        def func(inp, *, tag, ranks, group_size):
            if False:
                i = 10
                return i + 15
            x = inp + 1
            tensor_list = torch.ops.c10d_functional.all_gather_into_tensor_coalesced([x, inp], tag, ranks, group_size)
            y = x + 2
            ar0 = torch.ops.c10d_functional.wait_tensor(tensor_list[0])
            ar1 = torch.ops.c10d_functional.wait_tensor(tensor_list[1])
            other = torch.ones_like(inp) + 22
            return (ar0, y, other, ar1)
        inputs = torch.ones(4, 4, device='cuda')
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        FileCheck().check('buf0 = empty(').check('buf5 = empty(').check('triton_poi__0.run(arg0_1, buf0, buf5').check('buf1 = empty(').check('buf2 = empty(').check_not('copy_(').check('buf3_inputs = [buf0,arg0_1]').check('buf3 = [buf1,buf2]').check('buf3_work = fun_col_impl._all_gather_into_tensor_coalesced_fallback(output_tensors=buf3, input_tensors=buf3_inputs').check('fun_col_impl._register_tensor_work(buf3, buf3_work)').check('buf1 = _wait_tensor(buf1)').check('buf4 = buf1').check('buf6 = buf0; del buf0  # reuse').check('buf2 = _wait_tensor(buf2)').check('buf7 = buf2').check('return (buf1, buf5, buf6, buf2').run(code)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        assert same(out, correct), f'{out} va {correct}'

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @patch.object(torch._inductor.config.triton, 'descriptive_names', False)
    def test_inductor_reduce_scatter_coalesced(self):
        if False:
            while True:
                i = 10
        "\n        make sure that an intermediate that's going to be reuse isn't mutated unless copied\n        "
        torch._inductor.config.debug = True

        def func(inp, *, tag, ranks, group_size):
            if False:
                while True:
                    i = 10
            x = inp + 1
            tensor_list = torch.ops.c10d_functional.reduce_scatter_tensor_coalesced([x, inp], 'sum', tag, ranks, group_size)
            y = x + 2
            ar0 = torch.ops.c10d_functional.wait_tensor(tensor_list[0])
            ar1 = torch.ops.c10d_functional.wait_tensor(tensor_list[1])
            other = torch.ones_like(inp) + 22
            return (ar0, y, other, ar1)
        inputs = torch.ones(4, 4, device='cuda')
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        FileCheck().check('buf0 = empty(').check('buf5 = empty(').check('triton_poi__0.run(arg0_1, buf0, buf5').check('buf1 = empty(').check('buf2 = empty(').check_not('copy_(').check('buf3 = [buf1,buf2]').check('buf3_work = fun_col_impl._reduce_scatter_tensor_coalesced_fallback(output_tensors=buf3, input_tensors=buf3_inputs').check('fun_col_impl._register_tensor_work(buf3, buf3_work)').check('buf1 = _wait_tensor(buf1)').check('buf4 = buf1').check('buf6 = buf0; del buf0  # reuse').check('buf2 = _wait_tensor(buf2)').check('buf7 = buf2').check('return (buf1, buf5, buf6, buf2').run(code)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        assert same(out, correct), f'{out} va {correct}'
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    run_tests()