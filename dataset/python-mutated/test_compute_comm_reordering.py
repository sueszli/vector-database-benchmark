import unittest
from unittest.mock import patch
import torch
import torch._dynamo
import torch._dynamo.logging
import torch._dynamo.test_case
import torch.distributed._functional_collectives as _functional_collectives
from torch._C import FileCheck
from torch._dynamo.utils import same
from torch._inductor import ir
from torch._inductor.comm_analysis import baseLat, hwLat, llMaxBws, NCCL_ALGO, NCCL_HW, NCCL_PROTO, NVIDIA_GPU_TYPE
from torch._inductor.utils import run_and_get_triton_code
from torch.testing._internal.common_distributed import _dynamo_dist_per_rank_init, DynamoDistributedMultiProcTestCase, requires_nccl, skip_if_lt_x_gpu
from torch.utils._triton import has_triton

def get_snode_runtime_for_reorder_compute_test(snode):
    if False:
        while True:
            i = 10
    if isinstance(snode.node, ir.CollectiveKernel):
        if isinstance(snode.node, ir.AllReduce):
            return 100
        else:
            return 100
    elif isinstance(snode.node, ir.Wait):
        return 0
    elif isinstance(snode.node, ir.ExternKernel):
        return 5
    return 1

@requires_nccl()
class TestComputeCommReorderingMultiProc(DynamoDistributedMultiProcTestCase):
    """
    Run correctness checks in multi-proc runner, mark with minimum # GPUs to run under
    """

    def get_world_trs(self):
        if False:
            while True:
                i = 10
        return {'tag': '', 'ranks': list(range(self.world_size)), 'group_size': self.world_size}

    @property
    def world_size(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 2

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, 'allow_buffer_reuse', True)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    @patch.object(torch._inductor.config, 'reorder_for_compute_comm_overlap', True)
    @patch.object(torch._inductor.config, 'reorder_for_compute_comm_overlap_passes', ['sink_waits'])
    def test_sink_waits(self):
        if False:
            while True:
                i = 10

        def func(a, *, tag, ranks, group_size):
            if False:
                for i in range(10):
                    print('nop')
            ar = _functional_collectives.all_reduce(a, 'sum', ranks, tag)
            c = torch.relu(a)
            d = torch.matmul(c, c)
            e = d + ar
            return (e,)
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = torch.ones(4, 4, dtype=torch.float, device='cuda') + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            FileCheck().check('dist.all_reduce(').check('triton_poi_fused_relu').check('_wait_tensor(').run(code)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, 'allow_buffer_reuse', True)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    @patch.object(torch._inductor.config, 'reorder_for_compute_comm_overlap', True)
    @patch.object(torch._inductor.config, 'reorder_for_compute_comm_overlap_passes', ['raise_comms'])
    def test_raise_comms(self):
        if False:
            for i in range(10):
                print('nop')

        def func(a, *, tag, ranks, group_size):
            if False:
                print('Hello World!')
            c = torch.relu(a)
            d = torch.matmul(c, c)
            ar = _functional_collectives.all_reduce(a, 'sum', ranks, tag)
            e = d + ar
            return (e,)
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = torch.ones(4, 4, dtype=torch.float, device='cuda') + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            FileCheck().check('dist.all_reduce(').check('_wait_tensor(').check('triton_poi_fused_relu').check('extern_kernels.addmm(').run(code)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, 'allow_buffer_reuse', True)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    @patch.object(torch._inductor.config, 'reorder_for_compute_comm_overlap', True)
    @patch.object(torch._inductor.config, 'reorder_for_compute_comm_overlap_passes', ['sink_waits', 'raise_comms'])
    def test_sink_waits_raise_comms(self):
        if False:
            print('Hello World!')

        def func(a, *, tag, ranks, group_size):
            if False:
                return 10
            c = torch.relu(a)
            d = torch.matmul(c, c)
            ar = _functional_collectives.all_reduce(a, 'sum', ranks, tag)
            e = d + ar
            return (e,)
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = torch.ones(4, 4, dtype=torch.float, device='cuda') + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            FileCheck().check('dist.all_reduce(').check('triton_poi_fused_relu').check('_wait_tensor(').check('extern_kernels.addmm(').run(code)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, 'allow_buffer_reuse', True)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    @patch.object(torch._inductor.config, 'reorder_for_compute_comm_overlap', True)
    @patch.object(torch._inductor.config, 'reorder_for_compute_comm_overlap_passes', ['reorder_compute_for_overlap'])
    def test_reorder_compute_for_overlap(self):
        if False:
            print('Hello World!')

        def func(a, *, tag, ranks, group_size):
            if False:
                i = 10
                return i + 15
            ar = _functional_collectives.all_reduce(a, 'sum', ranks, tag)
            g = torch.matmul(a, a)
            c = torch.relu(a)
            d = torch.matmul(c, c)
            f = d * c * ar
            fr = _functional_collectives.all_reduce(f, 'sum', ranks, tag)
            e = torch.matmul(d + ar + fr, g)
            return (e,)
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = torch.ones(4, 4, dtype=torch.float, device='cuda') + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            FileCheck().check('dist.all_reduce(').check('triton_poi_fused_relu').check('extern_kernels.mm(').check('extern_kernels.mm(').check('_wait_tensor(').check('triton_poi_fused_mul').check('dist.all_reduce(').check('_wait_tensor(').check('triton_poi_fused_add').check('extern_kernels.mm(').run(code)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, 'allow_buffer_reuse', True)
    @patch.object(torch._inductor.config, 'compile_threads', 1)
    @patch.object(torch._inductor.config, 'reorder_for_compute_comm_overlap', True)
    @patch.object(torch._inductor.config, 'reorder_for_compute_comm_overlap_passes', ['reorder_compute_for_overlap'])
    @patch.object(torch._inductor.config, 'estimate_op_runtime', get_snode_runtime_for_reorder_compute_test)
    def test_reorder_compute_for_overlap_custom_runtime_estimation(self):
        if False:
            print('Hello World!')

        def func(a, *, tag, ranks, group_size):
            if False:
                print('Hello World!')
            ar = _functional_collectives.all_reduce(a, 'sum', ranks, tag)
            g = torch.matmul(a, a)
            c = torch.relu(a)
            d = torch.matmul(c, c)
            f = d * c * ar
            fr = _functional_collectives.all_reduce(f, 'sum', ranks, tag)
            e = torch.matmul(d + ar + fr, g)
            return (e,)
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = torch.ones(4, 4, dtype=torch.float, device='cuda') + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            FileCheck().check('dist.all_reduce(').check('triton_poi_fused_relu').check('extern_kernels.mm(').check('extern_kernels.mm(').check('_wait_tensor(').check('triton_poi_fused_mul').check('dist.all_reduce(').check('_wait_tensor(').check('triton_poi_fused_add').check('extern_kernels.mm(').run(code)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    def test_nccl_heuristics(self):
        if False:
            return 10
        assert list(baseLat.shape) == [len(NCCL_ALGO), len(NCCL_PROTO)]
        assert list(hwLat.shape) == [len(NCCL_HW), len(NCCL_ALGO), len(NCCL_PROTO)]
        assert llMaxBws.shape[0] == len(NVIDIA_GPU_TYPE)
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    run_tests()