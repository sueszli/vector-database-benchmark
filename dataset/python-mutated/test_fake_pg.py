import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import unittest
import torch.distributed._functional_collectives as funcol
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.testing import FileCheck
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._tensor import DeviceMesh, init_device_mesh
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.distributed.tensor.parallel import PairwiseParallel, SequenceParallel, parallelize_module
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
if not dist.is_available():
    print('Distributed not available, skipping tests', file=sys.stderr)
    sys.exit(0)
HAS_CUDA = torch.cuda.is_available()

class TestFakePG(TestCase):

    def tearDown(self):
        if False:
            return 10
        super().tearDown()
        dist.destroy_process_group()

    def test_all_reduce(self):
        if False:
            for i in range(10):
                print('nop')
        store = FakeStore()
        dist.init_process_group(backend='fake', rank=1, world_size=2, store=store)
        output = torch.ones(3, 3) * dist.get_rank()
        dist.all_reduce(output)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_allgather(self):
        if False:
            return 10
        store = FakeStore()
        dist.init_process_group(backend='fake', rank=1, world_size=2, store=store)
        input_tensor = torch.ones(3, 3) * dist.get_rank()
        output_tensors = [torch.empty_like(input_tensor) for _ in range(2)]
        dist.all_gather(output_tensors, input_tensor)
        for (_, out_tensor) in enumerate(output_tensors):
            self.assertEqual(tuple(out_tensor.shape), (3, 3))

    def test_reduce_scatter(self):
        if False:
            i = 10
            return i + 15
        store = FakeStore()
        dist.init_process_group(backend='fake', rank=1, world_size=2, store=store)
        to_reduce_scatter = [torch.ones(3, 3) * rank for rank in range(2)]
        output_tensor = torch.empty(3, 3)
        dist.reduce_scatter(output_tensor, to_reduce_scatter)
        self.assertEqual(tuple(output_tensor.shape), (3, 3))

    @unittest.skipIf(not HAS_CUDA, 'No CUDA')
    def test_construct_fsdp(self):
        if False:
            print('Hello World!')
        store = FakeStore()
        dist.init_process_group(backend='fake', rank=0, world_size=2, store=store)
        FSDP(nn.Linear(2, 3, device='cuda'))

    @unittest.skipIf(not HAS_CUDA, 'No CUDA')
    def test_fsdp_fake_e2e(self):
        if False:
            while True:
                i = 10
        store = dist.HashStore()
        dist.init_process_group(backend='fake', rank=0, world_size=2, store=store)
        my_module = nn.Sequential(nn.Linear(2, 3, device='cuda'), nn.ReLU(), nn.Linear(3, 2, device='cuda'))
        sharded_module = FSDP(my_module, use_orig_params=True)
        optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        input = torch.randn(2, 2)
        x = sharded_module(input)
        loss = x.sum()
        loss.backward()
        optim.step()

    @unittest.skipIf(not HAS_CUDA, 'No CUDA')
    def test_fake_pg_tracing(self):
        if False:
            print('Hello World!')
        store = dist.HashStore()
        dist.init_process_group(backend='fake', rank=0, world_size=2, store=store)
        default_pg = dist.distributed_c10d._get_default_group()

        def allgather_fn(tensor):
            if False:
                while True:
                    i = 10
            return funcol.all_gather_tensor(tensor, 0, default_pg)
        gm = make_fx(allgather_fn)(torch.randn(2, 2, device='cuda'))
        FileCheck().check('all_gather').check('wait_tensor').run(str(gm.graph))

    def test_broadcast(self):
        if False:
            for i in range(10):
                print('nop')
        store = FakeStore()
        dist.init_process_group(backend='fake', rank=0, world_size=2, store=store)
        output = torch.ones(3, 3)
        dist.broadcast(output, src=0)
        self.assertEqual(tuple(output.shape), (3, 3))
        output = torch.ones(3, 3)
        dist.broadcast(output, src=1)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_scatter(self):
        if False:
            for i in range(10):
                print('nop')
        store = FakeStore()
        dist.init_process_group(backend='fake', rank=0, world_size=2, store=store)
        output = torch.ones(3, 3)
        to_scatter = [torch.ones(3, 3) * rank for rank in range(2)]
        dist.scatter(output, to_scatter)
        self.assertEqual(tuple(output.shape), (3, 3))
        output = torch.ones(3, 3)
        dist.scatter(output, None, src=1)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_alltoall(self):
        if False:
            return 10
        store = FakeStore()
        dist.init_process_group(backend='fake', rank=0, world_size=2, store=store)
        output_list = [torch.ones(3, 3) for _ in range(2)]
        input_list = [torch.ones(3, 3) for _ in range(2)]
        dist.all_to_all(output_list, input_list)
        self.assertEqual(len(output_list), 2)
        for output in output_list:
            self.assertEqual(tuple(output.shape), (3, 3))

    def test_alltoall_base(self):
        if False:
            for i in range(10):
                print('nop')
        store = FakeStore()
        dist.init_process_group(backend='fake', rank=0, world_size=2, store=store)
        out_tensor = torch.ones(3, 3)
        in_tensor = torch.ones(3, 3)
        output_split = [1, 1]
        input_split = [1, 1]
        dist.all_to_all_single(out_tensor, in_tensor, output_split, input_split)
        self.assertEqual(tuple(out_tensor.shape), (3, 3))

    def test_send(self):
        if False:
            while True:
                i = 10
        store = FakeStore()
        dist.init_process_group(backend='fake', rank=0, world_size=2, store=store)
        tensor = torch.ones(3, 3)
        dist.send(tensor, 1)
        self.assertEqual(tuple(tensor.shape), (3, 3))

    def test_recv(self):
        if False:
            return 10
        store = FakeStore()
        dist.init_process_group(backend='fake', rank=0, world_size=2, store=store)
        output = torch.ones(3, 3)
        dist.recv(output, 1)
        self.assertEqual(tuple(output.shape), (3, 3))

    @unittest.skipIf(not HAS_CUDA, 'No CUDA or TP+FSDP')
    def test_fsdp_tp_fake_e2e(self):
        if False:
            for i in range(10):
                print('nop')
        world_size = 4
        tp_size = 2
        store = dist.HashStore()
        dist.init_process_group(backend='fake', rank=0, world_size=world_size, store=store)
        device_mesh = DeviceMesh('cuda', torch.arange(0, world_size).view(-1, tp_size))
        device_mesh = init_device_mesh('cuda', (world_size // tp_size, tp_size), mesh_dim_names=['dp', 'tp'])
        for parallel_style in [SequenceParallel(), PairwiseParallel()]:
            my_module = parallelize_module(MLPModule(device='cuda'), device_mesh['tp'], parallel_style)
            sharded_module = FSDP(my_module, use_orig_params=True, device_mesh=device_mesh['dp'])
            optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
            for i in range(10):
                dp_rank = dist.get_rank()
                torch.manual_seed(i + dp_rank)
                input = torch.randn(20, 10).cuda(dist.get_rank())
                x = sharded_module(input)
                loss = x.sum()
                loss.backward()
                optim.step()
if __name__ == '__main__':
    run_tests()