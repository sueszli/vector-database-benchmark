import os
import sys
import unittest
import weakref
from functools import partial, wraps
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
import torch.distributed._functional_collectives_impl as ft_c_impl
import torch.distributed._tensor as dt
import torch.distributed.distributed_c10d as c10d
from functorch import make_fx
from torch.testing import FileCheck
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._triton import has_triton
if not dist.is_available():
    print('Distributed not available, skipping tests', file=sys.stderr)
    sys.exit(0)
from torch.testing._internal.common_distributed import MultiProcessTestCase, MultiThreadedTestCase, requires_nccl, skip_if_lt_x_gpu, TEST_SKIPS
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase

def new_subgroups(group_size: int, pg_tag=None):
    if False:
        print('Hello World!')
    world_size = dist.get_world_size()
    subgroups = []
    cur_subgroup = None
    for subgroup_id in range(world_size // group_size):
        start_rank = subgroup_id * group_size
        end_rank = start_rank + group_size
        ranks_in_subgroup = list(range(start_rank, end_rank))
        subgroup = c10d._new_group_with_tag(ranks=ranks_in_subgroup, pg_tag=pg_tag)
        subgroups.append(subgroup)
        rank = dist.get_rank()
        if rank in ranks_in_subgroup:
            cur_subgroup = subgroup
    return (cur_subgroup, subgroups)

class TestExpand(MultiThreadedTestCase):

    @property
    def world_size(self):
        if False:
            return 10
        return 4

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self._spawn_threads()

    def test_expand_1d_rank_list(self):
        if False:
            i = 10
            return i + 15
        (tag, rankset, group_size) = ft_c._expand_group([0, 1, 2, 3])
        self.assertEqual('', tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(4, group_size)
        (tag, rankset, group_size) = ft_c._expand_group([0, 1, 2, 3], 'bla')
        self.assertEqual('bla', tag)

    def test_expand_2d_rank_list(self):
        if False:
            i = 10
            return i + 15
        (tag, rankset, group_size) = ft_c._expand_group([[0, 1], [2, 3]])
        self.assertEqual('', tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(2, group_size)
        (tag, rankset, group_size) = ft_c._expand_group([[0, 1], [2, 3]], 'blu')
        self.assertEqual('blu', tag)
        with self.assertRaisesRegex(ValueError, 'group sizes must be identical'):
            ft_c._expand_group([[0], [1, 2, 3]])

    def test_expand_process_group(self):
        if False:
            i = 10
            return i + 15
        (tag, rankset, group_size) = ft_c._expand_group(dist.group.WORLD)
        self.assertEqual(c10d._get_group_tag(dist.group.WORLD), tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(4, group_size)
        (tag, rankset, group_size) = ft_c._expand_group(dist.group.WORLD, 'bla')
        self.assertEqual('bla', tag)
        (my_pg, others) = new_subgroups(group_size=2)
        (tag, rankset, group_size) = ft_c._expand_group(my_pg)
        self.assertEqual(c10d._get_group_tag(my_pg), tag)
        self.assertEqual(dist.get_process_group_ranks(my_pg), rankset)
        self.assertEqual(2, group_size)
        my_pg = None
        for i in range(dist.get_world_size()):
            group = c10d._new_group_with_tag([i], pg_tag='my_pg')
            if i == dist.get_rank():
                my_pg = group
        (tag, rankset, group_size) = ft_c._expand_group(my_pg)
        self.assertEqual('my_pg', tag)
        self.assertEqual([dist.get_rank()], rankset)
        self.assertEqual(1, group_size)
        (tag, rankset, group_size) = ft_c._expand_group(my_pg, 'bla')
        self.assertEqual('bla', tag)

    def test_expand_device_mesh(self):
        if False:
            return 10
        mesh = dt.DeviceMesh('cpu', torch.arange(4))
        (tag, rankset, group_size) = ft_c._expand_group(mesh)
        self.assertEqual(c10d._get_group_tag(mesh.get_dim_groups()[0]), tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(4, group_size)
        mesh = dt.DeviceMesh('cpu', torch.arange(4))
        (tag, rankset, group_size) = ft_c._expand_group(mesh)
        self.assertEqual(c10d._get_group_tag(mesh.get_dim_groups()[0]), tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(4, group_size)

    def test_expand_device_mesh_tuple(self):
        if False:
            i = 10
            return i + 15
        mesh = dt.DeviceMesh('cpu', torch.arange(4).view(2, 2))
        with self.assertRaisesRegex(AssertionError, 'Only 1D mesh'):
            (tag, rankset, group_size) = ft_c._expand_group(mesh)
        (tag, rankset, group_size) = ft_c._expand_group((mesh, 0))
        self.assertEqual(c10d._get_group_tag(mesh.get_dim_groups()[0]), tag)
        expected_rankset = [0, 2] if dist.get_rank() in [0, 2] else [1, 3]
        self.assertEqual(expected_rankset, rankset)
        self.assertEqual(2, group_size)
        (tag, rankset, group_size) = ft_c._expand_group((mesh, 1))
        expected_rankset = [0, 1] if dist.get_rank() in [0, 1] else [2, 3]
        self.assertEqual(c10d._get_group_tag(mesh.get_dim_groups()[1]), tag)
        self.assertEqual(expected_rankset, rankset)
        self.assertEqual(2, group_size)

class TestPgTag(MultiThreadedTestCase):

    @property
    def world_size(self):
        if False:
            for i in range(10):
                print('nop')
        return 4

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self._spawn_threads()
    "\n    The behavior we want is as follow:\n\n    - rankset+tag will always result in the same PG.\n    Do we enforce this by failing creation of new PGs or returning existing ones?\n        Return existing one.\n\n    - default tag gives existing behavior.\n        This means we should create duplicates.\n    - _expand_group on _default-tagged pg should always resolve to it\n        This mean we can't depend on empty tag + rankset.\n    "

    def test_pg_creation_with_tag(self):
        if False:
            return 10
        (my_group, _) = new_subgroups(group_size=2, pg_tag='blu')
        (my_group2, _) = new_subgroups(group_size=2, pg_tag='blu')
        self.assertEqual(my_group, my_group2)
        (my_group3, _) = new_subgroups(group_size=2, pg_tag='blu2')
        self.assertNotEqual(my_group, my_group3)
        (my_group4, _) = new_subgroups(group_size=2)
        self.assertNotEqual(my_group, my_group4)
        (my_group5, _) = new_subgroups(group_size=2)
        self.assertNotEqual(my_group4, my_group5)

    def test_pg_lookup_roundtrip(self):
        if False:
            for i in range(10):
                print('nop')
        (pg_tag0, _) = new_subgroups(group_size=2, pg_tag='blu')
        (pg_tag1, _) = new_subgroups(group_size=2, pg_tag='blu2')
        (pg_notag0, _) = new_subgroups(group_size=2)
        (pg_notag1, _) = new_subgroups(group_size=2)

        def roundtrip(pg):
            if False:
                print('Hello World!')
            (tag, rankset, _) = ft_c._expand_group(pg)
            return c10d._find_pg_by_ranks_and_tag(tag, rankset)
        self.assertEqual(pg_tag0, roundtrip(pg_tag0))
        self.assertEqual(pg_tag1, roundtrip(pg_tag1))
        self.assertEqual(pg_notag0, roundtrip(pg_notag0))
        self.assertEqual(pg_notag1, roundtrip(pg_notag1))

    def test_pg_lookup_with_tag(self):
        if False:
            while True:
                i = 10
        (pg_tag0, _) = new_subgroups(group_size=2, pg_tag='blu')
        (pg_tag1, _) = new_subgroups(group_size=2, pg_tag='bla')
        (pg_notag0, _) = new_subgroups(group_size=2)

        def roundtrip(pg, pg_tag):
            if False:
                for i in range(10):
                    print('nop')
            (tag, rankset, _) = ft_c._expand_group(pg, pg_tag)
            return c10d._find_pg_by_ranks_and_tag(tag, rankset)
        self.assertEqual(pg_tag0, roundtrip(pg_tag1, 'blu'))
        self.assertEqual(pg_tag0, roundtrip(pg_notag0, 'blu'))
        self.assertEqual(pg_tag0, roundtrip(pg_tag0, ''))

    def test_find_or_create_pg(self):
        if False:
            return 10
        pg = c10d._find_or_create_pg_by_ranks_and_tag('blu', [0, 1, 2, 3], 2)
        (pg_tag0, _) = new_subgroups(group_size=2, pg_tag='blu')
        self.assertEqual(pg, pg_tag0)

    def test_find_root_pg(self):
        if False:
            while True:
                i = 10
        pg = c10d._find_pg_by_ranks_and_tag('', [0, 1, 2, 3])
        self.assertEqual(dist.group.WORLD, pg)

class TestTraceableCollectives(MultiThreadedTestCase):

    @property
    def world_size(self):
        if False:
            for i in range(10):
                print('nop')
        return 4

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self._spawn_threads()

    @parametrize('device', ['cpu', 'cuda'])
    def test_broadcast(self, device):
        if False:
            while True:
                i = 10
        if device == 'cuda':
            if torch.cuda.device_count() < self.world_size:
                self.skipTest('Not enough CUDA devices')
            torch.cuda.set_device(dist.get_rank())
        if dist.get_rank() == 0:
            tensor = torch.ones([4], device=device)
        else:
            tensor = torch.zeros([4], device=device)
        mesh = dt.DeviceMesh(device, torch.arange(4))
        res = ft_c.broadcast(tensor, 0, mesh)
        self.assertEqual(res, torch.ones([4], device=device))

    @parametrize('device', ['cpu', 'cuda'])
    def test_all_reduce_eager(self, device):
        if False:
            return 10
        if device == 'cuda':
            if torch.cuda.device_count() < self.world_size:
                self.skipTest('Not enough CUDA devices')
            torch.cuda.set_device(dist.get_rank())
        tensor = torch.ones([4], device=device)
        mesh = dt.DeviceMesh(device, torch.arange(4))
        res = ft_c.all_reduce(tensor, 'sum', mesh)
        self.assertEqual(res, torch.tensor([4, 4, 4, 4], dtype=torch.float))
        mesh = dt.DeviceMesh(device, torch.arange(4).view(2, 2))
        res2 = ft_c.all_reduce(tensor, 'sum', (mesh, 1))
        self.assertEqual(res2, torch.tensor([2, 2, 2, 2], dtype=torch.float))

    @parametrize('device', ['cpu', 'cuda'])
    def test_all_reduce_coalesced_eager(self, device):
        if False:
            print('Hello World!')
        if device == 'cuda':
            if torch.cuda.device_count() < self.world_size:
                self.skipTest('Not enough CUDA devices')
            torch.cuda.set_device(dist.get_rank())
        t0 = torch.ones([4], device=device)
        t1 = torch.ones([6], device=device) + 2
        mesh = dt.DeviceMesh(device, torch.arange(4))
        res = ft_c.all_reduce_coalesced([t0, t1], 'sum', mesh)
        self.assertEqual(res[0], t0 * 4)
        self.assertEqual(res[1], t1 * 4)

    @parametrize('device', ['cpu', 'cuda'])
    def test_all_gather_tensor(self, device):
        if False:
            i = 10
            return i + 15
        if device == 'cuda':
            if torch.cuda.device_count() < self.world_size:
                self.skipTest('Not enough CUDA devices')
            torch.cuda.set_device(dist.get_rank())
        mesh_1d = dt.DeviceMesh(device, torch.arange(self.world_size))
        mesh_2d = dt.DeviceMesh(device, torch.arange(self.world_size).view(2, 2))
        for mesh in [mesh_1d, mesh_2d]:
            dims_to_gather = [0, 1, 2]
            for dim in dims_to_gather:
                output_size = [3, 3, 3]
                output_size[dim] *= mesh.size(0)
                local_tensor = torch.ones([3, 3, 3], device=device)
                gathered_tensor = ft_c.all_gather_tensor(local_tensor, gather_dim=dim, group=(mesh, 0))
                self.assertEqual(gathered_tensor, torch.ones(output_size))

    @parametrize('device', ['cpu', 'cuda'])
    def test_all_gather_into_tensor_coalesced(self, device):
        if False:
            return 10
        if device == 'cuda':
            if torch.cuda.device_count() < self.world_size:
                self.skipTest('Not enough CUDA devices')
            torch.cuda.set_device(dist.get_rank())
        tensors = [torch.ones([4], device=device), torch.ones([4], device=device) + 1]
        mesh = dt.DeviceMesh(device, torch.arange(4))
        res = ft_c.all_gather_into_tensor_coalesced(tensors, mesh)
        self.assertEqual(2, len(res))
        self.assertEqual(torch.ones([4 * dist.get_world_size()], device=device), res[0])
        self.assertEqual(torch.ones([4 * dist.get_world_size()], device=device) + 1, res[1])

    @parametrize('device', ['cpu', 'cuda'])
    def test_reduce_scatter_tensor(self, device):
        if False:
            return 10
        if device == 'cuda':
            if torch.cuda.device_count() < self.world_size:
                self.skipTest('Not enough CUDA devices')
            torch.cuda.set_device(dist.get_rank())
        mesh_1d = dt.DeviceMesh(device, torch.arange(self.world_size))
        mesh_2d = dt.DeviceMesh(device, torch.arange(self.world_size).view(2, 2))
        for mesh in [mesh_1d, mesh_2d]:
            dims_to_scatter = [0, 1]
            for dim in dims_to_scatter:
                group_size = mesh.size(0)
                input_size = [3, 3]
                output_size = [3, 3]
                output_size[dim] *= group_size
                input_tensor = torch.ones(output_size, device=device)
                res_num = 1 * group_size
                rs_tensor = ft_c.reduce_scatter_tensor(input_tensor, 'sum', scatter_dim=dim, group=(mesh, 0))
                self.assertEqual(rs_tensor, torch.ones(input_size) * res_num)

    @parametrize('device', ['cpu', 'cuda'])
    def test_reduce_scatter_into_tensor_coalesced(self, device):
        if False:
            print('Hello World!')
        if device == 'cuda':
            if torch.cuda.device_count() < self.world_size:
                self.skipTest('Not enough CUDA devices')
            torch.cuda.set_device(dist.get_rank())
        tensors = [torch.ones([4], dtype=torch.int64, device=device), torch.ones([4], dtype=torch.int64, device=device) + 1]
        mesh = dt.DeviceMesh(device, torch.arange(4))
        res = ft_c.reduce_scatter_tensor_coalesced(tensors, 'sum', [0, 0], mesh)
        self.assertEqual(2, len(res))
        self.assertEqual(torch.tensor([4], device=device), res[0])
        self.assertEqual(torch.tensor([8], device=device), res[1])

class TestMetaCollectives(TestCase):

    def test_all_reduce(self):
        if False:
            print('Hello World!')
        x = torch.rand((2, 3, 4), device='meta')
        out = ft_c.all_reduce(x, 'sum', [1])
        self.assertEqual(x.size(), out.size())

class TestGradCollectives(MultiThreadedTestCase):

    @property
    def world_size(self):
        if False:
            while True:
                i = 10
        return 2

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self._spawn_threads()

    def test_all_reduce(self):
        if False:
            print('Hello World!')
        x = torch.rand([4], requires_grad=True)
        y = torch.rand([4], requires_grad=True)
        out = ft_c.all_reduce(x, 'sum', [0, 1])
        (out + y).sum().backward()
        self.assertIsNone(x.grad)

class TestMakeFx(MultiThreadedTestCase):

    @property
    def world_size(self):
        if False:
            while True:
                i = 10
        return 2

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self._spawn_threads()

    def test_all_reduce_tracing(self):
        if False:
            i = 10
            return i + 15

        def allred(input):
            if False:
                i = 10
                return i + 15
            return ft_c.all_reduce(input, 'sum', group=[0, 1]) + 1
        graph = make_fx(allred)(torch.rand(4))
        FileCheck().check('all_reduce').check('wait_tensor').run(str(graph.graph))
        mesh = dt.DeviceMesh('cpu', torch.arange(self.world_size))

        def allred_mesh(input):
            if False:
                for i in range(10):
                    print('nop')
            return ft_c.all_reduce(input, 'sum', mesh) + 1
        mesh_graph = make_fx(allred_mesh)(torch.rand(4))
        FileCheck().check_not('get_attr').check('wait_tensor').run(str(mesh_graph.graph))

        def allred_mesh_dim(input):
            if False:
                print('Hello World!')
            return ft_c.all_reduce(input, 'sum', (mesh, 0)) + 1
        mesh_dim_graph = make_fx(allred_mesh_dim)(torch.rand(4))
        FileCheck().check_not('get_attr').check('wait_tensor').run(str(mesh_dim_graph.graph))
instantiate_parametrized_tests(TestTraceableCollectives)
BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO
WORLD_SIZE = 2

def with_comms(func=None):
    if False:
        for i in range(10):
            print('nop')
    if func is None:
        return partial(with_comms)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if BACKEND == dist.Backend.NCCL and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f'multi-gpu-{self.world_size}'].exit_code)
        self.dist_init()
        func(self)
        self.destroy_comms()
    return wrapper

class TestCollectivesWithNCCL(MultiProcessTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['BACKEND'] = dist.Backend.NCCL
        self._spawn_processes()

    @property
    def device(self):
        if False:
            for i in range(10):
                print('nop')
        return torch.device(self.rank)

    @property
    def world_size(self):
        if False:
            while True:
                i = 10
        return WORLD_SIZE

    @property
    def process_group(self):
        if False:
            for i in range(10):
                print('nop')
        return dist.group.WORLD

    def dist_init(self):
        if False:
            while True:
                i = 10
        dist.init_process_group(backend=BACKEND, world_size=self.world_size, rank=self.rank, init_method=f'file://{self.file_name}')
        if BACKEND == 'nccl':
            torch.cuda.set_device(self.rank)

    def destroy_comms(self):
        if False:
            for i in range(10):
                print('nop')
        dist.barrier()
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @requires_nccl()
    @with_comms()
    def test_all_gather_into_tensor_coalesced(self):
        if False:
            i = 10
            return i + 15
        tensors = [torch.ones([4], device=f'cuda:{self.rank}'), torch.ones([4], device=f'cuda:{self.rank}') + 1]
        mesh = dt.DeviceMesh(f'cuda:{self.rank}', torch.arange(self.world_size))
        res = ft_c.all_gather_into_tensor_coalesced(tensors, mesh)
        self.assertEqual(2, len(res))
        self.assertEqual(torch.ones([4 * dist.get_world_size()]), res[0])
        self.assertEqual(torch.ones([4 * dist.get_world_size()]) + 1, res[1])

    @with_comms()
    def test_all_to_all_single(self):
        if False:
            return 10
        device = 'cuda' if BACKEND == dist.Backend.NCCL else 'cpu'
        mesh = dt.DeviceMesh(device, torch.arange(self.world_size))
        rank = dist.get_rank()
        row = self.world_size * (rank + 1) * (self.world_size + 1) / 2
        x = torch.ones(int(row), 5, device=device) * (rank + 1)
        split_sizes = [(i + 1) * (rank + 1) for i in range(self.world_size)]
        y = ft_c.all_to_all_single(x, output_split_sizes=split_sizes, input_split_sizes=split_sizes, group=mesh)
        expected = []
        for (idx, tensor) in enumerate(torch.split(x, split_sizes)):
            expected.append(torch.full_like(tensor, idx + 1))
        expected = torch.cat(expected)
        self.assertEqual(y, expected)

    @with_comms()
    def test_all_to_all_single_1d_input(self):
        if False:
            return 10
        device = 'cuda' if BACKEND == dist.Backend.NCCL else 'cpu'
        mesh = dt.DeviceMesh(device, torch.arange(self.world_size))
        rank = dist.get_rank()
        row = self.world_size * (rank + 1) * (self.world_size + 1) / 2
        x = torch.ones(int(row), device=device) * (rank + 1)
        split_sizes = [(i + 1) * (rank + 1) for i in range(self.world_size)]
        y = ft_c.all_to_all_single(x, output_split_sizes=split_sizes, input_split_sizes=split_sizes, group=mesh)
        expected = []
        for (idx, tensor) in enumerate(torch.split(x, split_sizes)):
            expected.append(torch.full_like(tensor, idx + 1))
        expected = torch.cat(expected)
        self.assertEqual(y, expected)

    @with_comms()
    def test_all_to_all_single_output_split_sizes_none(self):
        if False:
            for i in range(10):
                print('nop')
        device = 'cuda' if BACKEND == dist.Backend.NCCL else 'cpu'
        mesh = dt.DeviceMesh(device, torch.arange(self.world_size))
        rank = dist.get_rank()
        input_split_sizes = [1] * self.world_size
        x = torch.ones(self.world_size, self.world_size, device=device) * (rank + 1)
        y = ft_c.all_to_all_single(x, output_split_sizes=None, input_split_sizes=input_split_sizes, group=mesh)
        expected = []
        for (idx, tensor) in enumerate(torch.chunk(x, self.world_size)):
            expected.append(torch.full_like(tensor, idx + 1))
        expected = torch.cat(expected)
        self.assertEqual(y, expected)

    @with_comms()
    def test_all_to_all_single_input_split_sizes_none(self):
        if False:
            for i in range(10):
                print('nop')
        device = 'cuda' if BACKEND == dist.Backend.NCCL else 'cpu'
        mesh = dt.DeviceMesh(device, torch.arange(self.world_size))
        rank = dist.get_rank()
        output_split_sizes = [1] * self.world_size
        x = torch.ones(self.world_size, self.world_size, device=device) * (rank + 1)
        y = ft_c.all_to_all_single(x, output_split_sizes=output_split_sizes, input_split_sizes=None, group=mesh)
        expected = []
        for (idx, tensor) in enumerate(torch.chunk(x, self.world_size)):
            expected.append(torch.full_like(tensor, idx + 1))
        expected = torch.cat(expected)
        self.assertEqual(y, expected)

    @with_comms()
    def test_all_to_all_single_split_sizes_none(self):
        if False:
            for i in range(10):
                print('nop')
        device = 'cuda' if BACKEND == dist.Backend.NCCL else 'cpu'
        mesh = dt.DeviceMesh(device, torch.arange(self.world_size))
        rank = dist.get_rank()
        x = torch.ones(self.world_size, self.world_size, device=device) * (rank + 1)
        y = ft_c.all_to_all_single(x, output_split_sizes=None, input_split_sizes=None, group=mesh)
        expected = []
        for (idx, tensor) in enumerate(torch.chunk(x, self.world_size)):
            expected.append(torch.full_like(tensor, idx + 1))
        expected = torch.cat(expected)
        self.assertEqual(y, expected)

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    @skip_if_lt_x_gpu(WORLD_SIZE)
    @requires_nccl()
    @with_comms()
    def test_tracing(self):
        if False:
            print('Hello World!')

        def allreduce(t, pg):
            if False:
                return 10
            return ft_c.all_reduce(t, 'sum', pg)
        compiled_allreduce = torch.compile(allreduce, fullgraph=True)
        compiled_allreduce(torch.randn(8, device=self.device), self.process_group)

    @unittest.skipIf(not has_triton(), 'Inductor+gpu needs triton and recent GPU arch')
    def test_tracing_with_fakepg(self):
        if False:
            return 10

        def allreduce(t, pg):
            if False:
                print('Hello World!')
            return ft_c.all_reduce(t, 'sum', pg)
        compiled_allreduce = torch.compile(allreduce, fullgraph=True)
        dist.init_process_group(backend='fake', rank=0, world_size=8, store=FakeStore())
        allreduce(torch.randn(8, device=self.device), pg=dist.group.WORLD)

class TestOpWaitiness(MultiThreadedTestCase):

    @property
    def world_size(self):
        if False:
            return 10
        return 1

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self._spawn_threads()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        ft_c_impl._wait_all()

    def test_wait_reduce_outstanding_work_count(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(0, ft_c_impl._outstanding_wait_count())
        tensor = torch.ones([4])
        res = ft_c.all_reduce(tensor, 'sum', [0])
        self.assertEqual(1, ft_c_impl._outstanding_wait_count())
        self.assertTrue(ft_c_impl._tensor_needs_wait(res))
        res.trigger_wait()
        self.assertEqual(0, ft_c_impl._outstanding_wait_count())
        self.assertFalse(ft_c_impl._tensor_needs_wait(res))

    def test_add_triggers_wait(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(0, ft_c_impl._outstanding_wait_count())
        tensor = torch.ones([4])
        res = ft_c.all_reduce(tensor, 'sum', [0])
        self.assertEqual(1, ft_c_impl._outstanding_wait_count())
        self.assertTrue(ft_c_impl._tensor_needs_wait(res))
        foo = res + torch.ones([4])
        self.assertEqual(0, ft_c_impl._outstanding_wait_count())
        self.assertFalse(ft_c_impl._tensor_needs_wait(res))
        self.assertFalse(isinstance(foo, ft_c.AsyncCollectiveTensor))

    def test_view_does_not_trigger_wait(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(0, ft_c_impl._outstanding_wait_count())
        tensor = torch.ones([4])
        res = ft_c.all_reduce(tensor, 'sum', [0])
        self.assertEqual(1, ft_c_impl._outstanding_wait_count())
        self.assertTrue(ft_c_impl._tensor_needs_wait(res))
        foo = res.view([2, 2])
        self.assertEqual(1, ft_c_impl._outstanding_wait_count())
        self.assertTrue(ft_c_impl._tensor_needs_wait(res))
        self.assertTrue(ft_c_impl._tensor_needs_wait(foo))
        self.assertTrue(isinstance(foo, ft_c.AsyncCollectiveTensor))
        foo.trigger_wait()
        self.assertEqual(0, ft_c_impl._outstanding_wait_count())
        self.assertEqual(foo.tolist(), [[1.0, 1.0], [1.0, 1.0]])

    def test_dead_wrapper_triggers_wait(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(0, ft_c_impl._outstanding_wait_count())
        tensor = torch.ones([4])
        res = ft_c.all_reduce(tensor, 'sum', [0])
        wr = weakref.ref(res)
        self.assertTrue(wr() is not None)
        res = None
        self.assertTrue(wr() is None)
        self.assertEqual(0, ft_c_impl._outstanding_wait_count())

    def test_dead_wrapper_plus_view(self):
        if False:
            print('Hello World!')
        self.assertEqual(0, ft_c_impl._outstanding_wait_count())
        tensor = torch.ones([4])
        res = ft_c.all_reduce(tensor, 'sum', [0])
        res = res.view([2, 2])
        self.assertEqual(1, ft_c_impl._outstanding_wait_count())
        res = None
        self.assertEqual(0, ft_c_impl._outstanding_wait_count())
if __name__ == '__main__':
    run_tests()