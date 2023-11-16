import copy
import math
import os
import random
import re
import signal
import sys
import tempfile
import threading
import pickle
import time
import warnings
from contextlib import contextmanager
from datetime import timedelta
from itertools import chain, product
from unittest import mock
import torch
import torch.distributed as c10d
if not c10d.is_available() or not c10d.is_nccl_available():
    print('c10d NCCL not available, skipping tests', file=sys.stderr)
    sys.exit(0)
import test_c10d_common
import torch.distributed as dist
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from test_c10d_common import gpus_for_rank, DoubleGpuNet, ConvNet, ModuleForDdpCommHook
from torch import nn
from torch._C._distributed_c10d import OpType
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import MultiProcessTestCase, init_multigpu_helper, requires_nccl, requires_gloo, requires_nccl_version, skip_if_lt_x_gpu, get_timeout, skip_if_rocm, with_dist_debug_levels, with_nccl_blocking_wait
from torch.testing._internal.common_utils import TestCase, run_tests, retry_on_connect_failures, skipIfRocm, TEST_WITH_DEV_DBG_ASAN, TEST_WITH_ROCM, skip_but_pass_in_sandcastle, skip_but_pass_in_sandcastle_if
if TEST_WITH_DEV_DBG_ASAN:
    print('Skip ASAN as torch + multiprocessing spawn have known issues', file=sys.stderr)
    sys.exit(0)
BFLOAT16_AVAILABLE = torch.cuda.is_available() and (torch.version.cuda is not None and int(torch.version.cuda.split('.')[0]) >= 11 or torch.version.hip is not None)

class RendezvousEnvTest(TestCase):

    @retry_on_connect_failures
    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() == 0, 'No GPUs available, skipping test')
    def test_common_errors(self):
        if False:
            print('Hello World!')
        vars = {'WORLD_SIZE': '1', 'RANK': '0', 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(common.find_free_port())}

        class Env:

            def __init__(self, vars):
                if False:
                    while True:
                        i = 10
                self.env_patcher = mock.patch.dict(os.environ, vars, clear=True)

            def __enter__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.env_patcher.start()

            def __exit__(self, type, value, traceback):
                if False:
                    while True:
                        i = 10
                self.env_patcher.stop()

        def without(d, key):
            if False:
                return 10
            d = d.copy()
            d.pop(key)
            return d

        def withouts(d, keys):
            if False:
                i = 10
                return i + 15
            d = d.copy()
            for key in keys:
                d.pop(key)
            return d
        with Env(without(vars, 'WORLD_SIZE')):
            self.assertEqual(None, os.environ.get('WORLD_SIZE'))
            with self.assertRaisesRegex(ValueError, 'WORLD_SIZE expected'):
                gen = c10d.rendezvous('env://')
                next(gen)
            c10d.init_process_group(backend='nccl', world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()
        with Env(without(vars, 'RANK')):
            self.assertEqual(None, os.environ.get('RANK'))
            with self.assertRaisesRegex(ValueError, 'RANK expected'):
                gen = c10d.rendezvous('env://')
                next(gen)
            c10d.init_process_group(backend='nccl', rank=0)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()
        with Env(withouts(vars, ['RANK', 'WORLD_SIZE'])):
            self.assertEqual(None, os.environ.get('RANK'))
            self.assertEqual(None, os.environ.get('WORLD_SIZE'))
            c10d.init_process_group(backend='nccl', rank=0, world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()
        with Env(vars):
            c10d.init_process_group(backend='nccl')
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()
        with Env(without(vars, 'MASTER_ADDR')):
            self.assertEqual(None, os.environ.get('MASTER_ADDR'))
            with self.assertRaisesRegex(ValueError, 'MASTER_ADDR expected'):
                gen = c10d.rendezvous('env://')
                next(gen)
        with Env(without(vars, 'MASTER_PORT')):
            self.assertEqual(None, os.environ.get('MASTER_PORT'))
            with self.assertRaisesRegex(ValueError, 'MASTER_PORT expected'):
                gen = c10d.rendezvous('env://')
                next(gen)
        with Env(without(vars, 'WORLD_SIZE')):
            self.assertEqual(None, os.environ.get('WORLD_SIZE'))
            gen = c10d.rendezvous(f'env://?world_size={1}')
            (_, _, size) = next(gen)
            self.assertEqual(size, 1)
        with Env(without(vars, 'RANK')):
            self.assertEqual(None, os.environ.get('RANK'))
            gen = c10d.rendezvous(f'env://?rank={0}')
            (_, rank, _) = next(gen)
            self.assertEqual(rank, 0)
        with Env(withouts(vars, ['RANK', 'WORLD_SIZE'])):
            self.assertEqual(None, os.environ.get('RANK'))
            self.assertEqual(None, os.environ.get('WORLD_SIZE'))
            gen = c10d.rendezvous(f'env://?rank={0}&world_size={1}')
            (_, rank, size) = next(gen)
            self.assertEqual(rank, 0)
            self.assertEqual(size, 1)

class TimeoutTest(test_c10d_common.AbstractTimeoutTest, TestCase):

    @requires_nccl()
    @retry_on_connect_failures
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() == 0, 'No GPUs available, skipping test')
    def test_default_store_timeout_nccl(self):
        if False:
            while True:
                i = 10
        self._test_default_store_timeout('nccl')

class ProcessGroupNCCLNoGPUTest(TestCase):
    MAIN_PROCESS_RANK = 0

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.rank = self.MAIN_PROCESS_RANK
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile(delete=False)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() > 0, 'GPUs are available, skipping test')
    def test_init_no_gpus(self):
        if False:
            return 10
        store = c10d.FileStore(self.file.name, self.world_size)
        with self.assertRaisesRegex(ValueError, 'ProcessGroupNCCL is only supported with GPUs, no GPUs found!'):
            c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

class ProcessGroupNCCLTest(MultiProcessTestCase):

    def _create_process_group_nccl(self, store, opts):
        if False:
            return 10
        c10d.init_process_group('nccl', world_size=self.world_size, rank=self.rank, store=store, pg_options=opts)
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    def opts(self, high_priority_stream=False):
        if False:
            while True:
                i = 10
        opts = c10d.ProcessGroupNCCL.Options()
        opts.is_high_priority_stream = high_priority_stream
        return opts

    def setUp(self):
        if False:
            return 10
        super().setUp()
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        self._spawn_processes()

    def tearDown(self):
        if False:
            while True:
                i = 10
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        if False:
            while True:
                i = 10
        return 2

    @property
    def rank_to_GPU(self):
        if False:
            return 10
        return init_multigpu_helper(self.world_size, 'nccl')

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_empty_tensors(self):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_idx = self.rank_to_GPU[self.rank][0]
        xs = [torch.FloatTensor([]).cuda(local_device_idx)]
        pg.broadcast(xs).wait()
        self.assertEqual(0, xs[0].numel())
        pg.allreduce(xs).wait()
        self.assertEqual(0, xs[0].numel())
        pg.reduce(xs).wait()
        self.assertEqual(0, xs[0].numel())
        ys = [[torch.FloatTensor([]).cuda(local_device_idx) for _ in range(self.world_size)]]
        pg.allgather(ys, xs).wait()
        for y in ys[0]:
            self.assertEqual(0, y.numel())
        ys = [torch.FloatTensor([]).cuda(local_device_idx)]
        xs = [[torch.FloatTensor([]).cuda(local_device_idx) for _ in range(self.world_size)]]
        pg.reduce_scatter(ys, xs).wait()
        self.assertEqual(0, ys[0].numel())

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_broadcast_ops(self):
        if False:
            i = 10
            return i + 15
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())

        def broadcast(xs, rootRank, rootTensor):
            if False:
                for i in range(10):
                    print('nop')
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()
            return work.result()
        for i in range(self.world_size):
            x = torch.tensor([self.rank]).cuda(self.rank_to_GPU[self.rank][0])
            output = broadcast([x], i, 0)
            self.assertEqual(torch.tensor([i]), output[0])
            expected_tensor = torch.empty([i + 1, i + 1]).fill_(i + 1)
            xs = [torch.empty([i + 1, i + 1]).fill_(-1).cuda(device=device_idx) for device_idx in self.rank_to_GPU[self.rank]]
            for j in range(len(xs)):
                if self.rank == i:
                    xs[j] = expected_tensor.cuda(device=self.rank_to_GPU[self.rank][j])
                broadcast(xs, i, j)
                for tensor in xs:
                    self.assertEqual(tensor, expected_tensor)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_sparse_allreduce_ops(self):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        indices = torch.tensor([[0, 1]])
        values = torch.tensor([[1, 2, 0], [4, 0, 6]])
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(2, 3)).to(self.rank)
        try:
            work = pg.allreduce([sparse_tensor])
            work.wait()
            a = torch.tensor([[2, 4, 0], [8, 0, 12]]).to(self.rank)
            self.assertEqual(work.result()[0], a)
        except RuntimeError as e:
            if 'allreduce_sparse is only available in the NCCL experimental branch.' in str(e):
                pass
            else:
                raise

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_allreduce_ops(self):
        if False:
            for i in range(10):
                print('nop')
        store = c10d.FileStore(self.file_name, self.world_size)
        device_count = torch.cuda.device_count()
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_id = self.rank_to_GPU[self.rank][0]

        def allreduce(tensors, op):
            if False:
                print('Hello World!')
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            work = pg.allreduce(tensors, opts)
            work.wait()
        tensors = [torch.tensor([self.rank + 1]).cuda(local_device_id)]
        allreduce(tensors, c10d.ReduceOp.SUM)
        ndev = self.world_size
        self.assertEqual(torch.tensor([ndev * (ndev + 1) // 2]), tensors[0])
        if torch.cuda.nccl.version() >= (2, 10, 0):
            tensors = [torch.tensor([self.rank + 1.0]).cuda(local_device_id)]
            allreduce(tensors, c10d.ReduceOp.AVG)
            ndev = self.world_size
            self.assertEqual(torch.tensor([ndev * (ndev + 1.0) / (2.0 * ndev)]), tensors[0])
        if torch.cuda.nccl.version() >= (2, 11, 1):
            for dtype in (torch.half, torch.float, torch.double):
                for factor in (3.0, torch.tensor([5.0], device=local_device_id, dtype=dtype)):
                    tensors = [torch.tensor([self.rank + 1]).cuda(local_device_id).to(dtype=dtype)]
                    allreduce(tensors, c10d._make_nccl_premul_sum(factor))
                    self.assertEqual(factor * torch.tensor([self.world_size * (self.world_size + 1) / 2], dtype=dtype, device=local_device_id), tensors[0])
        tensors = [torch.tensor([self.rank + 1]).cuda(local_device_id)]
        allreduce(tensors, c10d.ReduceOp.PRODUCT)
        self.assertEqual(torch.tensor([math.factorial(self.world_size)]), tensors[0])
        tensors = [torch.tensor([self.rank + 1]).cuda(local_device_id)]
        allreduce(tensors, c10d.ReduceOp.MIN)
        self.assertEqual(torch.tensor([1]), tensors[0])
        tensors = [torch.tensor([self.rank + 1]).cuda(local_device_id)]
        allreduce(tensors, c10d.ReduceOp.MAX)
        self.assertEqual(torch.tensor([self.world_size]), tensors[0])
        for (op, err) in zip((c10d.ReduceOp.BAND, c10d.ReduceOp.BOR, c10d.ReduceOp.BXOR), ('ReduceOp.BAND', 'ReduceOp.BOR', 'ReduceOp.BXOR')):
            with self.assertRaisesRegex(ValueError, 'Cannot use ' + err + ' with NCCL'):
                allreduce(tensors, op)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_alltoall_ops_with_cudafree_race(self):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        opts = c10d.AllToAllOptions()
        local_device = f'cuda:{self.rank_to_GPU[self.rank][0]}'
        torch.cuda.set_device(local_device)
        input = torch.rand(1000, 1000, device=local_device)
        output = torch.rand(1000, 1000, device=local_device)
        race_tensors = []
        for _ in range(10):
            tmp = []
            for i in range(5):
                tmp.append(torch.rand(10 ** (3 + i), device=local_device))
            race_tensors.append(tmp)
        for i in range(10):
            race_tensors.pop()
            work = pg.alltoall_base(output, input, [], [], opts)
            torch.cuda.empty_cache()
            work.wait()
        torch.cuda.synchronize(local_device)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_allreduce_in_cudagraph(self):
        if False:
            for i in range(10):
                print('nop')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_idx = self.rank_to_GPU[self.rank][0]
        with torch.cuda.device(local_device_idx):
            xs = [torch.FloatTensor([1]).cuda(local_device_idx)]
            pg.allreduce(xs).wait()
            self.assertEqual(xs[0].item(), 2)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                pg.allreduce(xs).wait()
            self.assertEqual(xs[0].item(), 2)
            graph.replay()
            graph.replay()
            self.assertEqual(xs[0].item(), 8)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    @skipIfRocm()
    def test_nccl_watchdog_cudagraph(self):
        if False:
            i = 10
            return i + 15
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        rank = self.rank_to_GPU[self.rank][0]
        with torch.cuda.device(rank):
            for i in range(100):
                xs = [torch.FloatTensor([1]).cuda(rank)]
                ys = [torch.FloatTensor([4]).cuda(rank)]
                for _ in range(30):
                    pg.allreduce(xs[0]).wait()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    xs[0] += 0.0
                    pg.allreduce(xs[0]).wait()
                    pg.allreduce(xs[0]).wait()
                    pg.allreduce(xs[0]).wait()
                    xs[0] += 0.0
                for _ in range(1400):
                    graph.replay()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_reduce_ops(self):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_id = self.rank_to_GPU[self.rank][0]

        def reduce(xs, rootRank, rootTensor, op=None):
            if False:
                while True:
                    i = 10
            opts = c10d.ReduceOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            if op:
                opts.reduceOp = op
            work = pg.reduce(xs, opts)
            work.wait()
        for rt in range(self.world_size):
            tensors = [torch.tensor([self.rank + 1]).cuda(local_device_id)]
            reduce(tensors, rt, 0)
            if self.rank == rt:
                self.assertEqual(torch.tensor([self.world_size * (self.world_size + 1) // 2]), tensors[0])
            else:
                self.assertEqual(torch.tensor([self.rank + 1]), tensors[0])
            for (op, err) in zip((c10d.ReduceOp.BAND, c10d.ReduceOp.BOR, c10d.ReduceOp.BXOR), ('ReduceOp.BAND', 'ReduceOp.BOR', 'ReduceOp.BXOR')):
                with self.assertRaisesRegex(ValueError, 'Cannot use ' + err + ' with NCCL'):
                    reduce(tensors, self.rank, rt, op)
            if torch.cuda.nccl.version() >= (2, 11, 1):
                for factor in (3.0, torch.tensor([5.0], device=local_device_id)):
                    if isinstance(factor, torch.Tensor):
                        factor_ref = factor.cpu().item()
                    else:
                        factor_ref = factor
                    float_tensors = [torch.tensor([self.rank + 1.0], device=f'cuda:{local_device_id}')]
                    float_tensors_ref = [torch.tensor([(self.rank + 1.0) * factor_ref], device=f'cuda:{local_device_id}')]
                    reduce(float_tensors_ref, rt, 0)
                    reduce(float_tensors, rt, 0, c10d._make_nccl_premul_sum(factor))
                    if self.rank == rt:
                        self.assertEqual(float_tensors_ref[0], float_tensors[0])

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_allgather_ops(self):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_ids = self.rank_to_GPU[self.rank]

        def allgather(output_ts, input_ts):
            if False:
                print('Hello World!')
            work = pg.allgather(output_ts, input_ts)
            return work.wait()
        tensors = [torch.empty(2, 2).fill_(2).cuda(device=i) for i in local_device_ids]
        output_tensors = []
        expected_output = []
        output_per_gpu = [torch.empty(2, 2).fill_(-1)] * len(local_device_ids) * self.world_size
        expected_per_gpu = [torch.empty(2, 2).fill_(2)] * len(local_device_ids) * self.world_size
        for gpu in local_device_ids:
            output_tensors.append([t.cuda(device=gpu) for t in output_per_gpu])
            expected_output.append([t.cuda(device=gpu) for t in expected_per_gpu])
        result = allgather(output_tensors, tensors)
        self.assertEqual(output_tensors, expected_output)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_allgather_base_ops(self):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_id = self.rank_to_GPU[self.rank][0]

        def allgather_base(output_t, input_t):
            if False:
                for i in range(10):
                    print('nop')
            work = pg._allgather_base(output_t, input_t)
            work.wait()
        tensor = torch.tensor([self.rank]).cuda(local_device_id)
        output_t = torch.empty(self.world_size, dtype=tensor.dtype).cuda(local_device_id)
        allgather_base(output_t, tensor)
        self.assertEqual(torch.arange(self.world_size), output_t)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_allgather_base_basics(self):
        if False:
            for i in range(10):
                print('nop')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_id = self.rank_to_GPU[self.rank][0]

        def allgather_base(output_t, input_t):
            if False:
                while True:
                    i = 10
            work = pg._allgather_base(output_t, input_t)
            work.wait()
        with self.assertRaisesRegex(ValueError, 'output tensor size must be equal to world_size times input tensor size'):
            tensor = torch.tensor([self.rank]).cuda(local_device_id)
            output_t = torch.empty(self.world_size + 1, dtype=tensor.dtype).cuda(local_device_id)
            allgather_base(output_t, tensor)
        with self.assertRaisesRegex(TypeError, 'output tensor must have the same type as input tensor'):
            tensor = torch.tensor([self.rank], dtype=torch.float).cuda(local_device_id)
            output_t = torch.empty(self.world_size + 1, dtype=torch.long).cuda(local_device_id)
            allgather_base(output_t, tensor)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_gather_ops(self):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        def gather(output_t, input_t, rootRank):
            if False:
                i = 10
                return i + 15
            opts = c10d.GatherOptions()
            opts.rootRank = rootRank
            if rootRank == self.rank:
                work = pg.gather(output_t, input_t, opts)
            else:
                work = pg.gather([], input_t, opts)
            work.wait()
        tensors = []
        for device_id in local_device_ids:
            tensors.append(torch.tensor([self.rank]).cuda(device_id))
        output_ts = []
        for idx in range(num_gpus):
            gpu_idx = local_device_ids[idx]
            output_ts.append([])
            for rank in range(self.world_size):
                output_ts[idx].append(torch.tensor([-1]).cuda(gpu_idx))
        expected = [[torch.tensor([rank]) for rank in range(self.world_size)]]
        for rank in range(self.world_size):
            gather(output_ts, tensors, rank)
            if rank == self.rank:
                self.assertEqual(expected, output_ts)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_gather_stress(self):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        def gather(output_t, input_t, rootRank):
            if False:
                print('Hello World!')
            opts = c10d.GatherOptions()
            opts.rootRank = rootRank
            if rootRank == self.rank:
                work = pg.gather(output_t, input_t, opts)
            else:
                work = pg.gather([], input_t, opts)
            work.wait()
        stress_length = 1000
        tensors = []
        for i in range(stress_length):
            tensors.append([])
            for device_id in local_device_ids:
                tensors[i].append(torch.tensor([self.rank]).cuda(device_id))
        output_ts = []
        for i in range(stress_length):
            output_ts.append([[] for _ in range(num_gpus)])
            for (idx, ls) in enumerate(output_ts[i]):
                gpu_idx = local_device_ids[idx]
                for _ in range(self.world_size):
                    ls.append(torch.tensor([-1]).cuda(gpu_idx))
        expected = [[torch.tensor([rank]) for rank in range(self.world_size)]]
        for i in range(stress_length):
            for rank in range(self.world_size):
                gather(output_ts[i], tensors[i], rank)
                if rank == self.rank:
                    self.assertEqual(output_ts[i], expected)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_gather_checks(self):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)
        tensors = []
        for device_id in local_device_ids:
            tensors.append(torch.tensor([self.rank]).cuda(device_id))
        output_ts = []
        for idx in range(num_gpus):
            gpu_idx = local_device_ids[idx]
            output_ts.append([])
            for rank in range(self.world_size):
                output_ts[idx].append(torch.tensor([-1]).cuda(gpu_idx))
        with self.assertRaisesRegex(ValueError, 'invalid root rank'):
            opts = c10d.GatherOptions()
            opts.rootRank = -1
            pg.gather(output_ts, tensors, opts)
        with self.assertRaisesRegex(TypeError, 'incompatible function arguments'):
            pg.gather(output_ts, tensors, 0)
        with self.assertRaisesRegex(ValueError, 'invalid root rank'):
            opts = c10d.GatherOptions()
            opts.rootRank = self.world_size
            pg.gather(output_ts, tensors, opts)
        with self.assertRaisesRegex(RuntimeError, 'There were no tensor arguments to this function'):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather(output_ts, [], opts)
        with self.assertRaisesRegex(ValueError, 'Tensors must be on distinct GPU devices'):
            tensors2 = []
            for device_id in local_device_ids:
                tensors2.append(torch.tensor([self.rank]).cuda(device_id))
                tensors2.append(torch.tensor([self.rank]).cuda(device_id))
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather(output_ts, tensors2, opts)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_scatter_ops(self):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        def scatter(output_t, input_t, rootRank):
            if False:
                i = 10
                return i + 15
            opts = c10d.ScatterOptions()
            opts.rootRank = rootRank
            if rootRank == self.rank:
                work = pg.scatter(output_t, input_t, opts)
            else:
                work = pg.scatter(output_t, [], opts)
            work.wait()
        tensors = []
        for device_id in local_device_ids:
            tensors.append(torch.tensor([-1]).cuda(device_id))
        scatter_list = []
        for idx in range(num_gpus):
            gpu_idx = local_device_ids[idx]
            scatter_list.append([])
            for rank in range(self.world_size):
                scatter_list[idx].append(torch.tensor([rank]).cuda(gpu_idx))
        expected = [torch.tensor([self.rank])]
        for rank in range(self.world_size):
            scatter(tensors, scatter_list, rank)
            self.assertEqual(expected, tensors)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_scatter_stress(self):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        def scatter(output_t, input_t, rootRank):
            if False:
                return 10
            opts = c10d.ScatterOptions()
            opts.rootRank = rootRank
            if rootRank == self.rank:
                work = pg.scatter(output_t, input_t, opts)
            else:
                work = pg.scatter(output_t, [], opts)
            work.wait()
        stress_length = 1000
        tensors = []
        for i in range(stress_length):
            tensors.append([])
            for device_id in local_device_ids:
                tensors[i].append(torch.tensor([-1]).cuda(device_id))
        scatter_list = []
        for i in range(stress_length):
            scatter_list.append([[] for _ in range(num_gpus)])
            for (idx, ls) in enumerate(scatter_list[i]):
                gpu_idx = local_device_ids[idx]
                for rank in range(self.world_size):
                    ls.append(torch.tensor([rank]).cuda(gpu_idx))
        expected = [torch.tensor([self.rank])]
        for i in range(stress_length):
            for rank in range(self.world_size):
                scatter(tensors[i], scatter_list[i], rank)
                self.assertEqual(tensors[i], expected)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_scatter_checks(self):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)
        tensors = []
        for device_id in local_device_ids:
            tensors.append(torch.tensor([-1]).cuda(device_id))
        scatter_list = []
        for idx in range(num_gpus):
            gpu_idx = local_device_ids[idx]
            scatter_list.append([])
            for rank in range(self.world_size):
                scatter_list[idx].append(torch.tensor([rank]).cuda(gpu_idx))
        with self.assertRaisesRegex(ValueError, 'invalid root rank'):
            opts = c10d.ScatterOptions()
            opts.rootRank = -1
            pg.scatter(tensors, scatter_list, opts)
        with self.assertRaisesRegex(TypeError, 'incompatible function arguments'):
            pg.scatter(tensors, scatter_list, 0)
        with self.assertRaisesRegex(ValueError, 'invalid root rank'):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.world_size
            pg.scatter(tensors, scatter_list, opts)
        with self.assertRaisesRegex(RuntimeError, 'There were no tensor arguments to this function'):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([], scatter_list, opts)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_reduce_scatter_base_basics(self):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_id = self.rank_to_GPU[self.rank][0]

        def reduce_scatter_base(output_t, input_t):
            if False:
                print('Hello World!')
            work = pg._reduce_scatter_base(output_t, input_t)
            work.wait()
        with self.assertRaisesRegex(ValueError, 'input tensor must be the same size as output size times world size'):
            input_t = torch.tensor([self.rank]).cuda(local_device_id)
            output_t = torch.empty(self.world_size + 1, dtype=input_t.dtype).cuda(local_device_id)
            reduce_scatter_base(output_t, input_t)
        with self.assertRaisesRegex(TypeError, 'input tensor must be the same type as the output tensor.'):
            tensor = torch.tensor([self.rank], dtype=torch.float).cuda(local_device_id)
            output_t = torch.empty(self.world_size + 1, dtype=torch.long).cuda(local_device_id)
            reduce_scatter_base(output_t, tensor)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_reduce_scatter_ops(self):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        def reduce_scatter(outputs, input_lists, op):
            if False:
                while True:
                    i = 10
            opts = c10d.ReduceScatterOptions()
            opts.reduceOp = op
            work = pg.reduce_scatter(outputs, input_lists, opts)
            work.wait()
        output = [torch.tensor([0]).cuda(i) for i in local_device_ids]
        tensor_lists = []
        input_per_gpu = []
        for i in range(self.world_size):
            input_per_gpu.append(torch.tensor([self.rank + i + 1]))
        for gpu in local_device_ids:
            tensor_lists.append([t.cuda(device=gpu) for t in input_per_gpu])
        reduce_scatter(output, tensor_lists, c10d.ReduceOp.SUM)
        for i in range(num_gpus):
            expected = torch.tensor([(1 + self.world_size) * self.world_size // 2 + self.world_size * self.rank])
            self.assertEqual(expected, output[i])
        reduce_scatter(output, tensor_lists, c10d.ReduceOp.MIN)
        for i in range(num_gpus):
            expected = torch.tensor([self.rank + 1 + i])
            self.assertEqual(expected, output[i])
        reduce_scatter(output, tensor_lists, c10d.ReduceOp.MAX)
        for i in range(num_gpus):
            expected = torch.tensor([self.rank + self.world_size + i])
            self.assertEqual(expected, output[i])
        reduce_scatter(output, tensor_lists, c10d.ReduceOp.PRODUCT)

        def perm(n, k):
            if False:
                for i in range(10):
                    print('nop')
            prod_val = n
            for val in range(n - k + 1, n):
                prod_val *= val
            return prod_val
        for i in range(num_gpus):
            prod_val = perm(self.rank + self.world_size, self.world_size)
            expected = torch.tensor([prod_val])
            self.assertEqual(expected, output[i])
        output_tensor = torch.empty_like(input_per_gpu[0][0]).cuda(self.rank)
        input_list = [tensor[0].cuda(self.rank) for tensor in input_per_gpu]
        pg.reduce_scatter(output_tensor, input_list, c10d.ReduceOp.SUM).wait()
        expected = torch.tensor((1 + self.world_size) * self.world_size // 2 + self.world_size * self.rank)
        self.assertEqual(expected, output_tensor)
        pg.reduce_scatter(output_tensor, input_list, c10d.ReduceOp.MIN).wait()
        expected = torch.tensor(self.rank + 1)
        self.assertEqual(expected, output_tensor)
        pg.reduce_scatter(output_tensor, input_list, c10d.ReduceOp.MAX).wait()
        expected = torch.tensor(self.rank + self.world_size)
        self.assertEqual(expected, output_tensor)
        pg.reduce_scatter(output_tensor, input_list, c10d.ReduceOp.PRODUCT).wait()
        prod_val = self.rank + 1
        for k in range(1, self.world_size):
            prod_val = prod_val * (self.rank + 1 + k)
        expected = torch.tensor(prod_val)
        self.assertEqual(expected, output_tensor)
        if torch.cuda.nccl.version() >= (2, 11, 1):
            for factor in (3.0, torch.tensor([5.0], device=self.rank)):
                if isinstance(factor, torch.Tensor):
                    factor_ref = factor.cpu().item()
                else:
                    factor_ref = factor
                output = [t.float() for t in output]
                tensor_lists = [[t.float() for t in tl] for tl in tensor_lists]
                output_ref = [t.float() for t in output]
                tensor_lists_ref = [[t.float() * factor_ref for t in tl] for tl in tensor_lists]
                reduce_scatter(output, tensor_lists, c10d._make_nccl_premul_sum(factor))
                reduce_scatter(output_ref, tensor_lists_ref, c10d.ReduceOp.SUM)
                self.assertEqual(output_ref, output)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_reduce_scatter_base_ops(self):
        if False:
            for i in range(10):
                print('nop')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_id = self.rank_to_GPU[self.rank][0]

        def reduce_scatter_base(output_t, input_t):
            if False:
                while True:
                    i = 10
            work = pg._reduce_scatter_base(output_t, input_t)
            work.wait()
        output_t = torch.empty([1]).cuda(local_device_id)
        tensor = torch.arange(self.world_size, dtype=output_t.dtype).cuda(local_device_id)
        reduce_scatter_base(output_t, tensor)
        self.assertEqual(output_t[0], self.rank * self.world_size)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_barrier(self):
        if False:
            for i in range(10):
                print('nop')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_ids = self.rank_to_GPU[self.rank]

        def allreduce(tensors):
            if False:
                print('Hello World!')
            opts = c10d.AllreduceOptions()
            work = pg.allreduce(tensors, opts)
            return work
        tensors_list = [[] for _ in range(len(local_device_ids))]
        for i in range(1, len(local_device_ids) + 1):
            for j in range(i):
                tensors_list[i - 1].append(torch.tensor([j + 1]).cuda(local_device_ids[j]))
        works = []
        for tensors in tensors_list:
            work = allreduce(tensors)
            works.append(work)
        pg.barrier().wait()
        for i in range(1, len(local_device_ids) + 1):
            for j in range(i):
                self.assertEqual(torch.tensor([(j + 1) * self.world_size]), tensors_list[i - 1][j])

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_send_recv(self):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]
        torch.manual_seed(0)
        send_tensor = torch.rand(10, 10, device=device)
        if self.rank == 0:
            dist.send(send_tensor, 1)
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10, device=device)
            dist.recv(recv_tensor, 0)
            self.assertEqual(send_tensor, recv_tensor)
        send_tensor_view = send_tensor.t()
        if self.rank == 0:
            with self.assertRaisesRegex(ValueError, 'Tensors must be contiguous'):
                dist.send(send_tensor_view, 1)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 1, 'NCCL test requires 1 GPU')
    @skip_if_lt_x_gpu(1)
    def test_nccl_dist_backend_error(self):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        self._create_process_group_nccl(store, self.opts())
        with self.assertRaises(dist.DistBackendError) as cm:
            dist.broadcast(torch.tensor([1, 2, 3]).cuda(), 0)
        self.assertTrue(isinstance(cm.exception, dist.DistError))
        self.assertIsInstance(cm.exception, RuntimeError)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_abort_pg(self):
        if False:
            i = 10
            return i + 15
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '0'
        store = c10d.FileStore(self.file_name, self.world_size)
        self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]
        t = torch.rand(10, 10, device=device)
        dist.all_reduce(t)

        def abortpg():
            if False:
                print('Hello World!')
            c10d.distributed_c10d._get_default_group()._get_backend(torch.device(device))._shutdown()
        model = DistributedDataParallel(torch.nn.Linear(10, 10).to(device), device_ids=[device])
        model(t).sum().backward()
        if self.rank == 0:
            dist.all_reduce(t)
            thread = threading.Thread(target=abortpg)
            thread.start()
            t_cpu = t.cpu()
            thread.join()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_close_pg(self):
        if False:
            for i in range(10):
                print('nop')
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '0'
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]
        t = torch.rand(10, 10, device=device)
        pg.allreduce(t)
        dist.destroy_process_group()
        pg.allreduce([t])
        pg._get_backend(torch.device(device))._shutdown()
        with self.assertRaises(dist.DistBackendError):
            pg.allreduce([t])

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() == 0, 'No GPUs available, skipping test')
    def test_init_process_group_nccl_timeout(self):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        base_opts = dict(backend='nccl', store=store, rank=self.rank, world_size=self.world_size)

        def _check_nccl_timeout(expected_timeout):
            if False:
                print('Hello World!')
            pg = dist.distributed_c10d._get_default_group()
            options = pg._get_backend(torch.device(f'cuda:{self.rank}')).options
            self.assertEqual(options._timeout, expected_timeout)
        dist.init_process_group(**base_opts)
        _check_nccl_timeout(torch.distributed.constants.default_pg_nccl_timeout)
        dist.destroy_process_group()
        new_timeout = timedelta(seconds=123)
        dist.init_process_group(**base_opts, timeout=new_timeout)
        _check_nccl_timeout(new_timeout)
        dist.destroy_process_group()
        opts = dist.ProcessGroupNCCL.Options()
        opts._timeout = timedelta(seconds=123)
        with warnings.catch_warnings(record=True) as w:
            dist.init_process_group(**base_opts, pg_options=opts)
        _check_nccl_timeout(torch.distributed.constants.default_pg_nccl_timeout)
        dist.destroy_process_group()
        opts = dist.ProcessGroupNCCL.Options()
        opts._timeout = timedelta(seconds=123)
        dist.init_process_group(**base_opts, pg_options=opts, timeout=timedelta(seconds=1240))
        _check_nccl_timeout(timedelta(seconds=1240))
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_tensor_register_hook(self):
        if False:
            print('Hello World!')
        os.environ['NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK'] = '1'
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        local_device_id = self.rank_to_GPU[self.rank][0]

        def allgather_base(output_t, input_t):
            if False:
                for i in range(10):
                    print('nop')
            work = pg._allgather_base(output_t, input_t)
            work.wait()
        tensor = torch.tensor([self.rank]).cuda(local_device_id)
        output_t = torch.empty(self.world_size, dtype=tensor.dtype).cuda(local_device_id)
        allgather_base(output_t, tensor)
        self.assertEqual(torch.arange(self.world_size), output_t)

class DistributedDataParallelTest(test_c10d_common.CommonDistributedDataParallelTest, MultiProcessTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        self._spawn_processes()

    def _get_process_group(self):
        if False:
            for i in range(10):
                print('nop')
        store = self._get_store()
        c10d.init_process_group('nccl', store=store, rank=self.rank, world_size=self.world_size)
        return c10d.distributed_c10d._get_default_group()

    def _test_nccl_backend(self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False):
        if False:
            return 10
        process_group = self._get_process_group()
        self._test_ddp_with_process_group(process_group, devices, device_ids, multi_device, gradient_as_bucket_view)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_propagate_error_reason(self):
        if False:
            print('Hello World!')
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '0'
        os.environ['NCCL_BLOCKING_WAIT'] = '1'
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size, timeout=timedelta(seconds=15))
        pg_gloo = c10d.ProcessGroupGloo(store, self.rank, self.world_size)
        pg.barrier().wait(timedelta(seconds=5))
        if self.rank == 0:
            pg_gloo.barrier().wait()
        inp = torch.ones(1).cuda(self.rank)
        if self.rank != 0:
            with self.assertRaises(dist.DistBackendError):
                pg.allreduce([inp]).wait(timedelta(seconds=5))
            try:
                pg.allreduce([torch.ones(2).cuda(self.rank)]).wait()
            except dist.DistBackendError as e:
                self.assertTrue('aborted' in str(e))
            else:
                self.fail('Expected error to be raised!')
            pg_gloo.barrier().wait()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_multi_device_ids_not_allowed(self):
        if False:
            i = 10
            return i + 15
        int_devices = list(range(torch.cuda.device_count()))
        devices = [torch.device('cuda:' + str(i)) for i in int_devices]
        with self.assertRaisesRegex(ValueError, 'device_ids can only be None or contain a single element.'):
            self._test_nccl_backend(devices, int_devices)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_single_device_module_device_ids_None(self):
        if False:
            while True:
                i = 10
        self._test_nccl_backend(None, None)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_single_device_module_empty_device_ids(self):
        if False:
            print('Hello World!')
        self._test_nccl_backend(None, [])

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_backend_multi_device_module_device_ids_None(self):
        if False:
            print('Hello World!')
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device('cuda:' + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_1gpu_module_device_ids_integer_list(self):
        if False:
            print('Hello World!')
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device('cuda:' + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, int_devices)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_1gpu_module_device_ids_torch_device_list(self):
        if False:
            print('Hello World!')
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device('cuda:' + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, devices)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_backend_2gpu_module(self):
        if False:
            for i in range(10):
                print('nop')
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device('cuda:' + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(8)
    def test_nccl_backend_4gpu_module(self):
        if False:
            while True:
                i = 10
        int_devices = gpus_for_rank(self.world_size)[self.rank][:4]
        devices = [torch.device('cuda:' + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_ddp_multi_device_module_config(self):
        if False:
            print('Hello World!')
        gpus = gpus_for_rank(self.world_size)[self.rank]
        self.assertTrue(len(gpus) >= 2, 'expecting at least 2 gpus per process')
        process_group = self._get_process_group()
        gpus = gpus[:2]
        model = DoubleGpuNet(gpus)
        with self.assertRaisesRegex(ValueError, 'DistributedDataParallel device_ids and output_device arguments only work with single-device/multiple-device GPU modules or CPU modules'):
            ddp_model = DistributedDataParallel(model, output_device=gpus[1], process_group=process_group)
        with self.assertRaisesRegex(ValueError, 'device_ids can only be None or contain a single element.'):
            ddp_model = DistributedDataParallel(model, device_ids=gpus, process_group=process_group)
        with self.assertRaisesRegex(ValueError, 'input module must be on the same type of devices'):
            model.fc1 = model.fc1.cpu()
            ddp_model = DistributedDataParallel(model, process_group=process_group)
        model = model.cpu()
        with self.assertRaisesRegex(ValueError, 'device_ids can only be None or contain a single element.'):
            ddp_model = DistributedDataParallel(model, device_ids=gpus, process_group=process_group)

    def _test_fp16(self, gradient_as_bucket_view=False):
        if False:
            print('Hello World!')
        process_group = self._get_process_group()
        gpus = gpus_for_rank(self.world_size)[self.rank]
        model = nn.Linear(1, 1, bias=False).cuda(gpus[0]).half()
        nn.init.constant_(model.weight, 1)
        ddp_model = DistributedDataParallel(model, device_ids=[gpus[0]], process_group=process_group, bucket_cap_mb=0.001, gradient_as_bucket_view=gradient_as_bucket_view)
        input = torch.tensor([[2 ** 15]]).cuda(gpus[0]).half()
        ddp_model.train()
        output = ddp_model(input)
        loss = output.sum()
        loss.backward()
        self.assertFalse(any((torch.isinf(p.grad).any() for p in ddp_model.parameters())))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_fp16()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16_grad_is_view(self):
        if False:
            print('Hello World!')
        self._test_fp16(gradient_as_bucket_view=True)

    def _test_arbitrary_forward_return_value(self, gradient_as_bucket_view=False):
        if False:
            print('Hello World!')
        '\n        Note: this test can be sped up by only running it on a CPU module\n        once DistributedDataParallel supports them.\n        '
        process_group = self._get_process_group()

        class ForwardReturnValueModule(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x, fn):
                if False:
                    print('Hello World!')
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return fn(F.softmax(x, dim=1), F.softmax(self.fc3(x), dim=1))
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(ForwardReturnValueModule().float().to(device_id), device_ids=[device_id], process_group=process_group, gradient_as_bucket_view=gradient_as_bucket_view)
        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(device_id)

        def test(box, unbox):
            if False:
                print('Hello World!')
            output = model(input, fn=box)
            loss = criterion(unbox(output), target)
            loss.backward()
        test(box=lambda x, y: (x, y), unbox=lambda obj: obj[1])
        test(box=lambda x, y: ['foo', x, 'bar', y], unbox=lambda obj: obj[3])
        test(box=lambda x, y: ('foo', x, 'bar', y), unbox=lambda obj: obj[3])
        test(box=lambda x, y: {'foo': 'bar', 'a': x, 'b': y}, unbox=lambda obj: obj['b'])
        test(box=lambda x, y: ['foo', 'bar', {'a': x, 'b': y}], unbox=lambda obj: obj[2]['b'])
        test(box=lambda x, y: {'foo': 'bar', 'list': [0, x, 1, y]}, unbox=lambda obj: obj['list'][3])

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_arbitrary_forward_return_value(self):
        if False:
            while True:
                i = 10
        self._test_arbitrary_forward_return_value()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_arbitrary_forward_return_value_grad_is_view(self):
        if False:
            return 10
        self._test_arbitrary_forward_return_value(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_with_lazy_parameters(self):
        if False:
            while True:
                i = 10
        process_group = self._get_process_group()
        with self.assertRaisesRegex(RuntimeError, 'Modules with uninitialized parameters'):
            DistributedDataParallel(torch.nn.LazyLinear(10), process_group=process_group)

    def _test_find_unused_parameters_kwarg(self, gradient_as_bucket_view=False):
        if False:
            i = 10
            return i + 15
        '\n        Note: this test can be sped up by only running it on a CPU module\n        once DistributedDataParallel supports them.\n        '
        torch.cuda.set_device(self.rank)
        dist.init_process_group(backend='nccl', world_size=self.world_size, rank=self.rank, init_method=f'file://{self.file_name}')
        process_group = c10d.distributed_c10d._get_default_group()

        class FindUnusedParametersModule(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return (F.softmax(x, dim=1), self.fc3)
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(device_id)
        ddp_model = None

        def test_find_unused_parameters(find_unused_parameters, test_default=False, gradient_as_bucket_view=False):
            if False:
                for i in range(10):
                    print('nop')
            if test_default:
                model = DistributedDataParallel(FindUnusedParametersModule().float().to(device_id), device_ids=[device_id], process_group=process_group, gradient_as_bucket_view=gradient_as_bucket_view)
            else:
                model = DistributedDataParallel(FindUnusedParametersModule().float().to(device_id), device_ids=[device_id], process_group=process_group, find_unused_parameters=find_unused_parameters, gradient_as_bucket_view=gradient_as_bucket_view)
            nonlocal ddp_model
            ddp_model = model
            (output, fc3) = model(input)
            output = fc3(output)
            loss = criterion(output, target)
            loss.backward()
        try:
            test_find_unused_parameters(True, gradient_as_bucket_view=gradient_as_bucket_view)
        except Exception as ex:
            self.assertTrue(str(ex).startswith('Expected to mark a variable ready only once.'))
            unused_index = 2
            unused_index_str = f'Parameter at index {unused_index}'
            model = ddp_model.module
            for (module_name, module) in model.named_modules():
                if module == model.fc3:
                    for (parameter_name, _) in module.named_parameters(recurse=False):
                        unused_fqn = f'{module_name}.{parameter_name}'
                        break
            if dist.get_debug_level() != dist.DebugLevel.OFF:
                unused_index_str += f' with name {unused_fqn}'
            self.assertTrue(unused_index_str in str(ex))
        else:
            self.fail('Expected exception')
        dist.barrier(process_group)
        try:
            test_find_unused_parameters(False, gradient_as_bucket_view=gradient_as_bucket_view)
        except Exception as ex:
            self.fail('Unexpected exception: %s' % ex)
        try:
            test_find_unused_parameters(True, test_default=True, gradient_as_bucket_view=gradient_as_bucket_view)
        except Exception as ex:
            self.fail('Unexpected exception: %s' % ex)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=['DETAIL'])
    def test_find_unused_parameters_kwarg_debug_detail(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_find_unused_parameters_kwarg()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=['INFO'])
    def test_find_unused_parameters_kwarg_debug_info(self):
        if False:
            while True:
                i = 10
        self._test_find_unused_parameters_kwarg()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=['OFF'])
    def test_find_unused_parameters_kwarg_debug_off(self):
        if False:
            while True:
                i = 10
        self._test_find_unused_parameters_kwarg()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=['DETAIL'])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_detail(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=['INFO'])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_info(self):
        if False:
            while True:
                i = 10
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=['OFF'])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_off(self):
        if False:
            while True:
                i = 10
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    def _test_multiple_outputs_multiple_backward(self, gradient_as_bucket_view=False):
        if False:
            while True:
                i = 10
        '\n        Note: this test can be sped up by only running it on a CPU module\n        once DistributedDataParallel supports them.\n        '
        process_group = self._get_process_group()

        class MultipleOutputModule(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()

                def define_module():
                    if False:
                        while True:
                            i = 10
                    return nn.Sequential(nn.Linear(2, 10, bias=False), nn.ReLU(), nn.Linear(10, 4, bias=False), nn.ReLU())
                self.module0 = define_module()
                self.module1 = define_module()

            def forward(self, x):
                if False:
                    return 10
                return (F.softmax(self.module0(x), dim=1), F.softmax(self.module1(x), dim=1))
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(MultipleOutputModule().float().to(device_id), device_ids=[device_id], process_group=process_group, gradient_as_bucket_view=gradient_as_bucket_view)
        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(device_id)
        (output1, output2) = model(input)
        loss1 = criterion(output1, target)
        loss1.backward()
        loss2 = criterion(output2, target)
        loss2.backward()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_multiple_outputs_multiple_backward()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward_grad_is_view(self):
        if False:
            print('Hello World!')
        self._test_multiple_outputs_multiple_backward(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_no_grad(self):
        if False:
            i = 10
            return i + 15
        '\n        Note: this test can be sped up by only running it on a CPU module\n        once DistributedDataParallel supports them.\n        '
        process_group = self._get_process_group()

        class NoGradModule(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(NoGradModule().float().to(device_id), device_ids=[device_id], process_group=process_group)
        batch_size = 4
        input = torch.rand([batch_size, 2], dtype=torch.float)

        def check_no_grads():
            if False:
                print('Hello World!')
            for p in model.parameters():
                self.assertTrue(p.requires_grad)
                self.assertIsNone(p.grad)
        check_no_grads()
        with torch.no_grad():
            output = model(input)
            self.assertTrue(isinstance(output, torch.Tensor))
        check_no_grads()

    def _test_accumulate_gradients_module(self, gradient_as_bucket_view=False):
        if False:
            for i in range(10):
                print('nop')
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device('cuda:' + str(i)) for i in int_devices]
        process_group = self._get_process_group()
        global_batch_size = self.world_size
        (model, ddp_model, input, target) = self._prepare_single_device_module(process_group, devices, devices, global_batch_size, gradient_as_bucket_view)

        def step_model(model, input, target):
            if False:
                i = 10
                return i + 15
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target.to(output.device))
            loss.backward()
        with torch.no_grad():
            ddp_model.train()
            ddp_model.module(input)
        for iteration in range(4):
            step_model(model, input, target)
            if iteration % 2 == 0:
                step_model(ddp_model.module, input[self.rank:self.rank + 1], target[self.rank:self.rank + 1])
                for (i, j) in zip(model.parameters(), ddp_model.parameters()):
                    self.assertNotEqual(i.grad, j.grad)
            else:
                step_model(ddp_model, input[self.rank:self.rank + 1], target[self.rank:self.rank + 1])
                for (i, j) in zip(model.parameters(), ddp_model.parameters()):
                    self.assertEqual(i.grad, j.grad, rtol=1.3e-06, atol=5e-05)
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_accumulate_gradients_module(self):
        if False:
            i = 10
            return i + 15
        self._test_accumulate_gradients_module()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_accumulate_gradients_module_with_grad_is_view(self):
        if False:
            return 10
        self._test_accumulate_gradients_module(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_failure_recovery(self):
        if False:
            i = 10
            return i + 15
        process_group = self._get_process_group()
        recovery_filename = self.file_name + '_recovery'
        if self.rank == 0:
            open(recovery_filename, 'w').close()

        class TestModel(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = TestModel().float().to(device_id)
        ddp = DistributedDataParallel(model, device_ids=[device_id], process_group=process_group)
        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(device_id)
        for _ in range(6):
            output = ddp(input)
            loss = criterion(output, target)
            loss.backward()
        del ddp
        c10d.destroy_process_group(process_group)
        store = c10d.FileStore(recovery_filename, self.world_size)
        c10d.init_process_group('nccl', store=store, rank=self.rank, world_size=self.world_size)
        process_group = c10d.distributed_c10d._get_default_group()
        ddp = DistributedDataParallel(model, device_ids=[device_id], process_group=process_group)
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(device_id)
        for _ in range(6):
            output = ddp(input)
            loss = criterion(output, target)
            loss.backward()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_pass_default_pg(self):
        if False:
            while True:
                i = 10
        dist.init_process_group('nccl', init_method=f'file://{self.file_name}', world_size=self.world_size, rank=self.rank)
        default_pg = c10d.distributed_c10d._get_default_group()
        dist.destroy_process_group(default_pg)
        self.assertFalse(dist.is_initialized())

    def _test_grad_layout(self, replica_devices, layer_devs, local_batch_size):
        if False:
            while True:
                i = 10
        process_group = self._get_process_group()
        global_batch_size = local_batch_size * self.world_size
        bucketsizes = (1e-06, 25)
        layer_formats = ([torch.contiguous_format] * 4, [torch.channels_last] * 2 + [torch.contiguous_format] * 2, [torch.channels_last] * 4)
        layer_dtypes = ([torch.float] * 4, [torch.float] * 2 + [torch.half] * 2, [torch.half] * 4)
        input_dev = layer_devs[0] if isinstance(layer_devs, list) else layer_devs
        target_dev = layer_devs[-1] if isinstance(layer_devs, list) else layer_devs
        input = torch.randn((global_batch_size, 8, 8, 8), device=input_dev, dtype=torch.float)
        target = torch.randn((global_batch_size, 8, 4, 4), device=target_dev, dtype=torch.float)
        local_batch_start = self.rank * local_batch_size
        local_batch_end = (self.rank + 1) * local_batch_size

        @contextmanager
        def first_bucket_size(ddp_bucket_mb):
            if False:
                i = 10
                return i + 15
            old_DEFAULT_FIRST_BUCKET_BYTES = dist._DEFAULT_FIRST_BUCKET_BYTES
            dist._DEFAULT_FIRST_BUCKET_BYTES = int(ddp_bucket_mb * 1000000.0)
            try:
                yield
            finally:
                dist._DEFAULT_FIRST_BUCKET_BYTES = old_DEFAULT_FIRST_BUCKET_BYTES
        with torch.backends.cudnn.flags(enabled=True, deterministic=True, benchmark=False):
            for (formats, dtypes, bucketsize) in product(layer_formats, layer_dtypes, bucketsizes):
                with first_bucket_size(bucketsize):
                    model_msg = f'rank = {self.rank} formats = {formats} dtypes = {dtypes} bucketsize = {bucketsize} '
                    try:
                        m = ConvNet(layer_devs, formats, dtypes)
                        m_ddp = DistributedDataParallel(copy.deepcopy(m), device_ids=replica_devices, process_group=process_group, bucket_cap_mb=bucketsize)
                        opt = torch.optim.SGD(m.parameters(), lr=0.1)
                        opt_ddp = torch.optim.SGD(m_ddp.parameters(), lr=0.1)
                        has_half = any((p.dtype is torch.half for p in m.parameters()))
                        tol = 0.001 if has_half else 1e-05
                    except BaseException:
                        print('Caught exception during model creation for ' + model_msg, flush=True)
                        raise
                    for it in range(3):
                        iter_msg = f'iter = {it} ' + model_msg
                        named_msg = iter_msg
                        try:
                            F.mse_loss(m(input).float(), target).backward()
                            F.mse_loss(m_ddp(input[local_batch_start:local_batch_end]).float(), target[local_batch_start:local_batch_end]).backward()
                            for (i, ((layer_name, m_child), m_ddp_child)) in enumerate(zip(m.named_children(), m_ddp.module.children())):
                                named_msg = layer_name + '.weight' + ' ' + iter_msg
                                self.assertTrue(m_child.weight.grad.is_contiguous(memory_format=formats[i]), named_msg)
                                self.assertTrue(m_ddp_child.weight.grad.is_contiguous(memory_format=formats[i]), named_msg)
                                for (j, ((param_name, p), p_ddp)) in enumerate(zip(m_child.named_parameters(), m_ddp_child.parameters())):
                                    named_msg = layer_name + '.' + param_name + ' ' + iter_msg
                                    self.assertEqual(p.grad, p_ddp.grad, rtol=tol, atol=tol)
                            opt.step()
                            opt_ddp.step()
                            if it == 0:
                                for (p, p_ddp) in zip(m.parameters(), m_ddp.parameters()):
                                    p.grad = None
                                    p_ddp.grad = None
                            else:
                                m.zero_grad()
                                m_ddp.zero_grad()
                        except BaseException:
                            print('Caught exception during iterations at ' + named_msg, flush=True)
                            raise

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_grad_layout_1devicemodule_1replicaperprocess(self):
        if False:
            return 10
        dev0 = torch.device('cuda:' + str(gpus_for_rank(self.world_size)[self.rank][0]))
        replica_devices = [dev0]
        layer_devs = dev0
        local_batch_size = 8
        self._test_grad_layout(replica_devices, layer_devs, local_batch_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @skip_if_rocm
    def test_grad_layout_2devicemodule(self):
        if False:
            print('Hello World!')
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        dev0 = torch.device('cuda:' + str(int_devices[0]))
        dev1 = torch.device('cuda:' + str(int_devices[1]))
        replica_devices = None
        layer_devs = [dev0] * 2 + [dev1] * 2
        local_batch_size = 8
        self._test_grad_layout(replica_devices, layer_devs, local_batch_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_param_layout_mismatch_error(self):
        if False:
            print('Hello World!')
        process_group = self._get_process_group()
        dev0 = torch.device('cuda:' + str(gpus_for_rank(self.world_size)[self.rank][0]))
        layer_devs = dev0
        layer_formats = [torch.contiguous_format] * 4 if self.rank == 0 else [torch.channels_last] * 4
        layer_dtypes = [torch.float] * 4
        m = ConvNet(layer_devs, layer_formats, layer_dtypes)
        if self.rank == 0:
            m_ddp = DistributedDataParallel(m, device_ids=[dev0], process_group=process_group)
        else:
            with self.assertRaisesRegex(RuntimeError, '.* appears not to match strides of the same param in process 0'):
                m_ddp = DistributedDataParallel(m, device_ids=[dev0], process_group=process_group)

    def _gpu_model_with_ddp_comm_hook(self, process_group, hook=None, gradient_as_bucket_view=False, state=None, static_graph=False):
        if False:
            print('Hello World!')
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(ModuleForDdpCommHook().to(device_id), device_ids=[device_id], process_group=process_group, gradient_as_bucket_view=gradient_as_bucket_view, static_graph=static_graph)
        if hook is not None:
            gpu_model.register_comm_hook(state, hook)
        return gpu_model

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_future_passing_gpu_nccl(self):
        if False:
            i = 10
            return i + 15
        '\n        This unit test verifies whether the Future object is passed properly using nccl backend.\n        The hook callback function creates a Future object and sets a value to it.\n        '
        process_group = self._get_process_group()
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, self._simple_hook)
        self._run_and_verify_hook(gpu_model, 8, 2 * torch.ones(2, 2))

    def _test_ddp_comm_hook_allreduce_hook_nccl(self, gradient_as_bucket_view=False, static_graph=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        This unit test verifies whether a DDP communication hook that just calls\n        allreduce gives the same result with the case of no hook registered.\n        Without the then callback, the future_value in reducer is no longer\n        a PyObject, and this unit test verifies future_value is properly checked.\n        '
        process_group = self._get_process_group()

        def allreduce_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
            if False:
                print('Hello World!')
            tensors = [bucket.buffer() / self.world_size]
            return process_group.allreduce(tensors).get_future().then(lambda fut: fut.value()[0])
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, allreduce_hook, gradient_as_bucket_view, static_graph)
        self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_default_ddp_comm_hooks_nccl(self, gradient_as_bucket_view=False):
        if False:
            i = 10
            return i + 15
        '\n        This unit test verifies whether default Python DDP communication hooks ALLREDUCE, FP16_COMPRESS\n        and BF16_COMPRESS, can give the same result with the case of no hook registered.\n        '
        process_group = self._get_process_group()
        state = process_group
        hook_options = [default.allreduce_hook, default.fp16_compress_hook]
        if not TEST_WITH_ROCM and BFLOAT16_AVAILABLE and c10d.is_nccl_available() and (torch.cuda.nccl.version() >= (2, 10)):
            hook_options.append(default.bf16_compress_hook)
        for hook in hook_options:
            gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, hook, gradient_as_bucket_view, state)
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_fp16_compress_wrapper(self, gradient_as_bucket_view=False):
        if False:
            return 10
        '\n        This unit test verifies whether wrapping the ALLREDUCE and POWER_SGD hooks with\n        the FP16_WRAPPER can give the same result as when there is no hook registered.\n        '
        process_group = self._get_process_group()
        powerSGD_state = powerSGD.PowerSGDState(process_group=process_group)
        hook_args = [(powerSGD.powerSGD_hook, powerSGD_state), (default.allreduce_hook, process_group)]
        for (hook, state) in hook_args:
            gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, default.fp16_compress_wrapper(hook), gradient_as_bucket_view, state)
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_bf16_compress_wrapper(self, gradient_as_bucket_view=False):
        if False:
            i = 10
            return i + 15
        '\n        This unit test verifies whether wrapping the ALLREDUCE and POWER_SGD hooks with\n        the BF16_WRAPPER can give the same result as when there is no hook registered.\n        '
        process_group = self._get_process_group()
        powerSGD_state = powerSGD.PowerSGDState(process_group=process_group)
        hook_args = [(powerSGD.powerSGD_hook, powerSGD_state), (default.allreduce_hook, process_group)]
        for (hook, state) in hook_args:
            gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, default.bf16_compress_wrapper(hook), gradient_as_bucket_view, state)
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_powerSGD_ddp_comm_hook_nccl(self, gradient_as_bucket_view=False):
        if False:
            print('Hello World!')
        '\n        This unit test verifies whether Python DDP communication hook POWER_SGD\n        can give the same result with the case of no hook registered.\n        '
        process_group = self._get_process_group()
        for (use_error_feedback, warm_start, batch_tensors_with_same_shape) in product([True, False], [True, False], [True, False]):
            state = powerSGD.PowerSGDState(process_group=process_group, matrix_approximation_rank=1, use_error_feedback=use_error_feedback, warm_start=warm_start, batch_tensors_with_same_shape=batch_tensors_with_same_shape)
            for hook in [powerSGD.powerSGD_hook, powerSGD.batched_powerSGD_hook]:
                gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, hook, gradient_as_bucket_view, state)
                self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_builtin_ddp_comm_hooks_nccl(self, gradient_as_bucket_view=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        This unit test verifies whether built-in C++ DDP communication hooks ALLREDUCE and FP16_COMPRESS\n        can give the same result with the case of no hook registered.\n        '
        process_group = self._get_process_group()
        for comm_hook_type in [dist.BuiltinCommHookType.ALLREDUCE, dist.BuiltinCommHookType.FP16_COMPRESS]:
            gpu_model = self._gpu_model_with_builtin_ddp_comm_hook(process_group, comm_hook_type, gradient_as_bucket_view)
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_nccl(self):
        if False:
            i = 10
            return i + 15
        self._test_ddp_comm_hook_allreduce_hook_nccl()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_default_ddp_comm_hooks_nccl(self):
        if False:
            while True:
                i = 10
        self._test_default_ddp_comm_hooks_nccl()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16_compress_wrapper_nccl(self):
        if False:
            while True:
                i = 10
        self._test_fp16_compress_wrapper()

    @requires_nccl()
    @requires_nccl_version((2, 10), 'Need NCCL 2.10+ for BF16_COMPRESS')
    @skip_but_pass_in_sandcastle_if(not BFLOAT16_AVAILABLE, 'BFloat16 is only supported by CUDA 11+')
    @skip_if_lt_x_gpu(2)
    def test_bf16_compress_wrapper_nccl(self):
        if False:
            return 10
        self._test_bf16_compress_wrapper()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_builtin_ddp_comm_hooks_nccl(self):
        if False:
            print('Hello World!')
        self._test_builtin_ddp_comm_hooks_nccl()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_powerSGD_ddp_comm_hook_nccl(self):
        if False:
            i = 10
            return i + 15
        self._test_powerSGD_ddp_comm_hook_nccl()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_nccl_grad_is_view(self):
        if False:
            i = 10
            return i + 15
        self._test_ddp_comm_hook_allreduce_hook_nccl(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_nccl_static_graph(self):
        if False:
            return 10
        self._test_ddp_comm_hook_allreduce_hook_nccl(static_graph=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_default_ddp_comm_hooks_nccl_is_view(self):
        if False:
            i = 10
            return i + 15
        self._test_default_ddp_comm_hooks_nccl(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16_compress_wrapper_is_view(self):
        if False:
            print('Hello World!')
        self._test_fp16_compress_wrapper(gradient_as_bucket_view=True)

    @requires_nccl()
    @requires_nccl_version((2, 10), 'Need NCCL 2.10+ for BF16_COMPRESS')
    @skip_but_pass_in_sandcastle_if(not BFLOAT16_AVAILABLE, 'BFloat16 is only supported by CUDA 11+')
    @skip_if_lt_x_gpu(2)
    def test_bf16_compress_wrapper_is_view(self):
        if False:
            while True:
                i = 10
        self._test_bf16_compress_wrapper(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_builtin_ddp_comm_hooks_nccl_grad_is_view(self):
        if False:
            while True:
                i = 10
        self._test_builtin_ddp_comm_hooks_nccl(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_powerSGD_ddp_comm_hook_nccl_grad_is_view(self):
        if False:
            while True:
                i = 10
        self._test_powerSGD_ddp_comm_hook_nccl(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_with_then_hook_nccl(self):
        if False:
            print('Hello World!')
        '\n        This unit test verifies whether a DDP communication hook that calls allreduce and then\n        multiplies the result by ten and divides by two gives the expected result.\n        '
        process_group = self._get_process_group()

        def allreduce_with_then_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
            if False:
                for i in range(10):
                    print('nop')
            tensors = [bucket.buffer() / self.world_size]
            fut = process_group.allreduce(tensors).get_future()

            def mult(fut):
                if False:
                    print('Hello World!')
                return 10 * fut.value()[0]

            def div(fut):
                if False:
                    for i in range(10):
                        print('nop')
                return 0.5 * fut.value()
            return fut.then(mult).then(div)
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, allreduce_with_then_hook)
        self._run_and_verify_hook(gpu_model, 8, 1.25 * torch.ones(2, 2))

    class AcceptsParam(torch.nn.Module):

        def __init__(self, p, factor):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.a = p
            self.f = factor

        def forward(self, input):
            if False:
                return 10
            return input + self.a * self.f

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_weight_sharing(self):
        if False:
            i = 10
            return i + 15
        process_group = self._get_process_group()
        size = 2048 * 2048
        dev = self.rank
        world = self.world_size
        p = torch.nn.Parameter(torch.randn(size, requires_grad=True))
        for (try_set_to_none, use_bucket_view) in product((False, True), (False, True)):
            m = torch.nn.Sequential(self.AcceptsParam(p, dev + 1), self.AcceptsParam(p, dev + 1)).cuda(dev)
            m = torch.nn.parallel.DistributedDataParallel(m, bucket_cap_mb=1, gradient_as_bucket_view=use_bucket_view, device_ids=[dev], process_group=process_group)
            for i in range(3):
                m.zero_grad(set_to_none=try_set_to_none)
                m(1).sum().backward()
                analytic = torch.full_like(p, 2.0 * (world * (world + 1.0) / 2.0) / world, device=dev)
                for (name, p) in m.named_parameters():
                    self.assertEqual(p.grad, analytic, 'mismatch at ' + name + '.grad for ' + f'set_to_none = {try_set_to_none}, use_bucket_view = {use_bucket_view}')

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_packed_sequence(self):
        if False:
            while True:
                i = 10
        '\n        Tests that DDP with ``device_ids`` specified can run a forward and\n        backward pass with ``PackedSequence`` s with parity compared to a local\n        version of the model.\n        '
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = dist.init_process_group('nccl', world_size=self.world_size, rank=self.rank, store=store)
        seqs = ['sequence_sequence', 'seq', 'sequence']
        vocab = ['<pad>'] + sorted({ch for seq in seqs for ch in seq})
        vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]
        torch.manual_seed(0)
        embed = nn.Embedding(len(vocab), 4)
        lstm = nn.LSTM(input_size=4, hidden_size=2, batch_first=True).to(self.rank)
        lstm_ddp = DistributedDataParallel(copy.deepcopy(lstm), device_ids=[self.rank], process_group=process_group)
        for (p1, p2) in zip(lstm.parameters(), lstm_ddp.module.parameters()):
            self.assertEqual(p1, p2)
        seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))
        seq_tensor = torch.Tensor(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
        for (i, (seq, seq_len)) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[i, :seq_len] = torch.LongTensor(seq)
        (seq_lengths, permutation_idx) = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[permutation_idx]
        embedded_seq_tensor = embed(seq_tensor)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embedded_seq_tensor, seq_lengths, batch_first=True)
        packed_input_ddp = torch.nn.utils.rnn.pack_padded_sequence(embedded_seq_tensor.detach().clone(), seq_lengths, batch_first=True)
        (packed_output, (ht, ct)) = lstm(packed_input.to(self.rank))
        (packed_output_ddp, (ht_ddp, ct_ddp)) = lstm_ddp(packed_input_ddp)
        self.assertEqual(packed_output.data, packed_output_ddp.data)
        self.assertEqual(ht, ht_ddp)
        self.assertEqual(ct, ct_ddp)
        packed_output.data.sum().backward()
        packed_output_ddp.data.sum().backward()
        for (p1, p2) in zip(lstm.parameters(), lstm_ddp.parameters()):
            self.assertEqual(p1.grad, p2.grad)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_channels_last_contig(self):
        if False:
            while True:
                i = 10
        process_group = self._get_process_group()
        device = torch.device(f'cuda:{self.rank}')
        tensor = torch.ones((2, 16, 768, 1152), dtype=torch.float32, device=device).to(memory_format=torch.channels_last)
        process_group.broadcast([tensor]).wait()

class WorkHookTest(MultiProcessTestCase):

    @property
    def world_size(self):
        if False:
            for i in range(10):
                print('nop')
        return 2

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        os.environ['NCCL_ENABLE_TIMING'] = '1'
        self._spawn_processes()

    def tearDown(self):
        if False:
            return 10
        super().tearDown()
        del os.environ['NCCL_ENABLE_TIMING']
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _get_store(self):
        if False:
            i = 10
            return i + 15
        return dist.FileStore(self.file_name, self.world_size)

    def _get_process_group(self):
        if False:
            for i in range(10):
                print('nop')
        store = self._get_store()
        c10d.init_process_group('nccl', store=store, rank=self.rank, world_size=self.world_size)
        return c10d.distributed_c10d._get_default_group()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_on_completion_hook_broadcast(self):
        if False:
            while True:
                i = 10
        pg = self._get_process_group()
        num_hook_fired = 0
        durations: List[float] = []

        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            if False:
                return 10
            nonlocal num_hook_fired, durations
            num_hook_fired += 1
            durations.append(work_info.active_duration.total_seconds())
        pg._register_on_completion_hook(hook)
        tensor = torch.ones([2, 3]).cuda(self.rank) * self.rank
        pg.broadcast([tensor]).wait()
        pg.broadcast([tensor]).wait()
        c10d.destroy_process_group(pg)
        self.assertEqual(num_hook_fired, 2)
        self.assertEqual(len(durations), 2)
        for duration in durations:
            self.assertTrue(duration > 0)
        self.assertEqual(tensor, torch.zeros([2, 3]).cuda(self.rank))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_on_completion_hook_mixed_ops(self):
        if False:
            return 10
        pg = self._get_process_group()
        num_hook_fired = 0
        durations: List[float] = []

        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            if False:
                return 10
            nonlocal num_hook_fired, durations
            num_hook_fired += 1
            durations.append(work_info.active_duration.total_seconds())
        pg._register_on_completion_hook(hook)
        tensor = torch.ones([2, 3]).cuda(self.rank)
        tensor_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        pg.allreduce(tensor)
        pg.allgather(tensor_list, tensor)
        pg.allreduce(tensor)
        c10d.destroy_process_group(pg)
        self.assertEqual(num_hook_fired, 3)
        self.assertEqual(len(durations), 3)
        for duration in durations:
            self.assertTrue(duration > 0)
        self.assertEqual(tensor, torch.ones([2, 3]).cuda(self.rank) * self.world_size * self.world_size)
        self.assertEqual(tensor_list, [torch.ones([2, 3]).cuda(self.rank) * self.world_size for _ in range(self.world_size)])

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_on_completion_hook_with_ddp(self):
        if False:
            while True:
                i = 10
        pg = self._get_process_group()
        num_hook_fired: Dict[int, int] = {}
        durations: Dict[OpType, List[float]] = {}

        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            if False:
                print('Hello World!')
            nonlocal num_hook_fired, durations
            op_type = work_info.op_type
            if op_type not in num_hook_fired:
                num_hook_fired[op_type] = 0
                durations[op_type] = []
            num_hook_fired[op_type] += 1
            durations[op_type].append(work_info.active_duration.total_seconds())
        pg._register_on_completion_hook(hook)
        nlayers = 10
        net = nn.Sequential(*[nn.Linear(1000, 1000, bias=False) for _ in range(nlayers)]).to(self.rank)
        ddp = DistributedDataParallel(net, device_ids=[self.rank], process_group=pg, bucket_cap_mb=1)
        pg._wait_for_pending_works()
        self.assertTrue(num_hook_fired[OpType.BROADCAST] > 0)
        ctor_allreduce = num_hook_fired[OpType.ALLREDUCE] if OpType.ALLREDUCE in num_hook_fired else 0
        x = torch.zeros(2, 1000).cuda(self.rank)
        ddp(x).sum().backward()
        c10d.destroy_process_group(pg)
        self.assertTrue(OpType.ALLREDUCE in num_hook_fired)
        self.assertTrue(num_hook_fired[OpType.ALLREDUCE] - ctor_allreduce > 0)
        self.assertTrue(all((duration > 0 for duration in chain(*durations.values()))))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_on_completion_hook_all_gather_object(self):
        if False:
            while True:
                i = 10
        torch.cuda.set_device(self.rank)
        pg = self._get_process_group()
        num_hook_fired: Dict[int, int] = {}
        durations: Dict[OpType, List[float]] = {}

        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            if False:
                while True:
                    i = 10
            nonlocal num_hook_fired, durations
            op_type = work_info.op_type
            if op_type not in num_hook_fired:
                num_hook_fired[op_type] = 0
                durations[op_type] = []
            num_hook_fired[op_type] += 1
            durations[op_type].append(work_info.active_duration.total_seconds())
        pg._register_on_completion_hook(hook)
        obj = {'rank': self.rank, 'world_size': self.world_size}
        obj_list = [None for _ in range(self.world_size)]
        c10d.all_gather_object(obj_list, obj, group=pg)
        for (r, o) in enumerate(obj_list):
            self.assertTrue(isinstance(o, dict))
            self.assertTrue(set(o.keys()), {'rank', 'world_size'})
            self.assertEqual(o['rank'], r)
            self.assertEqual(o['world_size'], self.world_size)
        c10d.destroy_process_group(pg)
        self.assertTrue(OpType.ALLGATHER in num_hook_fired)
        self.assertEqual(len(num_hook_fired), 1)
        self.assertEqual(num_hook_fired[OpType.ALLGATHER], 2)
        self.assertTrue(all((duration > 0 for duration in durations[OpType.ALLGATHER])))

class NcclErrorHandlingTest(MultiProcessTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.skip_return_code_checks = [self.test_nccl_errors_blocking_abort.__wrapped__, self.test_nccl_errors_blocking_sigkill.__wrapped__, self.test_nccl_errors_blocking_sigterm.__wrapped__, self.test_nccl_errors_blocking_nonzero_exit.__wrapped__]
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        self._spawn_processes()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def op_timeout_sec(self):
        if False:
            i = 10
            return i + 15
        return 1

    @property
    def world_size(self):
        if False:
            i = 10
            return i + 15
        return 3

    @property
    def blocking_wait_error_msg(self):
        if False:
            print('Hello World!')
        return 'timeout'

    def _run_all_reduce(self, pg):
        if False:
            for i in range(10):
                print('nop')
        pg.allreduce(torch.rand(10).cuda(self.rank))

    @requires_nccl()
    @requires_nccl_version((2, 4, 0), 'Need NCCL 2.4+ for error checking')
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm
    @skip_but_pass_in_sandcastle('Test does not pass when run locally')
    def test_nccl_errors_nonblocking(self):
        if False:
            while True:
                i = 10
        prev_nccl_async_error_handling = os.environ.get('NCCL_ASYNC_ERROR_HANDLING', None)
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '0'
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        process_group.allreduce(torch.rand(10).cuda(self.rank))
        if self.rank == 0:
            work = process_group.allreduce(torch.rand(10).cuda(self.rank))
            work.wait()
            t = threading.Thread(target=self._run_all_reduce, args=(process_group,))
            t.daemon = True
            t.start()
            t.join(int(get_timeout(self.id()) / 5))
            self.assertTrue(t.is_alive())
        if prev_nccl_async_error_handling is not None:
            os.environ['NCCL_ASYNC_ERROR_HANDLING'] = prev_nccl_async_error_handling

    def _test_nccl_errors_blocking(self, func):
        if False:
            for i in range(10):
                print('nop')
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size, timeout=timedelta(seconds=10))
        process_group.allreduce(torch.rand(10).cuda(self.rank))
        if self.rank == 0:
            work = process_group.allreduce(torch.rand(10).cuda(self.rank))
            with self.assertRaisesRegex(dist.DistBackendError, self.blocking_wait_error_msg):
                work.wait(timeout=timedelta(seconds=self.op_timeout_sec))
            a = torch.rand(10).cuda(self.rank)
        elif self.rank == 1:
            del process_group
            func()

    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), 'Need NCCL 2.4+ for error checking')
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm
    def test_nccl_errors_blocking_clean_exit(self):
        if False:
            i = 10
            return i + 15
        self._test_nccl_errors_blocking(lambda : sys.exit(0))

    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), 'Need NCCL 2.4+ for error checking')
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm
    def test_nccl_errors_blocking_nonzero_exit(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_nccl_errors_blocking(lambda : sys.exit(1))

    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), 'Need NCCL 2.4+ for error checking')
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm
    @skip_but_pass_in_sandcastle('Frequently times out see https://github.com/pytorch/pytorch/issues/58920')
    def test_nccl_errors_blocking_abort(self):
        if False:
            return 10
        self._test_nccl_errors_blocking(lambda : os.abort())

    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), 'Need NCCL 2.4+ for error checking')
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm
    def test_nccl_errors_blocking_sigkill(self):
        if False:
            while True:
                i = 10
        self._test_nccl_errors_blocking(lambda : os.kill(os.getpid(), signal.SIGKILL))

    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), 'Need NCCL 2.4+ for error checking')
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm
    def test_nccl_errors_blocking_sigterm(self):
        if False:
            i = 10
            return i + 15
        self._test_nccl_errors_blocking(lambda : os.kill(os.getpid(), signal.SIGTERM))

    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), 'Need NCCL 2.4+ for error checking')
    @skip_if_lt_x_gpu(3)
    def test_nccl_blocking_wait_with_barrier(self):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size, timeout=timedelta(seconds=10))
        process_group.barrier().wait()
        if self.rank == 0:
            with self.assertRaisesRegex(dist.DistBackendError, self.blocking_wait_error_msg):
                process_group.barrier().wait(timeout=timedelta(seconds=self.op_timeout_sec))

    def _run_invalid_nccl_blocking_wait_env(self, val):
        if False:
            i = 10
            return i + 15
        os.environ['NCCL_BLOCKING_WAIT'] = val
        store = c10d.FileStore(self.file_name, self.world_size)
        with self.assertRaises(RuntimeError):
            process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(3)
    def test_invalid_nccl_blocking_wait_env(self):
        if False:
            while True:
                i = 10
        self._run_invalid_nccl_blocking_wait_env('abc')
        self._run_invalid_nccl_blocking_wait_env('-1')
        self._run_invalid_nccl_blocking_wait_env('2147483647')
        self._run_invalid_nccl_blocking_wait_env('4294967295')

    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_gloo()
    @skip_if_lt_x_gpu(3)
    def test_nccl_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size, timeout=timedelta(seconds=10))
        pg_gloo = c10d.ProcessGroupGloo(store, self.rank, self.world_size)
        failed_collective_timeout = timedelta(milliseconds=100)
        process_group.allreduce(torch.rand(10).cuda(self.rank)).wait(timeout=timedelta(seconds=5))
        if self.rank == 0:
            with self.assertRaisesRegex(dist.DistBackendError, self.blocking_wait_error_msg):
                process_group.allreduce(torch.rand(10).cuda(self.rank)).wait(timeout=failed_collective_timeout)
            pg_gloo.barrier().wait()
        else:
            try:
                pg_gloo.barrier().wait()
            except Exception as e:
                raise ValueError(f'Rank {self.rank} barrier timed out waiting for rank 0 with error: {str(e)}') from e

class CommTest(test_c10d_common.AbstractCommTest, MultiProcessTestCase):

    @property
    def device(self):
        if False:
            return 10
        return f'cuda:{self.rank}'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        self._spawn_processes()

    def tearDown(self):
        if False:
            return 10
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _test_broadcast_coalesced(self, process_group, device, root_rank):
        if False:
            while True:
                i = 10
        half = torch.float16
        if device == torch.device('cpu'):
            half = torch.float32
        target = torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float64, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)
        if self.rank == root_rank:
            tensors = [tensor.clone() for tensor in target]
        else:
            tensors = [torch.zeros_like(tensor) for tensor in target]
        if self.rank != root_rank:
            self.assertNotEqual(tensors, target)
        c10d._broadcast_coalesced(process_group, tensors, buffer_size=256, src=root_rank)
        if self.rank != root_rank:
            self.assertEqual(tensors, target)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_broadcast_coalesced_nccl(self):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(backend='nccl', store=store, rank=self.rank, world_size=self.world_size)
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device('cuda:%d' % self.rank)
        ranks = [0, 1]
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_all_reduce_coalesced_nccl(self):
        if False:
            for i in range(10):
                print('nop')
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(backend='nccl', store=store, rank=self.rank, world_size=self.world_size)
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device('cuda:%d' % self.rank)
        tensors = [torch.full((60 + i,), self.rank + 1 + i, device=device, dtype=torch.float) for i in range(5)]
        torch.distributed.all_reduce_coalesced(tensors, group=process_group)
        for (i, t) in enumerate(tensors):
            self.assertEqual(t, torch.full_like(t, self.world_size * (i + (self.world_size + 1.0) / 2.0)))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_default_pg_nccl(self):
        if False:
            for i in range(10):
                print('nop')
        torch.cuda.set_device(self.rank)
        self._test_sequence_num_set_default_pg(backend='nccl')

    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_sequence_num_incremented_nccl_default(self):
        if False:
            print('Hello World!')
        self._test_sequence_num_incremented_default_group('nccl')

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sequence_num_incremented_nccl_subgroup(self):
        if False:
            for i in range(10):
                print('nop')
        if self.world_size < 4:
            return skip_but_pass_in_sandcastle('Test requires world_size of at least 4')
        self._test_sequence_num_incremented_subgroup('nccl')

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_nccl_new_group(self):
        if False:
            i = 10
            return i + 15
        torch.cuda.set_device(self.rank)
        self._test_sequence_num_set_new_group(backend='nccl')

    def _test_pass_nccl_options(self, pg_opts):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group('nccl', world_size=self.world_size, rank=self.rank, store=store, pg_options=pg_opts)
        pg = c10d.new_group([0, 1], pg_options=pg_opts)
        t = torch.tensor([self.rank + 1] * 10).cuda(self.rank)
        pg.allreduce(t).wait()
        expected_tensor = torch.tensor([3] * 10).cuda(self.rank)
        self.assertEqual(expected_tensor, t)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_pass_nccl_options_high_priority_stream(self):
        if False:
            return 10
        pg_opts = c10d.ProcessGroupNCCL.Options()
        pg_opts.is_high_priority_stream = True
        self._test_pass_nccl_options(pg_opts)

    @requires_nccl()
    @requires_nccl_version((2, 17), 'Need NCCL 2.17+ for configuring NCCL communicators')
    @skip_if_lt_x_gpu(2)
    def test_pass_nccl_options_config(self):
        if False:
            return 10
        pg_opts = c10d.ProcessGroupNCCL.Options()
        pg_opts.config.max_ctas = 4
        pg_opts.config.min_ctas = 2
        pg_opts.config.cga_cluster_size = 2
        pg_opts.config.net_name = 'Socket'
        nccl_debug_file = tempfile.NamedTemporaryFile()
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_DEBUG_FILE'] = nccl_debug_file.name
        self._test_pass_nccl_options(pg_opts)
        nccl_debug_file_content = nccl_debug_file.read()
        max_ctas = re.search(b'Max CTAs.*(\\d+)|$', nccl_debug_file_content).group(1)
        min_ctas = re.search(b'Min CTAs.*(\\d+)|$', nccl_debug_file_content).group(1)
        cga_cluster_size = re.search(b'CGA cluster.*(\\d+)|$', nccl_debug_file_content).group(1)
        net_name = re.search(b'Using network.([a-zA-z]+)|$', nccl_debug_file_content).group(1)
        self.assertEqual(pg_opts.config.max_ctas, int(max_ctas))
        self.assertEqual(pg_opts.config.min_ctas, int(min_ctas))
        self.assertEqual(pg_opts.config.cga_cluster_size, int(cga_cluster_size))
        self.assertEqual(pg_opts.config.net_name, net_name.decode())

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_barrier(self):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size, store=store)
        t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
        c10d.all_reduce(t)
        expected_tensor = torch.tensor([3] * 10).cuda(2 * self.rank)
        self.assertEqual(expected_tensor, t)
        pg = c10d.new_group([0, 1])
        t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
        pg.allreduce(t).wait()
        self.assertEqual(expected_tensor, t)
        pg = c10d.new_group([0])
        if self.rank == 0:
            t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            expected_tensor = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            pg.allreduce(t).wait()
            self.assertEqual(expected_tensor, t)
        pg = c10d.new_group([1])
        if self.rank == 1:
            t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            expected_tensor = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            pg.allreduce(t).wait()
            self.assertEqual(expected_tensor, t)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_barrier_timeout(self):
        if False:
            print('Hello World!')
        os.environ['ENABLE_NCCL_HEALTH_CHECK'] = '1'
        store = c10d.FileStore(self.file_name, self.world_size)
        if self.rank == 0:
            with self.assertRaisesRegex(dist.DistBackendError, 'Health check failure'):
                c10d.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size, store=store, timeout=timedelta(seconds=10))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_barrier_device_ids(self):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size, store=store)
        c10d.barrier(device_ids=[self.rank])

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_barrier_device_ids_function_argument(self):
        if False:
            i = 10
            return i + 15
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size, store=store)
        with self.assertRaisesRegex(TypeError, 'Invalid function argument'):
            c10d.barrier(device_ids=self.rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=['DETAIL'])
    def test_nccl_warn_not_in_group_debug_detail(self):
        if False:
            i = 10
            return i + 15
        self._test_warn_not_in_group(backend='nccl')

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=['INFO'])
    def test_nccl_warn_not_in_group_debug_info(self):
        if False:
            print('Hello World!')
        self._test_warn_not_in_group(backend='nccl')

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=['OFF'])
    def test_nccl_warn_not_in_group_debug_off(self):
        if False:
            while True:
                i = 10
        self._test_warn_not_in_group(backend='nccl')

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nncl_rank_membership(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_rank_membership(backend='nccl')

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_tensor_dtype_mismatch(self):
        if False:
            i = 10
            return i + 15
        self._test_tensor_dtype_mismatch(backend='nccl')

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_tensor_dtype_complex(self):
        if False:
            print('Hello World!')
        self._test_tensor_dtype_complex(backend='nccl')

class CompilerTest(test_c10d_common.CompilerTest):

    @property
    def world_size(self):
        if False:
            print('Hello World!')
        return 2

    def _get_default_group(self):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size, store=store)
        return dist.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)
    def test_allreduce_work_wait_gpu(self):
        if False:
            i = 10
            return i + 15
        self._test_allreduce_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_allgather_work_wait_gpu(self):
        if False:
            while True:
                i = 10
        self._test_allgather_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_allgather_into_tensor_work_wait_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_allgather_into_tensor_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_work_wait_gpu(self):
        if False:
            print('Hello World!')
        self._test_reduce_scatter_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor_work_wait_gpu(self):
        if False:
            print('Hello World!')
        self._test_reduce_scatter_tensor_work_wait(torch.ones(4, 4, device=self.rank) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_broadcast_work_wait_gpu(self):
        if False:
            while True:
                i = 10
        self._test_broadcast_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_scatter_work_wait_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_scatter_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_alltoall_work_wait_gpu(self):
        if False:
            i = 10
            return i + 15
        self._test_alltoall_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_nested_comm_tensor_wrapping(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_nested_comm_tensor_wrapping(torch.ones(2, 2, device=self.rank) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_consecutive_comm_work_wait_gpu(self):
        if False:
            i = 10
            return i + 15
        self._test_consecutive_comm_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_base_k(self):
        if False:
            while True:
                i = 10
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group('nccl', world_size=self.world_size, rank=self.rank, store=store)
        output_tensor = torch.zeros(2, dtype=torch.int64).to(self.rank)
        input_tensors = torch.arange(self.world_size * 2, dtype=torch.int64).to(self.rank)
        input_tensors = torch.reshape(input_tensors, (self.world_size, 2))
        dist.reduce_scatter_tensor(output_tensor, input_tensors)
        self.assertEqual(output_tensor, input_tensors[self.rank] * self.world_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor_coalesced(self):
        if False:
            print('Hello World!')
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group('nccl', world_size=self.world_size, rank=self.rank, store=store)
        output_tensors = torch.zeros(2, 2).to(self.rank)
        input_tensors = [torch.ones(2, 2).to(self.rank) for _ in range(self.world_size)]
        with dist._coalescing_manager():
            for i in range(self.world_size):
                dist.reduce_scatter_tensor(output_tensors[i], input_tensors[i])
        self.assertEqual(output_tensors, input_tensors[self.rank] * self.world_size)

class NcclProcessGroupWithDispatchedCollectivesTests(test_c10d_common.ProcessGroupWithDispatchedCollectivesTests):

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_collectives(self):
        if False:
            print('Hello World!')
        self._test_collectives(backend='nccl')

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_allreduce_coalesced(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_allreduce_coalesced(backend='nccl')

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_all_to_all_single(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_all_to_all_single(backend='nccl')

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_allgather_base(self):
        if False:
            while True:
                i = 10
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group('nccl', world_size=self.world_size, rank=self.rank, store=store)
        device = 'cuda'
        tensor = torch.ones(10, 10, device=torch.device(device))
        output_tensor = torch.zeros(10, 10, device=torch.device(device))
        dist.all_gather_into_tensor(output_tensor, tensor)
        self.assertEqual(output_tensor, tensor)

class LargeCommTest(test_c10d_common.AbstractLargeCommTest, MultiProcessTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        self._spawn_processes()

    def tearDown(self):
        if False:
            while True:
                i = 10
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def device(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rank

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_new_group_local_sync(self):
        if False:
            while True:
                i = 10
        self._test_new_group_local_sync(backend='nccl')

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_new_group_local_sync_sanity_check(self):
        if False:
            while True:
                i = 10
        self._test_new_group_local_sync_sanity_check(backend='nccl')

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_new_group_local_sync_duplicated_pg(self):
        if False:
            i = 10
            return i + 15
        self._test_new_group_local_sync_duplicate_pg(backend='nccl')

class SparseCollective(MultiProcessTestCase):

    @property
    def world_size(self):
        if False:
            print('Hello World!')
        return 1

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        self._spawn_processes()

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    class ToyModel(nn.Module):

        def __init__(self, rank, vocab_size, embedding_dim):
            if False:
                while True:
                    i = 10
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, sparse=True).to(rank)
            self.linear = nn.Linear(embedding_dim, 1).to(rank)

        def forward(self, inputs):
            if False:
                return 10
            embedded = self.embedding(inputs)
            flattened = torch.mean(embedded, dim=1)
            output = self.linear(flattened)
            return output

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_ddp_set_sparse_metadata(self):
        if False:
            for i in range(10):
                print('nop')
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group('nccl', world_size=self.world_size, rank=self.rank, store=store)
        vocab_size = 5
        model = SparseCollective.ToyModel(self.rank, vocab_size=vocab_size, embedding_dim=10)
        ddp_model = DistributedDataParallel(model)
        inputs = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]]).to(self.rank)
        indices = torch.Tensor(list(range(vocab_size)))
        ddp_model._set_sparse_metadata({'embedding.weight': indices})
        try:
            output = ddp_model(inputs)
            loss = output.sum()
            loss.backward()
            self.assertTrue(ddp_model.module.embedding.weight.grad.indices, indices)
        except RuntimeError as e:
            if 'allreduce_sparse is only available in the NCCL experimental branch.' in str(e):
                pass
            else:
                raise

class NCCLTraceTest(MultiProcessTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        os.environ['TORCH_NCCL_TRACE_BUFFER_SIZE'] = '10'
        self._spawn_processes()

    @classmethod
    def _run(cls, parent_conn, rank: int, test_name: str, file_name: str, parent_pipe) -> None:
        if False:
            print('Hello World!')
        cls.parent = parent_conn
        super()._run(rank, test_name, file_name, parent_pipe)

    @property
    def local_device(self):
        if False:
            return 10
        return torch.device('cuda', self.rank_to_GPU[self.rank][0])

    def _join_processes(self, fn):
        if False:
            print('Hello World!')
        fn()
        super()._join_processes(fn)

    def _spawn_processes(self) -> None:
        if False:
            return 10
        proc = torch.multiprocessing.get_context('spawn').Process
        self.children_pipes = []
        parent_pipes = []
        for i in range(self.world_size):
            (parent_conn, child_conn) = torch.multiprocessing.Pipe()
            self.children_pipes.append(child_conn)
            parent_pipes.append(parent_conn)
        piter = iter(parent_pipes)

        def wrap(*positional, args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            args = (next(piter), *args)
            return proc(*positional, args=args, **kwargs)
        self._start_processes(wrap)

    def _create_process_group_nccl(self):
        if False:
            print('Hello World!')
        store = dist.FileStore(self.file_name, self.world_size)
        c10d.init_process_group('nccl', world_size=self.world_size, rank=self.rank, store=store)
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    def tearDown(self):
        if False:
            while True:
                i = 10
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        if False:
            print('Hello World!')
        return 2

    @property
    def rank_to_GPU(self):
        if False:
            print('Hello World!')
        return init_multigpu_helper(self.world_size, 'nccl')

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_short(self):
        if False:
            while True:
                i = 10
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        for i in range(2):
            f = pg.allreduce(a)
        f.wait()
        torch.cuda.synchronize(device=device)
        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
        self.assertEqual(len(t), 2)
        last = t[-1]
        self.assertEqual(last['state'], 'completed')
        self.assertIn('test_c10d_nccl.py', str(last['frames']))
        self.assertEqual(last['input_sizes'], ((3, 4),))
        self.assertEqual(last['output_sizes'], ((3, 4),))
        self.assertEqual(last['seq_id'], 2)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_long(self):
        if False:
            i = 10
            return i + 15
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        for i in range(2):
            xs = [torch.ones(3, 4, device=device)]
            pg.broadcast(xs).wait()
            pg.allreduce(xs).wait()
            pg.reduce(xs).wait()
            ys = [[torch.empty(3, 4, device=device) for _ in range(self.world_size)]]
            pg.allgather(ys, xs).wait()
            pg.reduce_scatter(xs, ys).wait()
            f = pg.allreduce(a)
        f.wait()
        torch.cuda.synchronize(device=device)
        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
        self.assertEqual(len(t), 10)
        first = t[0]
        last = t[-1]
        self.assertEqual(last['state'], 'completed')
        self.assertIn('test_c10d_nccl.py', str(last['frames']))
        self.assertEqual(last['input_sizes'], ((3, 4),))
        self.assertEqual(last['output_sizes'], ((3, 4),))
        self.assertEqual(last['seq_id'] - first['seq_id'], 9)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_trace_while_active(self):
        if False:
            print('Hello World!')
        if self.rank == self.MAIN_PROCESS_RANK:
            for c in self.children_pipes:
                self.assertEqual(c.recv(), 'next')
            for c in self.children_pipes:
                c.send('next')
            return
        pg = self._create_process_group_nccl()
        device = self.local_device
        with torch.cuda.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)
            pg.allreduce(a).wait()
            e = torch.cuda.Event()
            e.record()
            if self.rank != 0:
                pg.allreduce(a).wait()
            e.synchronize()
            t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
            if self.rank == 0:
                self.assertEqual(t[-1]['seq_id'], 1)
                self.assertEqual(t[-1]['state'], 'completed')
            else:
                self.assertEqual(t[-1]['seq_id'], 2)
                self.assertEqual(t[-1]['state'], 'started')
            self.parent.send('next')
            self.assertEqual('next', self.parent.recv())
            if self.rank == 0:
                pg.allreduce(a).wait()
            torch.cuda.synchronize(device=device)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(torch.cuda.device_count() < 2, 'NCCL test requires 2+ GPUs')
    def test_trace_while_stuck(self):
        if False:
            for i in range(10):
                print('nop')
        if self.rank == self.MAIN_PROCESS_RANK:
            for c in self.children_pipes:
                self.assertEqual(c.recv(), 'next')
            for c in self.children_pipes:
                c.send('next')
            return
        pg = self._create_process_group_nccl()
        device = self.local_device
        with torch.cuda.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)
            pg.allreduce(a).wait()
            e = torch.cuda.Event()
            e.record()

            def gather_trace():
                if False:
                    return 10
                e.synchronize()
                time.sleep(5)
                t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
                if self.rank == 0:
                    self.assertEqual(t[-1]['seq_id'], 1)
                    self.assertEqual(t[-1]['state'], 'completed')
                else:
                    self.assertEqual(t[-1]['seq_id'], 2)
                    self.assertEqual(t[-1]['state'], 'started')
                self.parent.send('next')
            if self.rank != 0:
                pg.allreduce(a).wait()
                th = threading.Thread(target=gather_trace)
                th.start()
                for i in range(2000):
                    a = a + a
                th.join()
            else:
                gather_trace()
            self.assertEqual('next', self.parent.recv())
            if self.rank == 0:
                pg.allreduce(a).wait()
            torch.cuda.synchronize(device=device)
if __name__ == '__main__':
    assert not torch.cuda._initialized, 'test_distributed must not have initialized CUDA context on main process'
    run_tests()