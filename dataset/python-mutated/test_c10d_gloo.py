import copy
import logging
import math
import operator
import os
import random
import sys
import tempfile
from functools import reduce
from itertools import groupby
import torch
import torch.distributed as c10d
if not c10d.is_available() or not c10d.is_gloo_available():
    print('c10d GLOO not available, skipping tests', file=sys.stderr)
    sys.exit(0)
import test_c10d_common
import torch.distributed as dist
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from test_c10d_common import gpus_for_rank, LOOPBACK, ModuleForDdpCommHook, SparseGradientModule, Task
from torch import nn
from torch.distributed._shard.sharded_tensor import init_from_local_shards, Shard, ShardedTensor, ShardMetadata
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import create_device, MultiProcessTestCase, requires_gloo, simple_sparse_reduce_tests, skip_if_lt_x_gpu, skip_if_win32, verify_ddp_error_logged
from torch.testing._internal.common_utils import retry_on_connect_failures, run_tests, skip_but_pass_in_sandcastle, TestCase

def simple_reduce_tests(rank, world_size):
    if False:
        return 10
    tests = [(c10d.ReduceOp.SUM, torch.tensor([rank + 1.0]), torch.tensor([float(world_size * (world_size + 1) / 2)])), (c10d.ReduceOp.PRODUCT, torch.tensor([rank + 1.0]), torch.tensor([float(math.factorial(world_size))])), (c10d.ReduceOp.MIN, torch.tensor([rank + 1.0]), torch.tensor([1.0])), (c10d.ReduceOp.MAX, torch.tensor([rank + 1.0]), torch.tensor([float(world_size)]))]
    for i in range(4):
        vin = rank | 1 << i
        vout = 1 << i
        tests.append((c10d.ReduceOp.BAND, torch.tensor([vin], dtype=torch.int32), torch.tensor([vout], dtype=torch.int32)))
    for i in range(1, 5):
        vin = reduce(operator.or_, [rank * i + j for j in range(i)])
        vout = reduce(operator.or_, range(world_size * i))
        tests.append((c10d.ReduceOp.BOR, torch.tensor([vin], dtype=torch.int32), torch.tensor([vout], dtype=torch.int32)))
    for i in range(1, 5):
        vin = reduce(operator.xor, [rank * i + j for j in range(i)])
        vout = reduce(operator.xor, range(world_size * i))
        tests.append((c10d.ReduceOp.BXOR, torch.tensor([vin], dtype=torch.int32), torch.tensor([vout], dtype=torch.int32)))
    return tests

def simple_coalesced_reduce_tests(rank, world_size):
    if False:
        i = 10
        return i + 15
    return [(c10d.ReduceOp.SUM, [torch.tensor([rank + 1.0]), torch.tensor([(rank + 1.0) ** 2])], [torch.tensor([float(world_size * (world_size + 1) / 2)]), torch.tensor([float(world_size * (world_size + 1) * (2 * world_size + 1) / 6)])]), (c10d.ReduceOp.PRODUCT, [torch.tensor([rank + 1.0]), torch.tensor([rank + 2.0])], [torch.tensor([float(math.factorial(world_size))]), torch.tensor([float(math.factorial(world_size + 1))])]), (c10d.ReduceOp.MIN, [torch.tensor([rank + x]) for x in [0.0, 1.0]], [torch.tensor([0.0]), torch.tensor([1.0])]), (c10d.ReduceOp.MAX, [torch.tensor([rank + x]) for x in [1.0, 2.0]], [torch.tensor([float(world_size)]), torch.tensor([world_size + 1.0])])]

def simple_multi_input_reduce_tests(rank, world_size):
    if False:
        for i in range(10):
            print('nop')
    return [(c10d.ReduceOp.SUM, [torch.tensor([2 * rank + 0.0]), torch.tensor([2 * rank + 1.0])], torch.tensor([float(world_size * (2 * world_size - 1))])), (c10d.ReduceOp.PRODUCT, [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])], torch.tensor([float(math.factorial(2 * world_size))])), (c10d.ReduceOp.MIN, [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])], torch.tensor([1.0])), (c10d.ReduceOp.MAX, [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])], torch.tensor([2.0 * world_size]))]

class RendezvousEnvTest(TestCase):

    @requires_gloo()
    @retry_on_connect_failures
    def test_logging_init(self):
        if False:
            print('Hello World!')
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(common.find_free_port())
        os.environ['RANK'] = '0'
        previous_handlers = logging.root.handlers
        c10d.init_process_group(backend='gloo', init_method='env://')
        current_handlers = logging.root.handlers
        self.assertEqual(len(previous_handlers), len(current_handlers))
        for (current, previous) in zip(current_handlers, previous_handlers):
            self.assertEqual(current, previous)
        c10d.destroy_process_group()

class TimeoutTest(test_c10d_common.AbstractTimeoutTest, TestCase):

    @requires_gloo()
    @retry_on_connect_failures
    def test_default_store_timeout_gloo(self):
        if False:
            return 10
        self._test_default_store_timeout('gloo')

class ProcessGroupGlooTest(MultiProcessTestCase):

    def _create_process_group_gloo(self, store, rank, world_size, opts):
        if False:
            print('Hello World!')
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, opts)
        dist.barrier(group=pg)
        return pg

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self._spawn_processes()

    def opts(self, threads=2):
        if False:
            for i in range(10):
                print('nop')
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = 50.0
        opts._devices = [create_device(interface=LOOPBACK)]
        opts._threads = threads
        return opts

    @requires_gloo()
    def test_multi_device_constructor(self):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = 5.0
        opts._devices = [create_device(interface=LOOPBACK), create_device(interface=LOOPBACK)]
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, opts)
        for fut in [pg.allreduce(torch.ones(i + 1)).get_future() for i in range(4)]:
            fut.wait()

    @requires_gloo()
    def test_empty_tensors(self):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        xs = [torch.FloatTensor([])]
        fut = pg.broadcast(xs).get_future()
        fut.wait()
        output = fut.value()
        self.assertEqual(0, output[0].numel())
        self.assertEqual(xs[0], output[0])

    @requires_gloo()
    def test_broadcast_checks(self):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, 'invalid root rank'):
            opts = c10d.BroadcastOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.broadcast([t1], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid root rank'):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.world_size
            opts.rootTensor = 0
            pg.broadcast([t1], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid root tensor'):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = -1
            pg.broadcast([t1], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid root tensor'):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 1
            pg.broadcast([t1], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid root tensor'):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid tensor type'):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([t1, t2], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid tensor size'):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([t1, t3], opts)

    def _test_broadcast_basics(self, fn):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())

        def broadcast(xs, rootRank, rootTensor):
            if False:
                for i in range(10):
                    print('nop')
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            fut = pg.broadcast(xs, opts).get_future()
            fut.wait()
            return fut.value()
        for i in range(self.world_size):
            x = fn(torch.tensor([self.rank]))
            output = broadcast([x], i, 0)
            self.assertEqual(torch.tensor([i]), output[0])
            num = 2
            for j in range(num):
                xs = [fn(torch.tensor([self.rank * num + 0.0])), fn(torch.tensor([self.rank * num + 1.0]))]
                output = broadcast(xs, i, j)
                self.assertEqual(torch.tensor([i * num + j], dtype=torch.float32), output[0])
                self.assertEqual(torch.tensor([i * num + j], dtype=torch.float32), output[1])
        x = torch.tensor([self.rank + 1.0])
        fut = pg.broadcast(x, root=0).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(torch.tensor([1.0]), result[0])

    @requires_gloo()
    def test_broadcast_basics(self):
        if False:
            return 10
        self._test_broadcast_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_broadcast_basics_cuda(self):
        if False:
            i = 10
            return i + 15
        self._test_broadcast_basics(lambda t: t.clone().cuda())

    def _test_broadcast_stress(self, inputs):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts(threads=8))
        work_handles = [pg.broadcast(inputs[i], root=i % self.world_size) for i in range(len(inputs))]
        for (i, work_handle) in enumerate(work_handles):
            work_handle.wait()
            self.assertEqual(torch.tensor([i * self.world_size + i % self.world_size]), inputs[i], msg='Mismatch in iteration %d' % i)

    @requires_gloo()
    def test_broadcast_stress(self):
        if False:
            while True:
                i = 10
        inputs = [torch.tensor([i * self.world_size + self.rank]) for i in range(1000)]
        self._test_broadcast_stress(inputs)

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_broadcast_stress_cuda(self):
        if False:
            for i in range(10):
                print('nop')
        inputs = [torch.tensor([i * self.world_size + self.rank]).cuda() for i in range(1000)]
        self._test_broadcast_stress(inputs)

    @requires_gloo()
    def test_allreduce_checks(self):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, 'requires non-empty tensor list'):
            opts = c10d.AllreduceOptions()
            pg.allreduce([], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid tensor type'):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t2], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid tensor size'):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t3], opts)

    def _test_allreduce_basics(self, fn):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        tests = simple_reduce_tests(self.rank, self.world_size)
        for (op, input, expected) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensor = fn(input)
            fut = pg.allreduce([tensor], opts).get_future()
            fut.wait()
            result = fut.value()
            self.assertEqual(expected, result[0])
        tests = simple_multi_input_reduce_tests(self.rank, self.world_size)
        for (op, inputs, output) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensors = [fn(input) for input in inputs]
            fut = pg.allreduce(tensors, opts).get_future()
            fut.wait()
            result = fut.value()
            for tensor in result:
                self.assertEqual(output, tensor)
        x = fn(torch.tensor([self.rank + 1.0]))
        fut = pg.allreduce(x).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(torch.tensor([float(self.world_size * (self.world_size + 1) / 2)]), result[0])

    @requires_gloo()
    def test_allreduce_basics(self):
        if False:
            while True:
                i = 10
        self._test_allreduce_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_allreduce_basics_cuda(self):
        if False:
            return 10
        self._test_allreduce_basics(lambda t: t.clone().cuda())

    def _test_allreduce_basics_using_work_api(self, fn):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        tests = simple_reduce_tests(self.rank, self.world_size)
        for (op, input, expected) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensor = fn(input)
            work = pg.allreduce([tensor], opts)
            work.wait()
            result = work.result()
            self.assertEqual(expected, result[0])
        tests = simple_multi_input_reduce_tests(self.rank, self.world_size)
        for (op, inputs, output) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensors = [fn(input) for input in inputs]
            work = pg.allreduce(tensors, opts)
            work.wait()
            result = work.result()
            for tensor in result:
                self.assertEqual(output, tensor)
        x = fn(torch.tensor([self.rank + 1.0]))
        work = pg.allreduce(x)
        work.wait()
        result = work.result()
        self.assertEqual(torch.tensor([float(self.world_size * (self.world_size + 1) / 2)]), result[0])

    @requires_gloo()
    def test_allreduce_basics_using_work_api(self):
        if False:
            return 10
        self._test_allreduce_basics_using_work_api(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_allreduce_basics_cuda_using_work_api(self):
        if False:
            i = 10
            return i + 15
        self._test_allreduce_basics_using_work_api(lambda t: t.clone().cuda())

    def _test_allreduce_stress(self, inputs):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts(threads=8))
        future_handles = [pg.allreduce(inputs[i]).get_future() for i in range(len(inputs))]
        for (i, future_handle) in enumerate(future_handles):
            future_handle.wait()
            self.assertEqual(torch.tensor([i * self.world_size + self.world_size * (self.world_size - 1) // 2]), future_handle.value()[0], msg='Mismatch in iteration %d' % i)

    @requires_gloo()
    def test_allreduce_stress(self):
        if False:
            i = 10
            return i + 15
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_allreduce_stress(inputs)

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_allreduce_stress_cuda(self):
        if False:
            for i in range(10):
                print('nop')
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_allreduce_stress(inputs)

    @requires_gloo()
    def test_allreduce_coalesced_checks(self):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        t1 = torch.zeros(1, dtype=torch.float32)
        t2 = torch.zeros(1, dtype=torch.float64)
        t3 = torch.sparse_coo_tensor([[0]], [1], size=(1,))
        with self.assertRaisesRegex(RuntimeError, 'requires non-empty tensor list'):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([], opts)
        with self.assertRaisesRegex(RuntimeError, 'tensors must all have the same type'):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1, t2], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid tensor layout at index'):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1, t3], opts)
        with self.assertRaisesRegex(RuntimeError, 'unsupported layout'):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t3, t3.clone()], opts)

    @skip_if_lt_x_gpu(1)
    @requires_gloo()
    def test_allreduce_coalesced_checks_cuda(self):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        t1 = torch.zeros(1, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, 'unsupported device type'):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1.cuda(), t1.cuda()], opts)

    def _test_allreduce_coalesced_basics(self, fn):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        test_cases = simple_coalesced_reduce_tests(self.rank, self.world_size)
        for (op, inputs, outputs) in test_cases:
            opts = c10d.AllreduceCoalescedOptions()
            opts.reduceOp = op
            tensors = [fn(x) for x in inputs]
            fut = pg.allreduce_coalesced(tensors, opts).get_future()
            fut.wait()
            result = fut.value()
            for (result_tensor, expected) in zip(result, outputs):
                self.assertEqual(result_tensor, expected)

    @requires_gloo()
    def test_allreduce_coalesced_basics(self):
        if False:
            print('Hello World!')
        self._test_allreduce_coalesced_basics(lambda t: t.clone())

    def _expected_output(self, i):
        if False:
            for i in range(10):
                print('nop')
        ws = self.world_size
        return 2 * [torch.tensor([i * ws + ws * (ws - 1) // 2])]

    def _test_allreduce_coalesced_stress(self, inputs):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts(threads=8))
        future_handles = [pg.allreduce_coalesced(input).get_future() for input in inputs]
        for (i, future_handle) in enumerate(future_handles):
            future_handle.wait()
            result = future_handle.value()
            self.assertEqual(self._expected_output(i), result, msg=f'Mismatch in iteration {i}')

    @requires_gloo()
    def test_allreduce_coalesced_stress(self):
        if False:
            i = 10
            return i + 15
        inputs = [2 * [torch.tensor([i + self.rank])] for i in range(1000)]
        self._test_allreduce_coalesced_stress(inputs)

    @requires_gloo()
    def test_allreduce_coalesced_async(self):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(backend='gloo', rank=self.rank, world_size=self.world_size, store=store)
        xs = [2 * [torch.tensor([i + self.rank])] for i in range(2)]
        futs = [c10d.all_reduce_coalesced(x, async_op=True) for x in xs]
        torch.futures.wait_all(futs)
        for (i, fut) in enumerate(futs):
            self.assertEqual(self._expected_output(i), fut.wait(), msg=f'Mismatch in iteration {i}')

    @requires_gloo()
    def test_sparse_allreduce_checks(self):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        t1 = torch.zeros([1])
        t2 = torch.sparse_coo_tensor([[0]], [1], size=(2,))
        t3 = torch.sparse_coo_tensor([[0]], [1], size=(4,))
        with self.assertRaisesRegex(RuntimeError, 'requires non-empty tensor list'):
            opts = c10d.AllreduceOptions()
            pg.allreduce([], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid tensor layout'):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t2], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid tensor size'):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t2, t3], opts)
        for op in [c10d.ReduceOp.PRODUCT, c10d.ReduceOp.MIN, c10d.ReduceOp.MAX]:
            with self.assertRaisesRegex(RuntimeError, 'unsupported reduction operation'):
                opts = c10d.AllreduceOptions()
                opts.reduceOp = op
                pg.allreduce([t3], opts)

    def _test_sparse_allreduce_basics(self, fn):
        if False:
            for i in range(10):
                print('nop')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        for num_inputs_per_rank in [1, 2]:
            tests = simple_sparse_reduce_tests(self.rank, self.world_size, num_inputs=num_inputs_per_rank)
            for (inputs, outputs) in tests:
                tensors = [fn(input) for input in inputs]
                fut = pg.allreduce(tensors).get_future()
                fut.wait()
                result = fut.value()
                self.assertEqual(tensors, outputs)
                self.assertEqual(result, outputs)

    @requires_gloo()
    def test_sparse_allreduce_basics(self):
        if False:
            while True:
                i = 10
        self._test_sparse_allreduce_basics(lambda t: t)

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_sparse_allreduce_basics_cuda(self):
        if False:
            print('Hello World!')
        self._test_sparse_allreduce_basics(lambda t: t.clone().cuda())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_sparse_allreduce_cuda_dispatched(self):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(backend='gloo', store=store, rank=self.rank, world_size=self.world_size)
        tests = simple_sparse_reduce_tests(self.rank, self.world_size, num_inputs=1)
        for (inputs, outputs) in tests:
            tensors = inputs[-1].clone().cuda()
            work = dist.all_reduce(tensors, async_op=True)
            work.wait()
            self.assertEqual([tensors], outputs)

    @requires_gloo()
    def test_scatter_checks(self):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, 'invalid root rank'):
            opts = c10d.ScatterOptions()
            opts.rootRank = -1
            pg.scatter([t1], [], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid root rank'):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.world_size
            pg.scatter([t1], [], opts)
        with self.assertRaisesRegex(RuntimeError, 'requires a single-element output tensor list'):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([], [], opts)
        with self.assertRaisesRegex(RuntimeError, 'requires a single-element output tensor list'):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([t1, t1], [], opts)
        with self.assertRaisesRegex(RuntimeError, 'requires a single-element input list'):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [], opts)
        with self.assertRaisesRegex(RuntimeError, 'requires a single-element input list'):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * self.world_size, [t1] * self.world_size], opts)
        desired_list_size = self.world_size
        incorrect_list_size = self.world_size - 1
        err_str = 'Incorrect input list size {}. Input list size should be {}'
        with self.assertRaisesRegex(RuntimeError, err_str.format(incorrect_list_size, desired_list_size)):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * incorrect_list_size], opts)
        incorrect_list_size = self.world_size + 1
        with self.assertRaisesRegex(RuntimeError, err_str.format(incorrect_list_size, desired_list_size)):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * incorrect_list_size], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid tensor type'):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t2] * self.world_size], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid tensor size'):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t3] * self.world_size], opts)
        with self.assertRaisesRegex(RuntimeError, 'requires empty input on non-root'):
            opts = c10d.ScatterOptions()
            opts.rootRank = (self.rank + 1) % self.world_size
            pg.scatter([t1], [[t1] * self.world_size], opts)

    def _test_scatter_basics(self, fn):
        if False:
            for i in range(10):
                print('nop')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        input = [fn(torch.tensor([self.rank])) for _ in range(self.world_size)]
        outputs = [fn(torch.tensor([-1])) for _ in range(self.world_size)]
        futures = []
        for i in range(self.world_size):
            opts = c10d.ScatterOptions()
            opts.rootRank = i
            if i == self.rank:
                futures.append(pg.scatter([outputs[i]], [input], opts).get_future())
            else:
                futures.append(pg.scatter([outputs[i]], [], opts).get_future())
        for i in range(self.world_size):
            futures[i].wait()
            result = futures[i].value()
            self.assertEqual(torch.tensor([i]), result[0])

    @requires_gloo()
    def test_scatter_basics(self):
        if False:
            return 10
        self._test_scatter_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_scatter_basics_cuda(self):
        if False:
            while True:
                i = 10
        self._test_scatter_basics(lambda t: t.clone().cuda())

    def _test_scatter_stress(self, inputs, fn):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts(threads=8))
        outputs = [[fn(torch.tensor([-1])) for _ in range(self.world_size)] for _ in range(len(inputs))]
        future_handles = []
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.ScatterOptions()
                opts.rootRank = root
                if root == self.rank:
                    fut = pg.scatter([outputs[i][root]], [[fn(e) for e in inputs[i]]], opts).get_future()
                else:
                    fut = pg.scatter([outputs[i][root]], [], opts).get_future()
                future_handles.append(fut)
        for (i, future_handle) in enumerate(future_handles):
            future_handle.wait()
            iter = i // self.world_size
            root = i % self.world_size
            result = future_handle.value()
            self.assertEqual(torch.tensor([iter + root]), result[0], msg='Mismatch in iteration %d for rank %d' % (iter, root))

    @requires_gloo()
    def test_scatter_stress(self):
        if False:
            for i in range(10):
                print('nop')
        inputs = [[torch.tensor([i + self.rank]) for _ in range(self.world_size)] for i in range(1000)]
        self._test_scatter_stress(inputs, lambda t: t.clone())

    @skip_but_pass_in_sandcastle('Test is flaky, see https://github.com/pytorch/pytorch/issues/15963')
    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_scatter_stress_cuda(self):
        if False:
            for i in range(10):
                print('nop')
        inputs = [[torch.tensor([i + self.rank]) for _ in range(self.world_size)] for i in range(1000)]
        self._test_scatter_stress(inputs, lambda t: t.clone().cuda())

    @requires_gloo()
    def test_gather_checks(self):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, 'invalid root rank'):
            opts = c10d.GatherOptions()
            opts.rootRank = -1
            pg.gather([], [t1], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid root rank'):
            opts = c10d.GatherOptions()
            opts.rootRank = self.world_size
            pg.gather([], [t1], opts)
        with self.assertRaisesRegex(RuntimeError, 'requires a single-element input tensor list'):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([], [], opts)
        with self.assertRaisesRegex(RuntimeError, 'requires a single-element input tensor list'):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([], [t1, t1], opts)
        with self.assertRaisesRegex(RuntimeError, 'requires a single-element output list'):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([], [t1], opts)
        with self.assertRaisesRegex(RuntimeError, 'requires a single-element output list'):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * self.world_size, [t1] * self.world_size], [t1], opts)
        desired_list_size = self.world_size
        incorrect_list_size = self.world_size - 1
        err_str = 'Incorrect output list size {}. Output list size should be {}'
        with self.assertRaisesRegex(RuntimeError, err_str.format(incorrect_list_size, desired_list_size)):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * incorrect_list_size], [t1], opts)
        incorrect_list_size = self.world_size + 1
        with self.assertRaisesRegex(RuntimeError, err_str.format(incorrect_list_size, desired_list_size)):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * incorrect_list_size], [t1], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid tensor type'):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t2] * self.world_size], [t1], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid tensor size'):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t3] * self.world_size], [t1], opts)
        with self.assertRaisesRegex(RuntimeError, 'requires empty output on non-root'):
            opts = c10d.GatherOptions()
            opts.rootRank = (self.rank + 1) % self.world_size
            pg.gather([[t1] * self.world_size], [t1], opts)

    def _test_gather_basics(self, fn):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        input = [fn(torch.tensor([self.rank]))]
        outputs = [fn(torch.tensor([-1])) for _ in range(self.world_size)]
        futures = []
        for i in range(self.world_size):
            opts = c10d.GatherOptions()
            opts.rootRank = i
            if i == self.rank:
                futures.append(pg.gather([outputs], input, opts).get_future())
            else:
                futures.append(pg.gather([], input, opts).get_future())
        expected = [fn(torch.tensor([rank])) for rank in range(self.world_size)]
        for i in range(self.world_size):
            futures[i].wait()
            result = futures[i].value()
            if i == self.rank:
                self.assertEqual(expected, result)

    @requires_gloo()
    def test_gather_basics(self):
        if False:
            print('Hello World!')
        self._test_gather_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_gather_basics_cuda(self):
        if False:
            i = 10
            return i + 15
        self._test_gather_basics(lambda t: t.clone().cuda())

    @requires_gloo()
    def test_gather_noncontiguous_input(self):
        if False:
            i = 10
            return i + 15
        self._test_gather_basics(lambda t: t.expand(2, 2).contiguous()[:, 0])

    def _test_gather_stress(self, inputs, fn):
        if False:
            for i in range(10):
                print('nop')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts(threads=8))
        future_handles = []
        outputs = [[[fn(torch.tensor([-1])) for _ in range(self.world_size)]] for _ in range(len(inputs))]
        expected_outputs = [[[torch.tensor([i + j]) for j in range(self.world_size)]] for i in range(len(inputs))]
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.GatherOptions()
                opts.rootRank = root
                if root == self.rank:
                    fut = pg.gather(outputs[i], [fn(inputs[i])], opts).get_future()
                else:
                    fut = pg.gather([], [fn(inputs[i])], opts).get_future()
                future_handles.append(fut)
        for (i, future_handle) in enumerate(future_handles):
            future_handle.wait()
            iter = i // self.world_size
            root = i % self.world_size
            if root == self.rank:
                result = future_handle.value()
                self.assertEqual(expected_outputs[iter], [result], msg='Mismatch in iteration %d for root %d' % (iter, root))

    @requires_gloo()
    def test_gather_stress(self):
        if False:
            print('Hello World!')
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_gather_stress(inputs, lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_gather_stress_cuda(self):
        if False:
            while True:
                i = 10
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_gather_stress(inputs, lambda t: t.clone().cuda())

    @requires_gloo()
    def test_allgather_checks(self):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, 'requires non-empty input tensor list'):
            pg.allgather([], [])
        with self.assertRaisesRegex(RuntimeError, 'requires input/output tensor lists to have the same length'):
            pg.allgather([], [t1])
        with self.assertRaisesRegex(RuntimeError, 'requires input/output tensor lists to have the same length'):
            pg.allgather([[t1] * self.world_size, [t1] * self.world_size], [t1])
        with self.assertRaisesRegex(RuntimeError, 'invalid output tensor list'):
            pg.allgather([[t1] * (self.world_size - 1)], [t1])
        with self.assertRaisesRegex(RuntimeError, 'invalid output tensor list'):
            pg.allgather([[t1] * (self.world_size + 1)], [t1])
        with self.assertRaisesRegex(RuntimeError, 'invalid tensor type'):
            pg.allgather([[t1, t1] * self.world_size, [t1, t1] * self.world_size], [t1, t2])
        with self.assertRaisesRegex(RuntimeError, 'invalid tensor size'):
            pg.allgather([[t1, t1] * self.world_size, [t1, t1] * self.world_size], [t1, t3])
        with self.assertRaisesRegex(RuntimeError, 'invalid tensor type'):
            pg.allgather([([t1, t2] * self.world_size)[:self.world_size]], [t1])
        with self.assertRaisesRegex(RuntimeError, 'invalid tensor size'):
            pg.allgather([([t1, t3] * self.world_size)[:self.world_size]], [t1])

    def _test_allgather_basics(self, fn):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        for n in [1, 2, 3]:
            input = [fn(torch.tensor([n * self.rank + i])) for i in range(n)]
            output = [[fn(torch.tensor([-1])) for _ in range(n * self.world_size)] for _ in range(n)]
            expected_output = [[fn(torch.tensor([i])) for i in range(n * self.world_size)] for _ in range(n)]
            fut = pg.allgather(output, input).get_future()
            fut.wait()
            result = fut.value()
            if n == 1:
                result = [result]
            self.assertEqual(expected_output, result)

    @requires_gloo()
    def test_allgather_basics(self):
        if False:
            i = 10
            return i + 15
        self._test_allgather_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_allgather_basics_cuda(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_allgather_basics(lambda t: t.clone().cuda())

    @requires_gloo()
    def test_allgather_noncontiguous_input(self):
        if False:
            while True:
                i = 10
        self._test_allgather_basics(lambda t: t.expand(2, 2).contiguous()[:, 0])

    def _test_allgather_stress(self, inputs, fn):
        if False:
            for i in range(10):
                print('nop')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts(threads=8))
        future_handles = []
        outputs = [[[fn(torch.tensor([-1])) for _ in range(self.world_size)]] for _ in range(len(inputs))]
        expected_outputs = [[[torch.tensor([i + j]) for j in range(self.world_size)]] for i in range(len(inputs))]
        input_holder = {}
        for i in range(len(inputs)):
            input_holder[i] = [fn(inputs[i])]
            fut = pg.allgather(outputs[i], input_holder[i]).get_future()
            future_handles.append(fut)
        for (i, future_handle) in enumerate(future_handles):
            future_handle.wait()
            result = future_handle.value()
            self.assertEqual(expected_outputs[i], [result], msg='Mismatch in iteration %d' % i)

    @requires_gloo()
    def test_allgather_stress(self):
        if False:
            print('Hello World!')
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_allgather_stress(inputs, lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_allgather_stress_cuda(self):
        if False:
            print('Hello World!')
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_allgather_stress(inputs, lambda t: t.clone().cuda())

    @requires_gloo()
    def test_allgather_coalesced_checks(self):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        dummy_input = [torch.zeros([1], dtype=torch.float32)]
        dummy_output_lists = [[torch.zeros([1], dtype=torch.float32)] for _ in range(self.world_size)]
        dummy_output_lists[0] = [torch.zeros([0], dtype=torch.float32)]
        with self.assertRaisesRegex(RuntimeError, 'invalid size of output tensor at index 0'):
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)
        dummy_output_lists[0] = [torch.zeros([1], dtype=torch.float64)]
        with self.assertRaisesRegex(RuntimeError, 'invalid tensor type at index 0'):
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)
        dummy_output_lists = [[torch.zeros([1], dtype=torch.float32)] for _ in range(self.world_size + 1)]
        with self.assertRaisesRegex(RuntimeError, 'output lists should be equal to world size'):
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)
        dummy_output_lists = [torch.zeros([0], dtype=torch.float32)]
        with self.assertRaisesRegex(TypeError, 'Invalid function argument.*output_tensor_lists'):
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

    @requires_gloo()
    def test_allgather_coalesced_async(self):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(backend='gloo', rank=self.rank, world_size=self.world_size, store=store)
        xxs = [2 * [torch.tensor([i + self.rank])] for i in range(2)]
        yys = [[[torch.zeros_like(x) for x in xx] for _ in range(self.world_size)] for xx in xxs]
        futs = [c10d.all_gather_coalesced(yy, xx, async_op=True) for (xx, yy) in zip(xxs, yys)]
        zzs = [[2 * [torch.tensor([i + r])] for r in range(self.world_size)] for i in range(2)]
        torch.futures.wait_all(futs)
        for (yy, zz) in zip(yys, zzs):
            for (y_out, z_out) in zip(yy, zz):
                for (y, z) in zip(y_out, z_out):
                    self.assertEqual(y, z)
        c10d.destroy_process_group()

    @requires_gloo()
    def test_reduce_checks(self):
        if False:
            i = 10
            return i + 15
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        t1 = torch.zeros([1], dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, 'invalid root rank'):
            opts = c10d.ReduceOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.reduce([t1], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid root rank'):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.world_size
            opts.rootTensor = 0
            pg.reduce([t1], opts)
        with self.assertRaisesRegex(RuntimeError, 'invalid root tensor'):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 1
            pg.reduce([t1], opts)
        with self.assertRaisesRegex(RuntimeError, 'requires a single-element tensor list'):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.reduce([t1, t1], opts)

    def _test_reduce_basics(self, fn):
        if False:
            i = 10
            return i + 15
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        for (op, input, output) in simple_reduce_tests(self.rank, self.world_size):
            for root in range(self.world_size):
                opts = c10d.ReduceOptions()
                opts.reduceOp = op
                opts.rootRank = root
                tmp = fn(input)
                fut = pg.reduce([tmp], opts).get_future()
                fut.wait()
                result = fut.value()
                if root == self.rank:
                    self.assertEqual(output, result[0])

    @requires_gloo()
    def test_reduce_basics(self):
        if False:
            i = 10
            return i + 15
        self._test_reduce_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_reduce_basics_cuda(self):
        if False:
            while True:
                i = 10
        self._test_reduce_basics(lambda t: t.clone().cuda())

    def _test_reduce_stress(self, inputs):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts(threads=8))
        future_handles = []
        outputs = []
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.ReduceOptions()
                opts.rootRank = root
                tmp = inputs[i].clone()
                outputs.append(tmp)
                fut = pg.reduce([tmp], opts).get_future()
                future_handles.append(fut)
        for (i, future_handle) in enumerate(future_handles):
            future_handle.wait()
            result = future_handle.value()
            iter = i // self.world_size
            root = i % self.world_size
            if root == self.rank:
                self.assertEqual(torch.tensor([iter * self.world_size + self.world_size * (self.world_size - 1) // 2]), result[0], msg='Mismatch in iteration %d with root rank %d' % (iter, root))

    @requires_gloo()
    def test_reduce_stress(self):
        if False:
            print('Hello World!')
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_reduce_stress(inputs)

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_reduce_stress_cuda(self):
        if False:
            i = 10
            return i + 15
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_reduce_stress(inputs)

    @requires_gloo()
    def test_send_recv_all_to_all(self):
        if False:
            for i in range(10):
                print('nop')
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        inputs = [torch.tensor([self.rank]) for _ in range(self.world_size)]
        outputs = [torch.tensor([-1]) for _ in range(self.world_size)]
        send_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            send_work.append(pg.send([inputs[i]], i, 0))
        recv_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            recv_work.append(pg.recv([outputs[i]], i, 0))
        for work in send_work:
            work.wait()
            self.assertTrue(work.is_completed())
        for work in recv_work:
            work.wait()
            self.assertTrue(work.is_completed())
        for i in range(self.world_size):
            if i == self.rank:
                continue
            self.assertEqual(torch.tensor([i]), outputs[i])

    @requires_gloo()
    def test_barrier_implies_wait(self):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, self.opts())
        size = (100, 100)
        num = 16
        tensors = [torch.full(size, float(i)) for i in range(num)]
        for tensor in tensors:
            pg.allreduce(tensor)
        pg.barrier().get_future().wait()
        for (i, tensor) in enumerate(tensors):
            self.assertEqual(torch.full(size, float(i * self.world_size)), tensor)

    @skip_if_win32()
    @requires_gloo()
    def test_round_robin(self):
        if False:
            print('Hello World!')
        num_process_groups = 2
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(backend='gloo', store=store, rank=self.rank, world_size=self.world_size)
        pg = c10d._round_robin_process_groups([c10d.new_group(pg_options=self.opts()) for i in range(num_process_groups)])
        for _ in range(num_process_groups + 1):
            tensor = torch.full([100, 100], float(self.rank))
            pg.broadcast(tensor, root=0).wait()
            self.assertEqual(torch.full([100, 100], 0.0), tensor)

    @skip_if_win32()
    @requires_gloo()
    def test_round_robin_create_destroy(self):
        if False:
            for i in range(10):
                print('nop')
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(backend='gloo', store=store, rank=self.rank, world_size=self.world_size)

        def create(num, prefix):
            if False:
                return 10
            return c10d._round_robin_process_groups([c10d.new_group(pg_options=self.opts()) for i in range(num)])
        for i in range(2):
            num_process_groups = 2
            pg = create(num=num_process_groups, prefix=i)
            for _ in range(3):
                tensor = torch.ones([10, 10])
                pg.allreduce(tensor).wait()
                self.assertEqual(torch.full([10, 10], float(self.world_size)), tensor)
            del pg

class DistributedDataParallelTest(test_c10d_common.CommonDistributedDataParallelTest, MultiProcessTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self._spawn_processes()

    def _get_process_group(self):
        if False:
            return 10
        store = self._get_store()
        c10d.init_process_group(backend='gloo', store=store, rank=self.rank, world_size=self.world_size)
        return c10d.distributed_c10d._get_default_group()

    def _test_gloo_backend(self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False):
        if False:
            return 10
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(backend='gloo', store=store, rank=self.rank, world_size=self.world_size)
        process_group = c10d.distributed_c10d._get_default_group()
        device = devices[-1]
        backend = process_group._get_backend(device)
        backend.create_device(interface=LOOPBACK)
        self._test_ddp_with_process_group(process_group, devices, device_ids, multi_device, gradient_as_bucket_view)

    @requires_gloo()
    def test_gloo_backend_cpu_module(self):
        if False:
            i = 10
            return i + 15
        self._test_gloo_backend([torch.device('cpu')], None)

    @requires_gloo()
    def test_gloo_backend_cpu_module_grad_is_view(self):
        if False:
            i = 10
            return i + 15
        self._test_gloo_backend([torch.device('cpu')], None, gradient_as_bucket_view=True)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_gloo_backend_1gpu_module_device_ids_integer_list(self):
        if False:
            for i in range(10):
                print('nop')
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device('cuda:' + str(i)) for i in int_devices]
        self._test_gloo_backend(devices, int_devices)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_gloo_backend_1gpu_module_device_ids_torch_device_list(self):
        if False:
            return 10
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device('cuda:' + str(i)) for i in int_devices]
        self._test_gloo_backend(devices, devices)

    @requires_gloo()
    @skip_if_lt_x_gpu(4)
    def test_gloo_backend_2gpu_module(self):
        if False:
            for i in range(10):
                print('nop')
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device('cuda:' + str(i)) for i in int_devices]
        self._test_gloo_backend(devices, None, multi_device=True)

    @requires_gloo()
    @skip_if_lt_x_gpu(8)
    def test_gloo_backend_4gpu_module(self):
        if False:
            i = 10
            return i + 15
        int_devices = gpus_for_rank(self.world_size)[self.rank][:4]
        devices = [torch.device('cuda:' + str(i)) for i in int_devices]
        self._test_gloo_backend(devices, None, multi_device=True)

    def _test_global_local_unused_params_grad(self, gradient_as_bucket_view=False, static_graph=False):
        if False:
            i = 10
            return i + 15
        '\n        By simulating a multi-task training, this test is to make sure:\n        1) DDP does not touch the grad of globally unused parameters.\n        2) DDP does update the grad of locally unused parameters.\n        '

        class GlobalLocalUnusedParamModule(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.t0 = Task()
                self.t1 = Task()
                self.task_unused = Task()

            def task_parameters(self):
                if False:
                    while True:
                        i = 10
                return (self.t0.p, self.t1.p, self.task_unused.p)

            def forward(self, x, rank):
                if False:
                    print('Hello World!')
                return self.t0(x) if rank == 0 else self.t1(x)

        def run_and_verify_grad(model):
            if False:
                print('Hello World!')
            output = model(8, self.rank)
            (t0_p, t1_p, task_unused_p) = model.module.task_parameters()
            self.assertIsNone(t0_p.grad)
            self.assertIsNone(t1_p.grad)
            self.assertIsNone(task_unused_p.grad)
            output.mean().backward()
            self.assertIsNotNone(t0_p.grad)
            self.assertIsNotNone(t1_p.grad)
            self.assertIsNone(task_unused_p.grad)
        process_group = self._get_process_group()
        cpu_model = DistributedDataParallel(GlobalLocalUnusedParamModule().cpu(), process_group=process_group, find_unused_parameters=True, gradient_as_bucket_view=gradient_as_bucket_view, static_graph=static_graph)
        run_and_verify_grad(cpu_model)
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(GlobalLocalUnusedParamModule().to(device_id), device_ids=[device_id], process_group=process_group, find_unused_parameters=True, gradient_as_bucket_view=gradient_as_bucket_view, static_graph=static_graph)
        run_and_verify_grad(gpu_model)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad(self):
        if False:
            return 10
        self._test_global_local_unused_params_grad()

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad_with_grad_is_view(self):
        if False:
            i = 10
            return i + 15
        self._test_global_local_unused_params_grad(gradient_as_bucket_view=True)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad_with_static_graph(self):
        if False:
            return 10
        self._test_global_local_unused_params_grad(static_graph=True)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_find_unused_parameters_when_unused_parameters_empty(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        An empty unused_parameters array does not imply find_unused_parameters =\n        false. This test makes sure that DDP allreduces unused parameters\n        accordingly where the forward pass in some process uses all parameters.\n        This unit test creates a module that uses all parameters in rank = 0, and\n        has unused parameters in other ranks.\n        '

        class FindUnusedParamModule(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.t0 = Task()
                self.t1 = Task()

            def task_parameters(self):
                if False:
                    while True:
                        i = 10
                return (self.t0.p, self.t1.p)

            def forward(self, x, rank):
                if False:
                    while True:
                        i = 10
                return self.t1(self.t0(x)) if rank == 0 else self.t1(x)

        def run_and_verify_grad(model):
            if False:
                return 10
            output = model(8, self.rank)
            [self.assertIsNone(t_p.grad) for t_p in model.module.task_parameters()]
            output.mean().backward()
            [self.assertIsNotNone(t_p.grad) for t_p in model.module.task_parameters()]
        process_group = self._get_process_group()
        cpu_model = DistributedDataParallel(FindUnusedParamModule().cpu(), process_group=process_group, find_unused_parameters=True)
        run_and_verify_grad(cpu_model)
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(FindUnusedParamModule().to(device_id), device_ids=[device_id], process_group=process_group, find_unused_parameters=True)
        run_and_verify_grad(gpu_model)

    @requires_gloo()
    def test_ignored_output(self):
        if False:
            return 10
        '\n        Test that the output of a model can be ignored and that there is no\n        implicit requirement that `backward` gets called.\n        '
        process_group = self._get_process_group()

        class IgnoredOutput(nn.Module):

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
                    i = 10
                    return i + 15
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)
        model = DistributedDataParallel(IgnoredOutput().float(), process_group=process_group)
        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        for _ in range(4):
            output = model(input)
            del output
        for _ in range(4):
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

    @requires_gloo()
    def test_ignored_output_with_unused_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that the output of a model can be ignored and that there is no\n        implicit requirement that `backward` gets called, if not all model\n        parameters participated in computing the model output.\n        '
        process_group = self._get_process_group()

        class IgnoredOutputWithUnusedParameters(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)
        model = DistributedDataParallel(IgnoredOutputWithUnusedParameters().float(), process_group=process_group, find_unused_parameters=True)
        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        for _ in range(4):
            output = model(input)
            del output
        for _ in range(4):
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_ignored_sharded_tensor(self):
        if False:
            i = 10
            return i + 15

        class MyModule(nn.Module):

            def __init__(self, shard_tensor: ShardedTensor) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.st = nn.Parameter(shard_tensor)
                self.relu = nn.ReLU()

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.relu(self.fc1(x))
                return F.softmax(x, dim=1)
        pg = dist.init_process_group('gloo', init_method=f'file://{self.file_name}', world_size=self.world_size, rank=self.rank)
        device = torch.device(f'cuda:{self.rank}')
        local_shard_metadata = ShardMetadata(shard_offsets=[self.rank % 2 * 5, 0], shard_sizes=[5, 10], placement=f'rank:{self.rank}/cuda:{self.rank}')
        local_shards = [Shard(torch.randn(5, 10, device=device), local_shard_metadata)]
        st = init_from_local_shards(local_shards, [10, 10])
        m = MyModule(st)
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(module=m, params_and_buffers_to_ignore={'st'})
        DistributedDataParallel(m, device_ids=[device] if device.type == 'gpu' else None, process_group=pg, gradient_as_bucket_view=True, broadcast_buffers=False, static_graph=True)

    def _run_and_verify_sparse_gradients(self, vanilla_model, ddp_model):
        if False:
            i = 10
            return i + 15
        mult = 2
        batch_size = mult * self.world_size
        criterion = nn.CrossEntropyLoss()
        input = torch.randint(0, 10, [batch_size, 2])
        target = torch.randint(0, 10, [batch_size])
        criterion(vanilla_model(input), target).backward()
        partial_input = input.split(mult)[self.rank]
        partial_target = target.split(mult)[self.rank]
        criterion(ddp_model(partial_input), partial_target).backward()
        vanilla_parameter = next(vanilla_model.parameters())
        ddp_parameter = next(ddp_model.parameters())
        self.assertEqual(vanilla_parameter.grad.coalesce(), ddp_parameter.grad.coalesce())

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_save_load_checkpoint(self):
        if False:
            print('Hello World!')
        dist.init_process_group('gloo', init_method=f'file://{self.file_name}', world_size=self.world_size, rank=self.rank)

        class TestModel(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        def train_loop(model, optimizer, iterations):
            if False:
                i = 10
                return i + 15
            for _ in range(iterations):
                optimizer.zero_grad()
                output = model(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model_withload = TestModel().float().to(device_id)
        model_withoutload = TestModel().float().to(device_id)
        ddp_withload = DistributedDataParallel(model_withload, device_ids=[device_id])
        ddp_withoutload = DistributedDataParallel(model_withoutload, device_ids=[device_id])
        for p in ddp_withload.parameters():
            with torch.no_grad():
                p.zero_()
        for p in model_withload.parameters():
            with torch.no_grad():
                p.zero_()
        for p in ddp_withoutload.parameters():
            with torch.no_grad():
                p.zero_()
        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        optimizer_withload = torch.optim.SGD(ddp_withload.parameters(), lr=0.001)
        optimizer_non_ddp_withload = torch.optim.SGD(model_withload.parameters(), lr=0.001)
        optimizer_withoutload = torch.optim.SGD(ddp_withoutload.parameters(), lr=0.001)
        input = torch.rand([batch_size, 2], dtype=torch.float).to(device_id)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(device_id)
        train_loop(ddp_withload, optimizer_withload, 3)
        checkpoint_path = tempfile.gettempdir() + '/model.checkpoint'
        if self.rank == 0:
            torch.save(ddp_withload.state_dict(), checkpoint_path)
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        ddp_state_dict = torch.load(checkpoint_path, map_location=map_location)
        for model in [ddp_withload, model_withload]:
            for p in ddp_withload.parameters():
                with torch.no_grad():
                    p.zero_()
        ddp_withload.load_state_dict(ddp_state_dict)
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(ddp_state_dict, 'module.')
        model_withload.load_state_dict(ddp_state_dict)
        train_loop(ddp_withload, optimizer_withload, 3)
        train_loop(model_withload, optimizer_non_ddp_withload, 3)
        train_loop(ddp_withoutload, optimizer_withoutload, 6)
        for (p_withload, p_withoutload, p_non_ddp_withload) in zip(ddp_withload.parameters(), ddp_withoutload.parameters(), model_withload.parameters()):
            self.assertEqual(p_withload, p_withoutload)
            self.assertEqual(p_non_ddp_withload, p_withoutload)

    def _test_sparse_gradients(self, gradient_as_bucket_view=False):
        if False:
            for i in range(10):
                print('nop')
        process_group = self._get_process_group()
        torch.manual_seed(1337)
        vanilla_model = SparseGradientModule()
        ddp_model = DistributedDataParallel(copy.deepcopy(vanilla_model), process_group=process_group, gradient_as_bucket_view=gradient_as_bucket_view)
        self._run_and_verify_sparse_gradients(vanilla_model, ddp_model)

    @requires_gloo()
    def test_sparse_gradients(self):
        if False:
            return 10
        self._test_sparse_gradients()

    @requires_gloo()
    def test_sparse_gradients_grad_is_view(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_sparse_gradients(gradient_as_bucket_view=True)

    @requires_gloo()
    def test_ddp_comm_hook_future_passing_cpu(self):
        if False:
            i = 10
            return i + 15
        '\n        This unit test verifies whether the Future object is passed properly.\n        The callback function creates a Future object and sets a value to it.\n        '
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = self._get_process_group()
        cpu_model = DistributedDataParallel(ModuleForDdpCommHook().cpu(), process_group=process_group)
        cpu_model.register_comm_hook(None, self._simple_hook)
        self._run_and_verify_hook(cpu_model, 8, 2 * torch.ones(2, 2))

    def _gpu_model_with_ddp_comm_hook(self, process_group, hook=None, gradient_as_bucket_view=False, state=None):
        if False:
            print('Hello World!')
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(ModuleForDdpCommHook().to(device_id), device_ids=[device_id], process_group=process_group, gradient_as_bucket_view=gradient_as_bucket_view)
        if hook is not None:
            gpu_model.register_comm_hook(state, hook)
        return gpu_model

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_future_passing_gpu_gloo(self):
        if False:
            return 10
        '\n        This unit test verifies whether the Future object is passed properly using gloo backend.\n        The hook callback function creates a Future object and sets a value to it.\n        '
        process_group = self._get_process_group()
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, self._simple_hook)
        self._run_and_verify_hook(gpu_model, 8, 2 * torch.ones(2, 2))

    @requires_gloo()
    def test_ddp_invalid_comm_hook_init(self):
        if False:
            while True:
                i = 10
        '\n        This unit test makes sure that register_comm_hook properly checks the format\n        of hook defined by user. The Python hook must be callable. This test also\n        checks whether bucket annotation checked properly if defined.\n        '
        process_group = self._get_process_group()
        model = DistributedDataParallel(ModuleForDdpCommHook(), process_group=process_group)
        with self.assertRaisesRegex(TypeError, 'Communication hook must be callable.'):
            model.register_comm_hook(state=None, hook=1)
        with self.assertRaisesRegex(ValueError, 'bucket annotation should be dist.GradBucket.'):

            def comm_hook(state: object, bucket: int) -> torch.futures.Future[torch.Tensor]:
                if False:
                    for i in range(10):
                        print('nop')
                return torch.futures.Future()
            model.register_comm_hook(state=None, hook=comm_hook)

    @requires_gloo()
    def test_ddp_invalid_comm_hook_return_type(self):
        if False:
            print('Hello World!')
        "\n        This test checks whether return annotation checked properly if defined. It also\n        checks whether an internal error is thrown if return type is incorrect and user\n        hasn't specified any return type annotation.\n        "
        process_group = self._get_process_group()
        model = DistributedDataParallel(ModuleForDdpCommHook(), process_group=process_group)
        expected_err = 'Communication hook: return annotation should be torch.futures.Future'
        with self.assertRaisesRegex(ValueError, expected_err):

            def comm_hook(state: object, bucket: dist.GradBucket) -> int:
                if False:
                    print('Hello World!')
                return torch.futures.Future()
            model.register_comm_hook(state=None, hook=comm_hook)
        verify_ddp_error_logged(model, expected_err)
        with self.assertRaisesRegex(RuntimeError, 'callback must return a torch.futures.Future object, but got'):

            def comm_hook(state: object, bucket: dist.GradBucket):
                if False:
                    print('Hello World!')
                return 1
            model.register_comm_hook(state=None, hook=comm_hook)
            output = model(8, self.rank)
            output.mean().backward()

    @requires_gloo()
    def test_ddp_comm_hook_register_just_once(self):
        if False:
            return 10
        '\n        DDP communication hook can only be registered once. This test validates whether\n        the error is thrown properly when register_comm_hook is called more than once.\n        '
        process_group = self._get_process_group()
        model = DistributedDataParallel(ModuleForDdpCommHook(), process_group=process_group)

        def dummy_hook(state, bucket):
            if False:
                print('Hello World!')
            fut = torch.futures.Future()
            fut.set_result([bucket.buffer()])
            return fut
        model.register_comm_hook(None, dummy_hook)
        with self.assertRaisesRegex(RuntimeError, 'register_comm_hook or register_builtin_comm_hook can only be called once.'):
            model.register_comm_hook(None, dummy_hook)

    @requires_gloo()
    def test_ddp_comm_hook_sparse_gradients(self):
        if False:
            while True:
                i = 10
        '\n        Runs "test_sparse_gradients" unit test with DDP communication hook. We define a\n        simple hook that does allreduce and works with gloo backend for this test.\n        '
        process_group = self._get_process_group()
        torch.manual_seed(1337)
        vanilla_model = SparseGradientModule()
        ddp_model = DistributedDataParallel(copy.deepcopy(vanilla_model), process_group=process_group)

        def allreduce_hook_gloo(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
            if False:
                print('Hello World!')

            def div_by_world_size(fut):
                if False:
                    while True:
                        i = 10
                return fut.wait()[0] / self.world_size
            fut = process_group.allreduce([bucket.buffer()]).get_future()
            return fut.then(div_by_world_size)
        ddp_model.register_comm_hook(None, allreduce_hook_gloo)
        self._run_and_verify_sparse_gradients(vanilla_model, ddp_model)

class ReducerModule(nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 4, bias=False)
        self.fc3 = nn.Linear(4, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, use_fc3=True):
        if False:
            return 10
        x = self.relu(self.fc1(x)).float()
        x = self.relu(self.fc2(x)).float()
        if use_fc3:
            x = self.fc3(x).float()
        return F.softmax(x, dim=1)

class ReducerTest(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.file = tempfile.NamedTemporaryFile(delete=False)
        world_size = 1
        self.store = c10d.FileStore(self.file.name, world_size)
        c10d.init_process_group(backend='gloo', store=self.store, rank=0, world_size=world_size)
        self.process_group = c10d.distributed_c10d._get_default_group()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        c10d.destroy_process_group()
        try:
            os.remove(self.file.name)
        except OSError as e:
            print(str(e))
            pass

    @requires_gloo()
    def test_single_dtype_single_bucket(self):
        if False:
            while True:
                i = 10
        model = ReducerModule()
        parameters = list(model.parameters())
        buckets = [list(range(len(parameters)))]
        dist.Reducer(parameters, buckets, [dist._DEFAULT_FIRST_BUCKET_BYTES], self.process_group)

    def _create_mixed_precision_model(self):
        if False:
            for i in range(10):
                print('nop')
        model = ReducerModule()
        model.float()
        model.fc1.double()
        return model

    @requires_gloo()
    def test_multi_dtype_single_bucket(self):
        if False:
            for i in range(10):
                print('nop')
        model = self._create_mixed_precision_model()
        with self.assertRaises(RuntimeError):
            parameters = list(model.parameters())
            buckets = [list(range(len(parameters)))]
            dist.Reducer(parameters, buckets, [dist._DEFAULT_FIRST_BUCKET_BYTES], self.process_group)

    @requires_gloo()
    def test_multi_dtype_multi_bucket(self):
        if False:
            while True:
                i = 10
        model = self._create_mixed_precision_model()
        parameters = list(model.parameters())
        group_by_dtype = groupby(range(len(parameters)), key=lambda i: parameters[i].dtype)
        buckets = [list(indices) for (_, indices) in group_by_dtype]
        dist.Reducer(parameters, buckets, [dist._DEFAULT_FIRST_BUCKET_BYTES for _ in buckets], self.process_group)

    def _create_reducer_for_models(self, models, find_unused_parameters=False):
        if False:
            print('Hello World!')
        self.assertEqual(len(models), 1)
        parameters = list(models[0].parameters())
        group_by_dtype = groupby(range(len(parameters)), key=lambda i: parameters[i].dtype)
        buckets = [list(indices) for (_, indices) in group_by_dtype]
        return dist.Reducer(parameters, buckets, [dist._DEFAULT_FIRST_BUCKET_BYTES for _ in range(len(buckets))], self.process_group, find_unused_parameters=find_unused_parameters)

    @requires_gloo()
    def test_forward_backward(self):
        if False:
            print('Hello World!')
        batch_size = 10
        model = self._create_mixed_precision_model()
        reducer = self._create_reducer_for_models([model])
        reducer.prepare_for_forward()
        loss = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.double)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        output = loss(model(input), target)
        reducer.prepare_for_backward(output)
        output.backward()

    @requires_gloo()
    def test_forward_backward_unused_parameters(self):
        if False:
            while True:
                i = 10
        batch_size = 10
        model = self._create_mixed_precision_model()
        reducer = self._create_reducer_for_models([model], find_unused_parameters=True)
        reducer.prepare_for_forward()
        loss = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.double)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        output = loss(model(input, use_fc3=False), target)
        self.assertEqual(None, model.fc3.weight.grad)
        reducer.prepare_for_backward(output)
        output.backward()
        self.assertEqual(None, model.fc3.weight.grad)

    @requires_gloo()
    def test_forward_backward_optimizer(self):
        if False:
            return 10
        batch_size = 10
        model = self._create_mixed_precision_model()
        reducer = self._create_reducer_for_models([model], find_unused_parameters=True)
        reducer.prepare_for_forward()
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        for i in range(3):
            input = torch.rand([batch_size, 2], dtype=torch.double)
            target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
            optimizer.zero_grad()
            output = loss(model(input, use_fc3=i > 0), target)
            reducer.prepare_for_backward(output)
            output.backward()
            optimizer.step()

class CommTest(test_c10d_common.AbstractCommTest, MultiProcessTestCase):

    @property
    def device(self):
        if False:
            return 10
        return 'cpu'

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _test_broadcast_coalesced(self, process_group, device, root_rank):
        if False:
            i = 10
            return i + 15
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

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_broadcast_coalesced_gloo_cuda(self):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(backend='gloo', store=store, rank=self.rank, world_size=self.world_size)
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device('cuda:%d' % self.rank)
        backend = process_group._get_backend(device)
        backend.create_device(interface=LOOPBACK)
        ranks = list(range(self.world_size))
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_gloo()
    def test_broadcast_coalesced_gloo_cpu(self):
        if False:
            i = 10
            return i + 15
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(backend='gloo', store=store, rank=self.rank, world_size=self.world_size)
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device('cpu')
        backend = process_group._get_backend(device)
        backend.create_device(interface=LOOPBACK)
        ranks = list(range(self.world_size))
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_default_pg_gloo(self):
        if False:
            i = 10
            return i + 15
        self._test_sequence_num_set_default_pg(backend='gloo')

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_gloo_new_group(self):
        if False:
            while True:
                i = 10
        self._test_sequence_num_set_new_group(backend='gloo')

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_sequence_num_incremented_gloo_default(self):
        if False:
            while True:
                i = 10
        self._test_sequence_num_incremented_default_group('gloo')

    @skip_if_lt_x_gpu(4)
    @requires_gloo()
    def test_sequence_num_incremented_gloo_subgroup(self):
        if False:
            return 10
        if self.world_size < 4:
            return skip_but_pass_in_sandcastle('Test requires world_size of at least 4')
        self._test_sequence_num_incremented_subgroup('gloo')

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_gloo_warn_not_in_group(self):
        if False:
            return 10
        self._test_warn_not_in_group(backend='gloo')

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_gloo_rank_membership(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_rank_membership(backend='gloo')

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_tensor_dtype_mismatch(self):
        if False:
            return 10
        self._test_tensor_dtype_mismatch(backend='gloo')

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_tensor_dtype_complex(self):
        if False:
            while True:
                i = 10
        self._test_tensor_dtype_complex(backend='gloo')

    @requires_gloo()
    def test_bool_tensors(self):
        if False:
            return 10
        self._test_bool_tensors(backend='gloo')

class GlooProcessGroupWithDispatchedCollectivesTests(test_c10d_common.ProcessGroupWithDispatchedCollectivesTests):

    @requires_gloo()
    def test_collectives(self):
        if False:
            i = 10
            return i + 15
        self._test_collectives(backend='gloo')

    @requires_gloo()
    def test_allreduce_coalesced(self):
        if False:
            return 10
        self._test_allreduce_coalesced(backend='gloo')

    @requires_gloo()
    def test_all_to_all_single(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_all_to_all_single(backend='gloo')

    @requires_gloo()
    def test_allgather_coalesced(self):
        if False:
            for i in range(10):
                print('nop')
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group('gloo', world_size=self.world_size, rank=self.rank, store=store)
        input_tensor = torch.ones(10, 10, dtype=torch.float32)
        output_tensor_list = [torch.zeros_like(input_tensor)]
        dist.all_gather_coalesced([output_tensor_list], [input_tensor])
        self.assertEqual(output_tensor_list, [input_tensor])

    @requires_gloo()
    def test_monitored_barrier(self):
        if False:
            print('Hello World!')
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group('gloo', world_size=self.world_size, rank=self.rank, store=store)
        dist.monitored_barrier()

class CompilerTest(test_c10d_common.CompilerTest):

    @property
    def world_size(self):
        if False:
            i = 10
            return i + 15
        return 2

    def _get_default_group(self):
        if False:
            print('Hello World!')
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(backend='gloo', rank=self.rank, world_size=self.world_size, store=store)
        return dist.distributed_c10d._get_default_group()

    def test_allreduce_work_wait_cpu(self):
        if False:
            return 10
        self._test_allreduce_work_wait(torch.ones(2, 2) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_allreduce_work_wait_gpu(self):
        if False:
            i = 10
            return i + 15
        self._test_allreduce_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    def test_allgather_work_wait_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_allgather_work_wait(torch.ones(2, 2) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_allgather_work_wait_gpu(self):
        if False:
            while True:
                i = 10
        self._test_allgather_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    def test_broadcast_work_wait_cpu(self):
        if False:
            print('Hello World!')
        self._test_broadcast_work_wait(torch.ones(2, 2) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_broadcast_work_wait_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_broadcast_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    def test_scatter_work_wait_cpu(self):
        if False:
            i = 10
            return i + 15
        self._test_scatter_work_wait(torch.ones(2, 2) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_scatter_work_wait_gpu(self):
        if False:
            return 10
        self._test_scatter_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    def test_nested_comm_tensor_wrapping(self):
        if False:
            i = 10
            return i + 15
        self._test_nested_comm_tensor_wrapping(torch.ones(2, 2) * self.rank)

    def test_consecutive_comm_work_wait_cpu(self):
        if False:
            while True:
                i = 10
        self._test_consecutive_comm_work_wait(torch.ones(2, 2) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_consecutive_comm_work_wait_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_consecutive_comm_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

class LargeCommTest(test_c10d_common.AbstractLargeCommTest, MultiProcessTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
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
    def device(self):
        if False:
            while True:
                i = 10
        return torch.device('cpu')

    @requires_gloo()
    def test_new_group_local_sync(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_new_group_local_sync(backend='gloo')

    @requires_gloo()
    def test_new_group_local_sync_sanity_check(self):
        if False:
            return 10
        self._test_new_group_local_sync_sanity_check(backend='gloo')

    @requires_gloo()
    def test_new_group_local_sync_duplicate_pg(self):
        if False:
            print('Hello World!')
        self._test_new_group_local_sync_duplicate_pg(backend='gloo')
if __name__ == '__main__':
    assert not torch.cuda._initialized, 'test_distributed must not have initialized CUDA context on main process'
    run_tests()