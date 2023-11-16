import copy
import logging
import math
import operator
import os
import random
import sys
import tempfile
from functools import reduce
import torch
import torch.distributed as c10d
if not c10d.is_available() or not c10d.is_ucc_available():
    print('c10d UCC not available, skipping tests', file=sys.stderr)
    sys.exit(0)
import test_c10d_common
import torch.distributed as dist
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from test_c10d_common import gpus_for_rank, Task, ModuleForDdpCommHook, SparseGradientModule
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import MultiProcessTestCase, requires_ucc, skip_if_lt_x_gpu, verify_ddp_error_logged
from torch.testing._internal.common_utils import TestCase, run_tests, retry_on_connect_failures, skip_but_pass_in_sandcastle

def simple_reduce_tests(rank, world_size):
    if False:
        while True:
            i = 10
    tests = [(c10d.ReduceOp.SUM, torch.tensor([rank + 1.0]), torch.tensor([float(world_size * (world_size + 1) / 2)])), (c10d.ReduceOp.PRODUCT, torch.tensor([rank + 1.0]), torch.tensor([float(math.factorial(world_size))])), (c10d.ReduceOp.MIN, torch.tensor([rank + 1.0]), torch.tensor([1.0])), (c10d.ReduceOp.MAX, torch.tensor([rank + 1.0]), torch.tensor([world_size]))]
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

class RendezvousEnvTest(TestCase):

    @requires_ucc()
    @retry_on_connect_failures
    def test_logging_init(self):
        if False:
            return 10
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(common.find_free_port())
        os.environ['RANK'] = '0'
        previous_handlers = logging.root.handlers
        c10d.init_process_group(backend='ucc', init_method='env://')
        current_handlers = logging.root.handlers
        self.assertEqual(len(previous_handlers), len(current_handlers))
        for (current, previous) in zip(current_handlers, previous_handlers):
            self.assertEqual(current, previous)
        c10d.destroy_process_group()

class TimeoutTest(test_c10d_common.AbstractTimeoutTest, TestCase):

    @requires_ucc()
    @retry_on_connect_failures
    def test_default_store_timeout_ucc(self):
        if False:
            i = 10
            return i + 15
        self._test_default_store_timeout('ucc')

class ProcessGroupUCCTest(MultiProcessTestCase):

    def _create_process_group_ucc(self):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        return c10d.ProcessGroupUCC(store, self.rank, self.world_size)

    def setUp(self):
        if False:
            i = 10
            return i + 15
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

    @requires_ucc()
    def test_empty_tensors(self):
        if False:
            i = 10
            return i + 15
        pg = self._create_process_group_ucc()
        xs = [torch.FloatTensor([])]
        fut = pg.broadcast(xs).get_future()
        fut.wait()
        output = fut.value()
        self.assertEqual(0, output[0].numel())
        self.assertEqual(xs[0], output[0], exact_dtype=False)

    def _test_broadcast_basics(self, fn):
        if False:
            for i in range(10):
                print('nop')
        pg = self._create_process_group_ucc()

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
            self.assertEqual(torch.tensor([i]), output[0], exact_dtype=False)
        x = torch.tensor([self.rank + 1.0])
        fut = pg.broadcast(x, root=0).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(torch.tensor([1.0]), result[0])

    @requires_ucc()
    def test_broadcast_basics(self):
        if False:
            return 10
        self._test_broadcast_basics(lambda t: t.clone())

    def _test_allreduce_basics(self, fn):
        if False:
            for i in range(10):
                print('nop')
        pg = self._create_process_group_ucc()
        tests = simple_reduce_tests(self.rank, self.world_size)
        for (op, input, expected) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensor = fn(input)
            fut = pg.allreduce([tensor], opts).get_future()
            fut.wait()
            result = fut.value()
            self.assertEqual(expected, result[0], exact_dtype=False)
        x = fn(torch.tensor([self.rank + 1.0]))
        fut = pg.allreduce(x).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(torch.tensor([float(self.world_size * (self.world_size + 1) / 2)]), result[0])

    @requires_ucc()
    def test_allreduce_basics(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_allreduce_basics(lambda t: t.clone())

    def _test_allgather_basics(self, fn):
        if False:
            i = 10
            return i + 15
        pg = self._create_process_group_ucc()
        for n in [1]:
            input = [fn(torch.tensor([n * self.rank + i])) for i in range(n)]
            output = [[fn(torch.tensor([-1])) for _ in range(n * self.world_size)] for _ in range(n)]
            expected_output = [[fn(torch.tensor([i])) for i in range(n * self.world_size)] for _ in range(n)]
            fut = pg.allgather(output, input).get_future()
            fut.wait()
            result = fut.value()
            if n == 1:
                result = [result]
            self.assertEqual(expected_output, result)

    def test_allgather_basics(self):
        if False:
            while True:
                i = 10
        self._test_allgather_basics(lambda t: t.clone())

    def _test_reduce_basics(self, fn):
        if False:
            return 10
        pg = self._create_process_group_ucc()
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
                    self.assertEqual(output, result[0], exact_dtype=False)

    @requires_ucc()
    def test_reduce_basics(self):
        if False:
            while True:
                i = 10
        self._test_reduce_basics(lambda t: t.clone())

    @requires_ucc()
    def test_send_recv_all_to_all(self):
        if False:
            i = 10
            return i + 15
        pg = self._create_process_group_ucc()
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

    @skip_but_pass_in_sandcastle('fails with numerical mismatch, skip for now')
    @requires_ucc()
    def test_barrier_implies_wait(self):
        if False:
            return 10
        pg = self._create_process_group_ucc()
        size = (100, 100)
        num = 16
        tensors = [torch.full(size, float(i)) for i in range(num)]
        for tensor in tensors:
            pg.allreduce(tensor)
        pg.barrier().get_future().wait()
        for (i, tensor) in enumerate(tensors):
            self.assertEqual(torch.full(size, float(i * self.world_size)), tensor)

class DistributedDataParallelTest(test_c10d_common.CommonDistributedDataParallelTest, MultiProcessTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self._spawn_processes()

    def _get_process_group(self):
        if False:
            while True:
                i = 10
        store = self._get_store()
        c10d.init_process_group('ucc', store=store, rank=self.rank, world_size=self.world_size)
        return c10d.distributed_c10d._get_default_group()

    def _test_ucc_backend(self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False):
        if False:
            while True:
                i = 10
        process_group = self._get_process_group()
        self._test_ddp_with_process_group(process_group, devices, device_ids, multi_device, gradient_as_bucket_view)

    @requires_ucc()
    def test_ucc_backend_cpu_module(self):
        if False:
            i = 10
            return i + 15
        self._test_ucc_backend([torch.device('cpu')], None)

    @requires_ucc()
    def test_ucc_backend_cpu_module_grad_is_view(self):
        if False:
            print('Hello World!')
        self._test_ucc_backend([torch.device('cpu')], None, gradient_as_bucket_view=True)

    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_ucc_backend_1gpu_module_device_ids_integer_list(self):
        if False:
            return 10
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device('cuda:' + str(i)) for i in int_devices]
        self._test_ucc_backend(devices, int_devices)

    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_ucc_backend_1gpu_module_device_ids_torch_device_list(self):
        if False:
            i = 10
            return i + 15
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device('cuda:' + str(i)) for i in int_devices]
        self._test_ucc_backend(devices, devices)

    @skip_but_pass_in_sandcastle('requires broadcast coalesced, which is not supported by ucc currently')
    @requires_ucc()
    @skip_if_lt_x_gpu(4)
    def test_ucc_backend_2gpu_module(self):
        if False:
            while True:
                i = 10
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device('cuda:' + str(i)) for i in int_devices]
        self._test_ucc_backend(devices, None, multi_device=True)

    @skip_but_pass_in_sandcastle('requires broadcast coalesced, which is not supported by ucc currently')
    @requires_ucc()
    @skip_if_lt_x_gpu(8)
    def test_ucc_backend_4gpu_module(self):
        if False:
            return 10
        int_devices = gpus_for_rank(self.world_size)[self.rank][:4]
        devices = [torch.device('cuda:' + str(i)) for i in int_devices]
        self._test_ucc_backend(devices, None, multi_device=True)

    def _test_global_local_unused_params_grad(self, gradient_as_bucket_view=False, static_graph=False):
        if False:
            return 10
        '\n        By simulating a multi-task training, this test is to make sure:\n        1) DDP does not touch the grad of globally unused parameters.\n        2) DDP does update the grad of locally unused parameters.\n        '

        class GlobalLocalUnusedParamModule(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.t0 = Task()
                self.t1 = Task()
                self.task_unused = Task()

            def task_parameters(self):
                if False:
                    for i in range(10):
                        print('nop')
                return (self.t0.p, self.t1.p, self.task_unused.p)

            def forward(self, x, rank):
                if False:
                    while True:
                        i = 10
                return self.t0(x) if rank == 0 else self.t1(x)

        def run_and_verify_grad(model):
            if False:
                while True:
                    i = 10
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

    @skip_but_pass_in_sandcastle('times out')
    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad(self):
        if False:
            i = 10
            return i + 15
        self._test_global_local_unused_params_grad()

    @skip_but_pass_in_sandcastle('times out')
    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad_with_grad_is_view(self):
        if False:
            print('Hello World!')
        self._test_global_local_unused_params_grad(gradient_as_bucket_view=True)

    @skip_but_pass_in_sandcastle('times out')
    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad_with_static_graph(self):
        if False:
            return 10
        self._test_global_local_unused_params_grad(static_graph=True)

    @skip_but_pass_in_sandcastle('times out')
    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_find_unused_parameters_when_unused_parameters_empty(self):
        if False:
            while True:
                i = 10
        '\n        An empty unused_parameters array does not imply find_unused_parameters =\n        false. This test makes sure that DDP allreduces unused parameters\n        accordingly where the forward pass in some process uses all parameters.\n        This unit test creates a module that uses all parameters in rank = 0, and\n        has unused parameters in other ranks.\n        '

        class FindUnusedParamModule(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
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
                    print('Hello World!')
                return self.t1(self.t0(x)) if rank == 0 else self.t1(x)

        def run_and_verify_grad(model):
            if False:
                i = 10
                return i + 15
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

    @requires_ucc()
    def test_ignored_output(self):
        if False:
            print('Hello World!')
        '\n        Test that the output of a model can be ignored and that there is no\n        implicit requirement that `backward` gets called.\n        '
        process_group = self._get_process_group()

        class IgnoredOutput(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                if False:
                    return 10
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

    @requires_ucc()
    def test_ignored_output_with_unused_parameters(self):
        if False:
            while True:
                i = 10
        '\n        Test that the output of a model can be ignored and that there is no\n        implicit requirement that `backward` gets called, if not all model\n        parameters participated in computing the model output.\n        '
        process_group = self._get_process_group()

        class IgnoredOutputWithUnusedParameters(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
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

    def _run_and_verify_sparse_gradients(self, vanilla_model, ddp_model):
        if False:
            print('Hello World!')
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

    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_save_load_checkpoint(self):
        if False:
            i = 10
            return i + 15
        dist.init_process_group('ucc', init_method=f'file://{self.file_name}', world_size=self.world_size, rank=self.rank)

        class TestModel(nn.Module):

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
                    return 10
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        def train_loop(model, optimizer, iterations):
            if False:
                while True:
                    i = 10
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
            return 10
        process_group = self._get_process_group()
        torch.manual_seed(1337)
        vanilla_model = SparseGradientModule()
        ddp_model = DistributedDataParallel(copy.deepcopy(vanilla_model), process_group=process_group, gradient_as_bucket_view=gradient_as_bucket_view)
        self._run_and_verify_sparse_gradients(vanilla_model, ddp_model)

    @skip_but_pass_in_sandcastle('backward pass: input tensor has to be dense')
    @requires_ucc()
    def test_sparse_gradients(self):
        if False:
            return 10
        self._test_sparse_gradients()

    @skip_but_pass_in_sandcastle('backward pass: input tensor has to be dense')
    @requires_ucc()
    def test_sparse_gradients_grad_is_view(self):
        if False:
            print('Hello World!')
        self._test_sparse_gradients(gradient_as_bucket_view=True)

    @requires_ucc()
    def test_ddp_comm_hook_future_passing_cpu(self):
        if False:
            print('Hello World!')
        '\n        This unit test verifies whether the Future object is passed properly.\n        The callback function creates a Future object and sets a value to it.\n        '
        process_group = self._get_process_group()
        cpu_model = DistributedDataParallel(ModuleForDdpCommHook().cpu(), process_group=process_group)
        cpu_model.register_comm_hook(None, self._simple_hook)
        self._run_and_verify_hook(cpu_model, 8, 2 * torch.ones(2, 2))

    def _gpu_model_with_ddp_comm_hook(self, process_group, hook=None, gradient_as_bucket_view=False, state=None):
        if False:
            while True:
                i = 10
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(ModuleForDdpCommHook().to(device_id), device_ids=[device_id], process_group=process_group, gradient_as_bucket_view=gradient_as_bucket_view)
        if hook is not None:
            gpu_model.register_comm_hook(state, hook)
        return gpu_model

    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_future_passing_gpu_ucc(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This unit test verifies whether the Future object is passed properly using ucc backend.\n        The hook callback function creates a Future object and sets a value to it.\n        '
        process_group = self._get_process_group()
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, self._simple_hook)
        self._run_and_verify_hook(gpu_model, 8, 2 * torch.ones(2, 2))

    @requires_ucc()
    def test_ddp_invalid_comm_hook_init(self):
        if False:
            i = 10
            return i + 15
        '\n        This unit test makes sure that register_comm_hook properly checks the format\n        of hook defined by user. The Python hook must be callable. This test also\n        checks whether bucket annotation checked properly if defined.\n        '
        process_group = self._get_process_group()
        model = DistributedDataParallel(ModuleForDdpCommHook(), process_group=process_group)
        with self.assertRaisesRegex(TypeError, 'Communication hook must be callable.'):
            model.register_comm_hook(state=None, hook=1)
        with self.assertRaisesRegex(ValueError, 'bucket annotation should be dist.GradBucket.'):

            def comm_hook(state: object, bucket: int) -> torch.futures.Future[torch.Tensor]:
                if False:
                    return 10
                return torch.futures.Future()
            model.register_comm_hook(state=None, hook=comm_hook)

    @requires_ucc()
    def test_ddp_invalid_comm_hook_return_type(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        This test checks whether return annotation checked properly if defined. It also\n        checks whether an internal error is thrown if return type is incorrect and user\n        hasn't specified any return type annotation.\n        "
        process_group = self._get_process_group()
        model = DistributedDataParallel(ModuleForDdpCommHook(), process_group=process_group)
        expected_err = 'Communication hook: return annotation should be torch.futures.Future'
        with self.assertRaisesRegex(ValueError, expected_err):

            def comm_hook(state: object, bucket: dist.GradBucket) -> int:
                if False:
                    return 10
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

    @requires_ucc()
    def test_ddp_comm_hook_register_just_once(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        DDP communication hook can only be registered once. This test validates whether\n        the error is thrown properly when register_comm_hook is called more than once.\n        '
        process_group = self._get_process_group()
        model = DistributedDataParallel(ModuleForDdpCommHook(), process_group=process_group)

        def dummy_hook(state, bucket):
            if False:
                while True:
                    i = 10
            fut = torch.futures.Future()
            fut.set_result([bucket.buffer()])
            return fut
        model.register_comm_hook(None, dummy_hook)
        with self.assertRaisesRegex(RuntimeError, 'register_comm_hook or register_builtin_comm_hook can only be called once.'):
            model.register_comm_hook(None, dummy_hook)

    @skip_but_pass_in_sandcastle('backward pass: input tensor has to be dense')
    @requires_ucc()
    def test_ddp_comm_hook_sparse_gradients(self):
        if False:
            print('Hello World!')
        '\n        Runs "test_sparse_gradients" unit test with DDP communication hook. We define a\n        simple hook that does allreduce and works with ucc backend for this test.\n        '
        process_group = self._get_process_group()
        torch.manual_seed(1337)
        vanilla_model = SparseGradientModule()
        ddp_model = DistributedDataParallel(copy.deepcopy(vanilla_model), process_group=process_group)

        def allreduce_hook_ucc(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
            if False:
                i = 10
                return i + 15

            def div_by_world_size(fut):
                if False:
                    while True:
                        i = 10
                return fut.wait()[0] / self.world_size
            fut = process_group.allreduce([bucket.buffer()]).get_future()
            return fut.then(div_by_world_size)
        ddp_model.register_comm_hook(None, allreduce_hook_ucc)
        self._run_and_verify_sparse_gradients(vanilla_model, ddp_model)

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
            while True:
                i = 10
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_default_pg_ucc(self):
        if False:
            print('Hello World!')
        self._test_sequence_num_set_default_pg(backend='ucc')

    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_ucc_new_group(self):
        if False:
            return 10
        self._test_sequence_num_set_new_group(backend='ucc')

    @skip_if_lt_x_gpu(2)
    @requires_ucc()
    def test_sequence_num_incremented_ucc_default(self):
        if False:
            i = 10
            return i + 15
        self._test_sequence_num_incremented_default_group('ucc')

    @skip_if_lt_x_gpu(4)
    @requires_ucc()
    def test_sequence_num_incremented_ucc_subgroup(self):
        if False:
            return 10
        if self.world_size < 4:
            return skip_but_pass_in_sandcastle('Test requires world_size of at least 4')
        self._test_sequence_num_incremented_subgroup('ucc')

    @skip_but_pass_in_sandcastle('Fails on M60')
    @requires_ucc()
    def test_ucc_barrier_device_ids(self):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(backend='ucc', rank=self.rank, world_size=self.world_size, store=store)
        with self.assertRaisesRegex(RuntimeError, 'device_ids not supported'):
            c10d.barrier(device_ids=[self.rank])

    @skip_but_pass_in_sandcastle('Fails on M60')
    @skip_if_lt_x_gpu(2)
    @requires_ucc()
    def test_ucc_warn_not_in_group(self):
        if False:
            print('Hello World!')
        self._test_warn_not_in_group(backend='ucc')

    @skip_if_lt_x_gpu(2)
    @requires_ucc()
    def test_ucc_rank_membership(self):
        if False:
            print('Hello World!')
        self._test_rank_membership(backend='ucc')

    @skip_if_lt_x_gpu(2)
    @requires_ucc()
    def test_tensor_dtype_mismatch(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_tensor_dtype_mismatch(backend='ucc')

    @skip_if_lt_x_gpu(2)
    @requires_ucc()
    def test_tensor_dtype_complex(self):
        if False:
            while True:
                i = 10
        self._test_tensor_dtype_complex(backend='ucc')

class CompilerTest(test_c10d_common.CompilerTest):

    @property
    def world_size(self):
        if False:
            for i in range(10):
                print('nop')
        return 2

    def _get_default_group(self):
        if False:
            i = 10
            return i + 15
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(backend='ucc', rank=self.rank, world_size=self.world_size, store=store)
        return dist.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)
    def test_allreduce_work_wait_gpu(self):
        if False:
            while True:
                i = 10
        self._test_allreduce_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_allgather_work_wait_gpu(self):
        if False:
            i = 10
            return i + 15
        self._test_allgather_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_broadcast_work_wait_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_broadcast_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    @skip_if_lt_x_gpu(2)
    def test_nested_comm_tensor_wrapping_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_nested_comm_tensor_wrapping(torch.ones(2, 2, device=self.rank) * self.rank)

    def test_consecutive_comm_work_wait_gpu(self):
        if False:
            i = 10
            return i + 15
        self._test_consecutive_comm_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    def test_allreduce_work_wait_cpu(self):
        if False:
            return 10
        self._test_allreduce_work_wait(torch.ones(2, 2) * self.rank)

    def test_allgather_work_wait_cpu(self):
        if False:
            return 10
        self._test_allgather_work_wait(torch.ones(2, 2) * self.rank)

    def test_broadcast_work_wait_cpu(self):
        if False:
            print('Hello World!')
        self._test_broadcast_work_wait(torch.ones(2, 2) * self.rank)

    def test_nested_comm_tensor_wrapping_cpu(self):
        if False:
            while True:
                i = 10
        self._test_nested_comm_tensor_wrapping(torch.ones(2, 2) * self.rank)

    def test_consecutive_comm_work_wait_cpu(self):
        if False:
            i = 10
            return i + 15
        self._test_consecutive_comm_work_wait(torch.ones(2, 2) * self.rank)

class UccProcessGroupWithDispatchedCollectivesTests(test_c10d_common.ProcessGroupWithDispatchedCollectivesTests):

    @skip_but_pass_in_sandcastle('Fails on M60')
    @requires_ucc()
    @skip_if_lt_x_gpu(1)
    def test_collectives(self):
        if False:
            i = 10
            return i + 15
        self._test_collectives(backend='ucc')

    @skip_but_pass_in_sandcastle('Fails on M60')
    @requires_ucc()
    @skip_if_lt_x_gpu(1)
    def test_allgather_base(self):
        if False:
            i = 10
            return i + 15
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group('ucc', world_size=self.world_size, rank=self.rank, store=store)
        device = 'cuda'
        tensor = torch.ones(10, 10, device=torch.device(device))
        output_tensor = torch.zeros(10, 10, device=torch.device(device))
        dist.all_gather_into_tensor(output_tensor, tensor)
        self.assertEqual(output_tensor, tensor)
if __name__ == '__main__':
    assert not torch.cuda._initialized, 'test_distributed must not have initialized CUDA context on main process'
    run_tests()