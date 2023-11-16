import os
import sys
import torch
from torch import nn
import torch.distributed as dist
if not dist.is_available():
    print('Distributed not available, skipping tests', file=sys.stderr)
    sys.exit(0)
from torch.distributed.algorithms.ddp_comm_hooks import DDPCommHookType, register_ddp_comm_hook
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import MultiProcessTestCase, requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
if TEST_WITH_DEV_DBG_ASAN:
    print('Multiprocessing spawn is not compatible with dev/dbg asan', file=sys.stderr)
    sys.exit(0)

def gpus_for_rank(world_size):
    if False:
        print('Hello World!')
    visible_devices = list(range(torch.cuda.device_count()))
    gpus_per_process = torch.cuda.device_count() // world_size
    gpus_for_rank = []
    for rank in range(world_size):
        gpus_for_rank.append(visible_devices[rank * gpus_per_process:(rank + 1) * gpus_per_process])
    return gpus_for_rank

class Task(nn.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        torch.manual_seed(0)
        self.p = nn.Parameter(torch.randn(40, 20))

    def forward(self, x):
        if False:
            while True:
                i = 10
        return self.p * x

class TestDdpCommHook(nn.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.t0 = Task()

    def forward(self, x, rank):
        if False:
            while True:
                i = 10
        return self.t0(x ** (1 + rank))

class DistributedDataParallelCommHookTest(MultiProcessTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        if False:
            print('Hello World!')
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _get_process_group_nccl(self):
        if False:
            for i in range(10):
                print('nop')
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(backend='nccl', world_size=self.world_size, rank=self.rank, store=store)
        return dist.distributed_c10d._get_default_group()

    @property
    def world_size(self):
        if False:
            i = 10
            return i + 15
        return 2

    def _local_model(self):
        if False:
            while True:
                i = 10
        local_model = TestDdpCommHook().cpu()
        return local_model

    def _get_grads(self, process_group, hook_type=None):
        if False:
            print('Hello World!')
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(TestDdpCommHook().to(device_id), device_ids=[device_id], process_group=process_group)
        if hook_type is not None:
            register_ddp_comm_hook(comm_hook_type=hook_type, model=gpu_model, state=process_group)
        return self._run_and_get_grads(gpu_model)

    def _run_and_get_grads(self, model):
        if False:
            return 10
        torch.manual_seed(2020)
        input = torch.randn(40, 20)
        output = model(input, self.rank)
        output.mean().backward()
        param = next(model.parameters())
        return param.grad

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook(self):
        if False:
            print('Hello World!')
        '\n        This unit test verifies the ``allreduce`` hook registered case gives same result\n        with no hook registered case.\n        '
        process_group = self._get_process_group_nccl()
        reference_grads = self._get_grads(process_group, None)
        hook_grads = self._get_grads(process_group, DDPCommHookType.ALLREDUCE)
        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-05, atol=0)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_fp16compress_hook(self):
        if False:
            return 10
        '\n        This unit test verifies the ``fp16 compress`` hook registered case\n        gives close result with no hook registered case.\n        '
        process_group = self._get_process_group_nccl()
        reference_grads = self._get_grads(process_group, None)
        hook_grads = self._get_grads(process_group, DDPCommHookType.FP16_COMPRESS)
        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-05, atol=0.0001)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_quantize_per_tensor_hook(self):
        if False:
            i = 10
            return i + 15
        '\n        This unit test verifies the ``quantize per tensor`` hook registered case\n        gives close result with no hook registered case.\n        '
        process_group = self._get_process_group_nccl()
        reference_grads = self._get_grads(process_group, None)
        hook_grads = self._get_grads(process_group, DDPCommHookType.QUANTIZE_PER_TENSOR)
        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-05, atol=0.0001)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_quantize_per_channel_hook(self):
        if False:
            return 10
        '\n        This unit test verifies the ``quantize per channel`` hook registered case\n        gives close result with no hook registered case.\n        '
        process_group = self._get_process_group_nccl()
        reference_grads = self._get_grads(process_group, None)
        hook_grads = self._get_grads(process_group, DDPCommHookType.QUANTIZE_PER_CHANNEL)
        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-05, atol=0.0001)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_noop_hook(self):
        if False:
            while True:
                i = 10
        '\n        This unit test verifies the ``noop`` hook registered case and a subsequent allreduce\n        gives same result with no hook registered case.\n        '
        process_group = self._get_process_group_nccl()
        reference_grads = self._get_grads(process_group, None)
        hook_grads = self._get_grads(process_group, DDPCommHookType.NOOP)
        hook_grads.div_(self.world_size)
        dist.all_reduce(hook_grads, group=process_group)
        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-05, atol=0)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_is_last_hook(self):
        if False:
            i = 10
            return i + 15
        process_group = self._get_process_group_nccl()

        def hook(flags, bucket):
            if False:
                for i in range(10):
                    print('nop')
            flags.append(bucket.is_last())
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut
        flags = []
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = nn.Sequential(nn.Linear(2, 4000, bias=False), *[nn.Linear(4000, 4000, bias=False) for _ in range(10)])
        gpu_model = DistributedDataParallel(model.to(device_id), device_ids=[device_id], process_group=process_group)
        gpu_model.register_comm_hook(state=flags, hook=hook)
        input = torch.randn(10, 2)
        gpu_model(input).sum().backward()
        self.assertTrue(flags[-1])
        self.assertFalse(any(flags[:-1]))
if __name__ == '__main__':
    assert not torch.cuda._initialized, 'test_distributed must not have initialized CUDA context on main process'
    run_tests()