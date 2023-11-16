import sys
import time
from statistics import mean
from unittest.mock import patch
import torch
import torch.nn as nn
from torch import distributed as dist
from torch.cuda import Event
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import get_cycles_per_ms, run_tests, TEST_WITH_DEV_DBG_ASAN
if not dist.is_available():
    print('Distributed not available, skipping tests', file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print('Skip dev-asan as torch + multiprocessing spawn have known issues', file=sys.stderr)
    sys.exit(0)

class Layer(nn.Module):

    def __init__(self, compute_cycles, has_params: bool):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.sleep_cycles = compute_cycles
        self.optional_param = None
        if has_params:
            self.optional_param = nn.Parameter(torch.rand(1))

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        self.e1 = Event(enable_timing=True)
        self.e2 = Event(enable_timing=True)
        self.e1.record()
        if self.sleep_cycles > 0:
            torch.cuda._sleep(self.sleep_cycles)
        if self.optional_param is not None:
            x = x + self.optional_param
        self.e2.record()
        return x

    def get_time(self):
        if False:
            i = 10
            return i + 15
        return self.e1.elapsed_time(self.e2)

def _create_model(compute_cycles, has_params: bool):
    if False:
        return 10
    model = FSDP(nn.Sequential(FSDP(Layer(compute_cycles, has_params), limit_all_gathers=False), FSDP(Layer(compute_cycles, has_params), limit_all_gathers=False), FSDP(Layer(compute_cycles, has_params), limit_all_gathers=False), FSDP(Layer(compute_cycles, has_params), limit_all_gathers=False)), limit_all_gathers=False).cuda()
    return model

class Min10:

    def __init__(self):
        if False:
            return 10
        self.data = []

    def add(self, new_data):
        if False:
            for i in range(10):
                print('nop')
        if len(self.data) < 10:
            self.data.append(new_data)
        else:
            self.data = sorted(self.data)
            if new_data < self.data[-1]:
                self.data[-1] = new_data

    def avg(self):
        if False:
            i = 10
            return i + 15
        return mean(self.data)

class TestForwardOverlapWorldSizeOne(FSDPTest):

    @property
    def world_size(self):
        if False:
            while True:
                i = 10
        return 1

    def _dist_train(self):
        if False:
            while True:
                i = 10
        rank = self.rank
        world_size = self.world_size
        orig_all_gather = torch.distributed.all_gather_into_tensor

        def run(compute_cycles, all_gather_cycles):
            if False:
                print('Hello World!')
            has_params = all_gather_cycles > 0
            model = _create_model(compute_cycles, has_params)
            batch = torch.rand(1).cuda()
            batch.requires_grad = True
            out = model(batch)
            out.backward()
            model.zero_grad(set_to_none=True)
            cpu_iter = Min10()
            cpu_wait = Min10()
            gpu_compute = Min10()
            gpu_total = Min10()
            for _ in range(20):
                e1 = Event(enable_timing=True)
                e2 = Event(enable_timing=True)
                cpu_start = time.process_time()
                all_gather_called = False

                def _delayed_all_gather(*args, **kwargs):
                    if False:
                        return 10
                    nonlocal all_gather_called
                    all_gather_called = True
                    torch.cuda._sleep(all_gather_cycles)
                    assert orig_all_gather
                    return orig_all_gather(*args, **kwargs)
                e1.record()
                with patch('torch.distributed.all_gather_into_tensor', _delayed_all_gather):
                    out = model(batch)
                    if has_params and world_size > 1:
                        self.assertTrue(all_gather_called)
                    else:
                        self.assertFalse(all_gather_called)
                e2.record()
                out.backward()
                model.zero_grad(set_to_none=True)
                cpu_iter_time = time.process_time() - cpu_start
                out.item()
                cpu_wait_for_gpu_time = time.process_time() - cpu_start - cpu_iter_time
                times = []
                for mod in model.modules():
                    if not isinstance(mod, Layer):
                        continue
                    times.append(mod.get_time())
                overall_gpu_time = e1.elapsed_time(e2)
                cpu_iter.add(cpu_iter_time)
                cpu_wait.add(cpu_wait_for_gpu_time)
                gpu_compute.add(sum(times))
                gpu_total.add(overall_gpu_time)
            del model
            return {'cpu_iter': cpu_iter.avg(), 'cpu_wait': cpu_wait.avg(), 'gpu_compute': gpu_compute.avg(), 'gpu_total': gpu_total.avg()}
        sleep_cycles = int(100 * get_cycles_per_ms())
        e1 = run(0, 0)
        e2 = run(0, sleep_cycles)
        e3 = run(sleep_cycles, 0)
        e4 = run(sleep_cycles, sleep_cycles)
        debug_string = f'\nrank{rank}:\n  e1: {e1}\n  e2: {e2}\n  e3: {e3}\n  e4: {e4}'
        print(debug_string)
        short = [e1['cpu_iter'], e2['cpu_iter'], e3['cpu_iter'], e1['cpu_wait']]
        long = [e3['cpu_wait'], e4['cpu_wait']]
        if world_size == 1:
            short.append(e2['cpu_wait'])
        else:
            long.append(e2['cpu_wait'])
        for s in short:
            for l in long:
                self.assertTrue(s * 10 < l)
        short = [e1['gpu_compute'], e1['gpu_total'], e2['gpu_compute']]
        long = [e3['gpu_compute'], e3['gpu_total'], e4['gpu_compute'], e4['gpu_total']]
        if world_size == 1:
            short.append(e2['gpu_total'])
        else:
            long.append(e2['gpu_total'])
        for s in short:
            for l in long:
                self.assertTrue(s * 10 < l)
        if world_size > 1:
            compute_only = e3['gpu_compute']
            all_gather_only = e2['gpu_total']
            both = e4['gpu_total']
            self.assertTrue(compute_only + all_gather_only > 1.1 * both)

    @skip_if_lt_x_gpu(2)
    def test_forward_overlap(self):
        if False:
            return 10
        self._dist_train()

class TestForwardOverlapWorldSizeTwo(TestForwardOverlapWorldSizeOne):

    @property
    def world_size(self):
        if False:
            print('Hello World!')
        return 2
if __name__ == '__main__':
    run_tests()