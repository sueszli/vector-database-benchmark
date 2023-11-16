import numpy as np
import pytest
import torch
import torch.distributed as dist
import ray
from ray.air.util.torch_dist import init_torch_dist_process_group, shutdown_torch_dist_process_group, TorchDistributedWorker

def test_torch_process_group_gloo():
    if False:
        print('Hello World!')

    @ray.remote
    class TestWorker(TorchDistributedWorker):

        def run(self):
            if False:
                print('Hello World!')
            tensor = torch.tensor([1.0])
            dist.all_reduce(tensor)
            return tensor.numpy()
    workers = [TestWorker.remote() for _ in range(5)]
    init_torch_dist_process_group(workers, backend='gloo', init_method='env')
    reduced = ray.get([w.run.remote() for w in workers])
    assert len(reduced) == 5
    for r in reduced:
        assert len(r) == 1
        assert r.dtype == np.float32
        assert r[0] == 5.0
    shutdown_torch_dist_process_group(workers)

def test_torch_process_group_nccl():
    if False:
        while True:
            i = 10

    @ray.remote(num_gpus=2)
    class TestWorker(TorchDistributedWorker):

        def __init__(self):
            if False:
                while True:
                    i = 10
            super().__init__()
            self.dev = f'cuda:{ray.get_gpu_ids()[0]}'

        def run(self):
            if False:
                for i in range(10):
                    print('nop')
            tensor = torch.tensor([1.0]).to(self.dev)
            dist.all_reduce(tensor)
            return tensor.cpu().numpy()
    workers = [TestWorker.remote() for _ in range(2)]
    init_torch_dist_process_group(workers, backend='nccl', init_method='env')
    reduced = ray.get([w.run.remote() for w in workers])
    assert len(reduced) == 2
    for r in reduced:
        assert len(r) == 1
        assert r.dtype == np.float32
        assert r[0] == 2.0
    shutdown_torch_dist_process_group(workers)
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', '-x', __file__]))