import pytest
import torch
from torch.distributed.fsdp import FullyShardedDataParallel
import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

@pytest.fixture
def ray_start_4_cpus_2_gpus():
    if False:
        print('Hello World!')
    address_info = ray.init(num_cpus=4, num_gpus=2)
    yield address_info
    ray.shutdown()

def test_torch_fsdp(ray_start_4_cpus_2_gpus):
    if False:
        i = 10
        return i + 15
    'Tests if ``prepare_model`` correctly wraps in FSDP.'

    def train_fn():
        if False:
            return 10
        model = torch.nn.Linear(1, 1)
        model = train.torch.prepare_model(model, parallel_strategy='fsdp')
        assert isinstance(model, FullyShardedDataParallel)
        assert next(model.parameters()).is_cuda
    trainer = TorchTrainer(train_fn, scaling_config=ScalingConfig(num_workers=2, use_gpu=True))
    trainer.fit()
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', '-x', '-s', __file__]))