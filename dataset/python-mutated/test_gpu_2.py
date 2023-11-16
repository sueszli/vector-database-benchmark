import numpy as np
import pytest
import torch
import ray
import ray.data
import ray.train as train
from ray import tune
from ray.air.config import ScalingConfig
from ray.train.examples.pytorch.torch_linear_example import LinearDataset
from ray.train.torch.torch_trainer import TorchTrainer

class LinearDatasetDict(LinearDataset):
    """Modifies the LinearDataset to return a Dict instead of a Tuple."""

    def __getitem__(self, index):
        if False:
            print('Hello World!')
        return {'x': self.x[index, None], 'y': self.y[index, None]}

class NonTensorDataset(LinearDataset):
    """Modifies the LinearDataset to also return non-tensor objects."""

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        return {'x': self.x[index, None], 'y': 2}

class TorchTrainerPatchedMultipleReturns(TorchTrainer):

    def _report(self, training_iterator) -> None:
        if False:
            print('Hello World!')
        for results in training_iterator:
            tune.report(results=results)

@pytest.mark.parametrize('use_gpu', (True, False))
def test_torch_iter_torch_batches_auto_device(ray_start_4_cpus_2_gpus, use_gpu):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests that iter_torch_batches in TorchTrainer worker function uses the\n    default device.\n    '

    def train_fn():
        if False:
            return 10
        dataset = train.get_dataset_shard('train')
        for batch in dataset.iter_torch_batches(dtypes=torch.float, device='cpu'):
            assert str(batch['data'].device) == 'cpu'
        for batch in dataset.iter_torch_batches(dtypes=torch.float):
            assert str(batch['data'].device) == str(train.torch.get_device())
    dataset = ray.data.from_numpy(np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]).T)
    for batch in dataset.iter_torch_batches(dtypes=torch.float, device='cpu'):
        assert str(batch['data'].device) == 'cpu'
    trainer = TorchTrainer(train_fn, scaling_config=ScalingConfig(num_workers=2, use_gpu=use_gpu), datasets={'train': dataset})
    trainer.fit()
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', '-x', '-s', __file__]))