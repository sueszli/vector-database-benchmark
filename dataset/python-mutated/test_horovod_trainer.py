import os
import pytest
import torch
import torch.nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import ray
from ray.train import ScalingConfig
from ray.train.examples.horovod.horovod_pytorch_example import Net
from ray.train.examples.horovod.horovod_pytorch_example import train_func as hvd_train_func
from ray.train.horovod import HorovodTrainer
from ray.train.torch import TorchPredictor

@pytest.fixture
def ray_start_4_cpus():
    if False:
        print('Hello World!')
    address_info = ray.init(num_cpus=4)
    yield address_info
    ray.shutdown()

def run_image_prediction(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    if False:
        print('Hello World!')
    model.eval()
    with torch.no_grad():
        return torch.exp(model(images)).argmax(dim=1)

def test_horovod(ray_start_4_cpus):
    if False:
        print('Hello World!')

    def train_func(config):
        if False:
            print('Hello World!')
        result = hvd_train_func(config)
        assert len(result) == epochs
        assert result[-1] < result[0]
    num_workers = 1
    epochs = 10
    scaling_config = ScalingConfig(num_workers=num_workers)
    config = {'num_epochs': epochs, 'save_model_as_dict': False}
    trainer = HorovodTrainer(train_loop_per_worker=train_func, train_loop_config=config, scaling_config=scaling_config)
    result = trainer.fit()
    model = Net()
    with result.checkpoint.as_directory() as checkpoint_dir:
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'model.pt')))
    predictor = TorchPredictor(model=model)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    test_dataloader = DataLoader(test_set, batch_size=10)
    test_dataloader_iter = iter(test_dataloader)
    (images, labels) = next(test_dataloader_iter)
    predicted_labels = run_image_prediction(predictor.model, images)
    assert torch.equal(predicted_labels, labels)
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', '-x', __file__]))