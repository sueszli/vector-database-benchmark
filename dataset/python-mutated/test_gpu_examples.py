import os
from tempfile import TemporaryDirectory
import pytest
import torch
from ray import train
from ray.air.constants import TRAINING_ITERATION
from ray.train import Checkpoint, ScalingConfig
from ray.train.examples.horovod.horovod_example import train_func as horovod_torch_train_func
from ray.train.examples.pytorch.torch_fashion_mnist_example import train_func_per_worker as fashion_mnist_train_func
from ray.train.examples.tf.tensorflow_mnist_example import train_func as tensorflow_mnist_train_func
from ray.train.horovod.horovod_trainer import HorovodTrainer
from ray.train.tensorflow.tensorflow_trainer import TensorflowTrainer
from ray.train.tests.test_tune import torch_fashion_mnist, tune_tensorflow_mnist
from ray.train.torch.torch_trainer import TorchTrainer

def test_tensorflow_mnist_gpu(ray_start_4_cpus_2_gpus):
    if False:
        return 10
    num_workers = 2
    epochs = 3
    config = {'lr': 0.001, 'batch_size': 64, 'epochs': epochs}
    trainer = TensorflowTrainer(tensorflow_mnist_train_func, train_loop_config=config, scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=True))
    results = trainer.fit()
    result = results.metrics
    assert result[TRAINING_ITERATION] == epochs

def test_torch_fashion_mnist_gpu(ray_start_4_cpus_2_gpus):
    if False:
        for i in range(10):
            print('nop')
    num_workers = 2
    epochs = 3
    config = {'lr': 0.001, 'batch_size_per_worker': 32, 'epochs': epochs}
    trainer = TorchTrainer(fashion_mnist_train_func, train_loop_config=config, scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=True))
    results = trainer.fit()
    result = results.metrics
    assert result[TRAINING_ITERATION] == epochs

def test_horovod_torch_mnist_gpu(ray_start_4_cpus_2_gpus):
    if False:
        i = 10
        return i + 15
    num_workers = 2
    num_epochs = 2
    trainer = HorovodTrainer(horovod_torch_train_func, train_loop_config={'num_epochs': num_epochs, 'lr': 0.001}, scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=True))
    results = trainer.fit()
    result = results.metrics
    assert result[TRAINING_ITERATION] == num_workers

def test_horovod_torch_mnist_gpu_checkpoint(ray_start_4_cpus_2_gpus):
    if False:
        return 10

    def checkpointing_func(config):
        if False:
            i = 10
            return i + 15
        net = torch.nn.Linear(in_features=8, out_features=16)
        net.to('cuda')
        with TemporaryDirectory() as tmpdir:
            torch.save(net.state_dict(), os.path.join(tmpdir, 'checkpoint.pt'))
            train.report({'metric': 1}, checkpoint=Checkpoint.from_directory(tmpdir))
    num_workers = 2
    trainer = HorovodTrainer(checkpointing_func, scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=True))
    trainer.fit()

def test_tune_fashion_mnist_gpu(ray_start_4_cpus_2_gpus):
    if False:
        while True:
            i = 10
    torch_fashion_mnist(num_workers=2, use_gpu=True, num_samples=1)

def test_concurrent_tune_fashion_mnist_gpu(ray_start_4_cpus_2_gpus):
    if False:
        print('Hello World!')
    torch_fashion_mnist(num_workers=1, use_gpu=True, num_samples=2)

def test_tune_tensorflow_mnist_gpu(ray_start_4_cpus_2_gpus):
    if False:
        while True:
            i = 10
    tune_tensorflow_mnist(num_workers=2, use_gpu=True, num_samples=1)

def test_train_linear_dataset_gpu(ray_start_4_cpus_2_gpus):
    if False:
        print('Hello World!')
    from ray.train.examples.pytorch.torch_regression_example import train_regression
    assert train_regression(num_workers=2, use_gpu=True)
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', '-x', '-s', __file__]))