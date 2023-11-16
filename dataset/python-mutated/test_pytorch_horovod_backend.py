from bigdl.orca.learn.pytorch import Estimator
from unittest import TestCase
import pytest
import torch
import torch.nn as nn
import os
import numpy as np
from tempfile import TemporaryDirectory

class LinearDataset(torch.utils.data.Dataset):
    """y = a * x + b"""

    def __init__(self, a, b, size=1000):
        if False:
            return 10
        x = np.arange(0, 10, 10 / size, dtype=np.float32)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(a * x + b)

    def __getitem__(self, index):
        if False:
            return 10
        return (self.x[index, None], self.y[index, None])

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.x)

def model_creator(config):
    if False:
        return 10
    'Returns a torch.nn.Module object.'
    return nn.Linear(1, config.get('hidden_size', 1))

def optimizer_creator(model, config):
    if False:
        while True:
            i = 10
    'Returns optimizer defined upon the model parameters.'
    return torch.optim.SGD(model.parameters(), lr=config.get('lr', 0.01))

def scheduler_creator(optimizer, config):
    if False:
        while True:
            i = 10
    'Returns a learning rate scheduler wrapping the optimizer.'
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

def train_data_creator(config, batch_size):
    if False:
        return 10
    train_dataset = LinearDataset(2, 5, size=config.get('data_size', 1000))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    return train_loader

def validation_data_creator(config, batch_size):
    if False:
        for i in range(10):
            print('nop')
    val_dataset = LinearDataset(2, 5, size=config.get('val_size', 400))
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return validation_loader

class TestPyTorchEstimator(TestCase):

    def test_train(self):
        if False:
            print('Hello World!')
        estimator = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator, loss=nn.MSELoss(), scheduler_creator=scheduler_creator, config={'lr': 0.01, 'hidden_size': 1}, backend='horovod', workers_per_node=2)
        stats1 = estimator.fit(train_data_creator, batch_size=4, epochs=5)
        train_loss1 = stats1[-1]['train_loss']
        validation_loss1 = estimator.evaluate(validation_data_creator)['val_loss']
        stats2 = estimator.fit(train_data_creator, batch_size=4, epochs=3)
        train_loss2 = stats2[-1]['train_loss']
        validation_loss2 = estimator.evaluate(validation_data_creator)['val_loss']
        import ray
        import numpy as np
        remote_workers = estimator.remote_workers
        state_dicts = ray.get([worker.get_state_dict.remote() for worker in remote_workers])
        weights = [state['models'] for state in state_dicts]
        worker1_weights = weights[0][0]
        worker2_weights = weights[1][0]
        for layer in list(worker1_weights.keys()):
            assert np.allclose(worker1_weights[layer].numpy(), worker2_weights[layer].numpy())
        assert train_loss2 <= train_loss1, (train_loss2, train_loss1)
        estimator.shutdown()

    def test_horovod_initialized_correctly(self):
        if False:
            for i in range(10):
                print('nop')
        estimator = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator, loss=nn.MSELoss(), scheduler_creator=scheduler_creator, config={'lr': 0.01, 'hidden_size': 1}, backend='horovod', workers_per_node=2)

        def get_size():
            if False:
                return 10
            import horovod.torch as hvd
            return hvd.size()
        results = estimator.horovod_runner.run(get_size)
        assert results == [2, 2]

        def get_rank():
            if False:
                return 10
            import horovod.torch as hvd
            return hvd.rank()
        results = estimator.horovod_runner.run(get_rank)
        results = sorted(results)
        assert results == [0, 1]
        estimator.shutdown()

    def test_save_and_restore(self):
        if False:
            while True:
                i = 10
        estimator1 = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator, loss=nn.MSELoss(), scheduler_creator=scheduler_creator, config={'lr': 0.01, 'hidden_size': 1}, backend='horovod')
        with TemporaryDirectory() as tmp_path:
            estimator1.fit(train_data_creator, batch_size=4, epochs=1)
            checkpoint_path = os.path.join(tmp_path, 'checkpoint')
            estimator1.save(checkpoint_path)
            model1 = estimator1.get_model()
            estimator1.shutdown()
            estimator2 = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator, loss=nn.MSELoss(), scheduler_creator=scheduler_creator, config={'lr': 0.01, 'hidden_size': 1}, backend='horovod')
            estimator2.load(checkpoint_path)
            model2 = estimator2.get_model()
        model1_state_dict = model1.state_dict()
        model2_state_dict = model2.state_dict()
        assert set(model1_state_dict.keys()) == set(model2_state_dict.keys())
        for k in model1_state_dict:
            assert torch.equal(model1_state_dict[k], model2_state_dict[k])
        estimator2.shutdown()
if __name__ == '__main__':
    pytest.main([__file__])