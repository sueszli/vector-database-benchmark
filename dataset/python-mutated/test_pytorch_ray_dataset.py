import pytest
import logging
from unittest import TestCase
import ray
from ray.data import Dataset
import torch.nn as nn
import torch.optim as optim
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import MSE

def train_data_creator(a=5, b=10, size=1000):
    if False:
        while True:
            i = 10

    def get_dataset(a, b, size) -> Dataset:
        if False:
            while True:
                i = 10
        items = [i / size for i in range(size)]
        dataset = ray.data.from_items([{'x': x, 'y': a * x + b} for x in items])
        return dataset
    train_dataset = get_dataset(a, b, size)
    return train_dataset

def val_data_creator(a=5, b=10, size=100):
    if False:
        return 10

    def get_dataset(a, b, size) -> Dataset:
        if False:
            for i in range(10):
                print('nop')
        items = [i / size for i in range(size)]
        dataset = ray.data.from_items([{'x': x, 'y': a * x + b} for x in items])
        return dataset
    val_dataset = get_dataset(a, b, size)
    return val_dataset

class Net(nn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.fc1 = nn.Linear(1, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        if False:
            while True:
                i = 10
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y

def model_creator(config):
    if False:
        i = 10
        return i + 15
    net = Net()
    net = net.double()
    return net

def optim_creator(model, config):
    if False:
        i = 10
        return i + 15
    optimizer = optim.SGD(model.parameters(), lr=config.get('lr', 0.001), momentum=config.get('momentum', 0.9))
    return optimizer

def get_estimator(workers_per_node=2, model_fn=model_creator, sync_stats=False, log_level=logging.INFO, loss=nn.MSELoss(), optimizer=optim_creator):
    if False:
        i = 10
        return i + 15
    estimator = Estimator.from_torch(model=model_fn, optimizer=optimizer, loss=loss, metrics=[MSE()], config={'lr': 0.01}, workers_per_node=workers_per_node, backend='ray', sync_stats=sync_stats, log_level=log_level)
    return estimator

class TestPytorchEstimator(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        init_orca_context(runtime='ray', address='localhost:6379')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        stop_orca_context()

    def test_train_and_evaluate(self):
        if False:
            i = 10
            return i + 15
        orca_estimator = get_estimator(workers_per_node=2)
        train_dataset = train_data_creator()
        val_dataset = val_data_creator()
        start_val_stats = orca_estimator.evaluate(data=val_dataset, batch_size=32, label_cols='y', feature_cols=['x'])
        print(start_val_stats)
        train_stats = orca_estimator.fit(data=train_dataset, epochs=2, batch_size=32, label_cols='y', feature_cols=['x'])
        end_val_stats = orca_estimator.evaluate(data=val_dataset, batch_size=32, label_cols='y', feature_cols=['x'])
        print(end_val_stats)
        assert orca_estimator.get_model()
        dloss = end_val_stats['val_loss'] - start_val_stats['val_loss']
        print(f'dLoss: {dloss}')
        assert dloss < 0, 'training sanity check failed. loss increased!'

    def test_predict(self):
        if False:
            for i in range(10):
                print('nop')
        orca_estimator = get_estimator(workers_per_node=2)
        train_dataset = train_data_creator()
        result_shards = orca_estimator.predict(data=train_dataset, feature_cols=['y'])
        print('Finished Training:', result_shards)
        result_shards.show()
        assert isinstance(result_shards, ray.data.Dataset)
if __name__ == '__main__':
    pytest.main([__file__])