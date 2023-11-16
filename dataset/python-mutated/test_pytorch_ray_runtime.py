import pytest
from unittest import TestCase
import torch
import torch.nn as nn
from bigdl.orca.learn.metrics import Accuracy, Metric
from bigdl.orca.learn.pytorch.pytorch_metrics import PytorchMetric
from bigdl.orca.learn.pytorch.pytorch_metrics import Accuracy as AccuracyMetric
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator

class LinearDataset(torch.utils.data.Dataset):

    def __init__(self, size=1000, nested_input=False):
        if False:
            for i in range(10):
                print('nop')
        self.nested_input = nested_input
        X1 = torch.randn(size // 2, 50)
        X2 = torch.randn(size // 2, 50) + 1.5
        self.x = torch.cat([X1, X2], dim=0)
        Y1 = torch.zeros(size // 2, 1)
        Y2 = torch.ones(size // 2, 1)
        self.y = torch.cat([Y1, Y2], dim=0)

    def __getitem__(self, index):
        if False:
            print('Hello World!')
        if self.nested_input:
            return ({'x': self.x[index, None]}, self.y[index, None])
        else:
            return (self.x[index, None], self.y[index, None])

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.x)

class SingleListDataset(torch.utils.data.Dataset):

    def __init__(self, size=1000, nested_input=True) -> None:
        if False:
            return 10
        super().__init__()
        self.size = size
        self.nested_input = nested_input
        X1_1 = torch.rand(self.size // 2, 1)
        X1_2 = torch.rand(self.size // 2, 1) + 1.5
        self.X1 = torch.cat([X1_1, X1_2], dim=0)
        X2_1 = torch.rand(self.size // 2, 1) + 1.5
        X2_2 = torch.rand(self.size // 2, 1) + 3.0
        self.X2 = torch.cat([X2_1, X2_2], dim=0)
        Y1 = torch.zeros(self.size // 2, 1)
        Y2 = torch.ones(self.size // 2, 1)
        self.Y = torch.cat([Y1, Y2], dim=0)

    def __getitem__(self, index):
        if False:
            return 10
        if self.nested_input:
            return ([self.X1[index], self.X2[index]], self.Y[index])
        else:
            return (self.X1[index], self.X2[index], self.Y[index])

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.size

class MultiTargetDataset(torch.utils.data.Dataset):

    def __init__(self, size=1000, nested_input=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.size = size
        self.nested_input = nested_input
        X1 = torch.randn(size // 2, 50)
        X2 = torch.randn(size // 2, 50) + 1.5
        self.X = torch.cat([X1, X2], dim=0)
        self.Y1 = torch.full((size, 1), 0.5)
        Y2_1 = torch.full((size // 2, 1), -0.5)
        Y2_2 = torch.full((size // 2, 1), 0.5)
        self.Y2 = torch.cat([Y2_1, Y2_2], dim=0)

    def __getitem__(self, index):
        if False:
            return 10
        return (self.X[index], [self.Y1[index], self.Y2[index]])

    def __len__(self):
        if False:
            return 10
        return self.size

class ComplicatedInputDataset(torch.utils.data.Dataset):

    def __init__(self, size=1000, nested_input=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.size = size
        X1_1 = torch.rand(self.size // 2, 1)
        X1_2 = torch.rand(self.size // 2, 1) + 1.5
        self.X1 = torch.cat([X1_1, X1_2], dim=0)
        X2_1 = torch.rand(self.size // 2, 1) + 1.5
        X2_2 = torch.rand(self.size // 2, 1) + 3.0
        self.X2 = torch.cat([X2_1, X2_2], dim=0)
        X3_1 = torch.rand(self.size // 2, 1) + 3.0
        X3_2 = torch.rand(self.size // 2, 1) + 4.5
        self.X3 = torch.cat([X3_1, X3_2], dim=0)
        X4_1 = torch.rand(self.size // 2, 1) + 4.5
        X4_2 = torch.rand(self.size // 2, 1) + 6.0
        self.X4 = torch.cat([X4_1, X4_2], dim=0)
        Y1 = torch.zeros(self.size // 2, 1)
        Y2 = torch.ones(self.size // 2, 1)
        self.Y = torch.cat([Y1, Y2], dim=0)

    def __getitem__(self, index):
        if False:
            return 10
        return ((self.X1[index], self.X2[index]), {'x3': self.X3[index]}, self.X4[index], self.Y[index])

    def __len__(self):
        if False:
            print('Hello World!')
        return self.size
DataSetMap = {'LinearDataset': LinearDataset, 'SingleListDataset': SingleListDataset, 'ComplicatedInputDataset': ComplicatedInputDataset, 'MultiTargetDataset': MultiTargetDataset}

def train_data_loader(config, batch_size):
    if False:
        i = 10
        return i + 15
    train_dataset = DataSetMap[config.get('dataset', 'LinearDataset')](size=config.get('data_size', 1000), nested_input=config.get('nested_input', False))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    return train_loader

def val_data_loader(config, batch_size):
    if False:
        for i in range(10):
            print('nop')
    val_dataset = DataSetMap[config.get('dataset', 'LinearDataset')](size=config.get('val_size', 400), nested_input=config.get('nested_input', False))
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return validation_loader

class Net(nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        if False:
            i = 10
            return i + 15
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y

class DictInputNet(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        if False:
            i = 10
            return i + 15
        a1 = self.fc1(input_['x'])
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y

class SingleListInputModel(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.fc = nn.Linear(2, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_list):
        if False:
            i = 10
            return i + 15
        x = torch.cat(input_list, dim=1)
        x = self.fc(x)
        x = self.out_act(x)
        return x

class MultiInputModel(nn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.fc = nn.Linear(2, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x1, x2):
        if False:
            i = 10
            return i + 15
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        x = self.out_act(x)
        return x

class MultiOutputModel(nn.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.fc = nn.Linear(50, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.fc(x)
        x = self.out_act(x)
        return (x[:-3], x[-3:])

class ComplicatedInputModel(nn.Module):

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()
        self.fc = nn.Linear(4, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x1_x2, x3_dict, x4):
        if False:
            return 10
        x = torch.cat((x1_x2[0], x1_x2[1], x3_dict['x3'], x4), dim=1)
        x = self.fc(x)
        x = self.out_act(x)
        return x

class MultiInputLoss:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.rootLoss = nn.BCELoss()

    def __call__(self, x1, x2, y1, y2):
        if False:
            for i in range(10):
                print('nop')
        x = torch.cat((x1, x2), dim=0)
        y = y1 + y2
        return self.rootLoss(x, y)

class CustomAccuracy(Metric):

    def get_pytorch_metric(self):
        if False:
            while True:
                i = 10

        class CustomAccuracyMetric(AccuracyMetric):

            def __call__(self, preds, targets):
                if False:
                    for i in range(10):
                        print('nop')
                preds = torch.cat(preds, dim=0)
                target = sum(targets)
                super().__call__(preds, target)
        return CustomAccuracyMetric()

    def get_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'Accuracy'
ModelMap = {'Net': Net, 'SingleListInputModel': SingleListInputModel, 'MultiInputModel': MultiInputModel, 'DictInputNet': DictInputNet, 'ComplicatedInputModel': ComplicatedInputModel, 'MultiOutputModel': MultiOutputModel}

def get_model(config):
    if False:
        return 10
    torch.manual_seed(0)
    return ModelMap[config.get('model', 'Net')]()

def get_optimizer(model, config):
    if False:
        i = 10
        return i + 15
    return torch.optim.SGD(model.parameters(), lr=config.get('lr', 0.01))

class TestPytorchEstimator(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        init_orca_context(runtime='ray', address='localhost:6379')

    def tearDown(self):
        if False:
            return 10
        stop_orca_context()

    def test_train(self):
        if False:
            return 10
        estimator = Estimator.from_torch(model=get_model, optimizer=get_optimizer, loss=nn.BCELoss(), metrics=Accuracy(), config={'lr': 0.01}, workers_per_node=2, backend='ray', sync_stats=True)
        start_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(start_val_stats)
        train_stats = estimator.fit(train_data_loader, epochs=1, batch_size=32)
        print(train_stats)
        end_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(end_val_stats)
        assert 0 < end_val_stats['Accuracy'] < 1
        assert estimator.get_model()
        dloss = end_val_stats['val_loss'] - start_val_stats['val_loss']
        dacc = end_val_stats['Accuracy'] - start_val_stats['Accuracy']
        print(f'dLoss: {dloss}, dAcc: {dacc}')
        assert dloss < 0 < dacc, 'training sanity check failed. loss increased!'

    def test_singlelist_input(self):
        if False:
            for i in range(10):
                print('nop')
        estimator = Estimator.from_torch(model=get_model, optimizer=get_optimizer, loss=nn.BCELoss(), metrics=Accuracy(), config={'lr': 0.01, 'model': 'SingleListInputModel', 'dataset': 'SingleListDataset', 'nested_input': True}, workers_per_node=2, backend='ray', sync_stats=True)
        start_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(start_val_stats)
        train_stats = estimator.fit(train_data_loader, epochs=1, batch_size=32)
        print(train_stats)
        end_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(end_val_stats)
        assert 0 < end_val_stats['Accuracy'] < 1
        assert estimator.get_model()
        dloss = end_val_stats['val_loss'] - start_val_stats['val_loss']
        dacc = end_val_stats['Accuracy'] - start_val_stats['Accuracy']
        print(f'dLoss: {dloss}, dAcc: {dacc}')
        assert dloss < 0 < dacc, 'training sanity check failed. loss increased!'

    def test_multi_input(self):
        if False:
            for i in range(10):
                print('nop')
        estimator = Estimator.from_torch(model=get_model, optimizer=get_optimizer, loss=nn.BCELoss(), metrics=Accuracy(), config={'lr': 0.01, 'model': 'MultiInputModel', 'dataset': 'SingleListDataset', 'nested_input': False}, workers_per_node=2, backend='ray', sync_stats=True)
        start_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(start_val_stats)
        train_stats = estimator.fit(train_data_loader, epochs=1, batch_size=32)
        print(train_stats)
        end_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(end_val_stats)
        assert 0 < end_val_stats['Accuracy'] < 1
        assert estimator.get_model()
        dloss = end_val_stats['val_loss'] - start_val_stats['val_loss']
        dacc = end_val_stats['Accuracy'] - start_val_stats['Accuracy']
        print(f'dLoss: {dloss}, dAcc: {dacc}')
        assert dloss < 0 < dacc, 'training sanity check failed. loss increased!'

    def test_dict_input(self):
        if False:
            for i in range(10):
                print('nop')
        estimator = Estimator.from_torch(model=get_model, optimizer=get_optimizer, loss=nn.BCELoss(), metrics=Accuracy(), config={'lr': 0.01, 'model': 'DictInputNet', 'dataset': 'LinearDataset', 'nested_input': True}, workers_per_node=2, backend='ray', sync_stats=True)
        start_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(start_val_stats)
        train_stats = estimator.fit(train_data_loader, epochs=1, batch_size=32)
        print(train_stats)
        end_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(end_val_stats)
        assert 0 < end_val_stats['Accuracy'] < 1
        assert estimator.get_model()
        dloss = end_val_stats['val_loss'] - start_val_stats['val_loss']
        dacc = end_val_stats['Accuracy'] - start_val_stats['Accuracy']
        print(f'dLoss: {dloss}, dAcc: {dacc}')
        assert dloss < 0 < dacc, 'training sanity check failed. loss increased!'

    def test_complicated_input(self):
        if False:
            i = 10
            return i + 15
        estimator = Estimator.from_torch(model=get_model, optimizer=get_optimizer, loss=nn.BCELoss(), metrics=Accuracy(), config={'lr': 0.01, 'model': 'ComplicatedInputModel', 'dataset': 'ComplicatedInputDataset'}, workers_per_node=2, backend='ray', sync_stats=True)
        start_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(start_val_stats)
        train_stats = estimator.fit(train_data_loader, epochs=1, batch_size=32)
        print(train_stats)
        end_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(end_val_stats)
        assert 0 < end_val_stats['Accuracy'] < 1
        assert estimator.get_model()
        dloss = end_val_stats['val_loss'] - start_val_stats['val_loss']
        dacc = end_val_stats['Accuracy'] - start_val_stats['Accuracy']
        print(f'dLoss: {dloss}, dAcc: {dacc}')
        assert dloss < 0 < dacc, 'training sanity check failed. loss increased!'

    def test_complicated_output(self):
        if False:
            while True:
                i = 10
        estimator = Estimator.from_torch(model=get_model, optimizer=get_optimizer, loss=lambda _: MultiInputLoss(), metrics=CustomAccuracy(), config={'lr': 0.01, 'model': 'MultiOutputModel', 'dataset': 'MultiTargetDataset', 'nested_input': False}, workers_per_node=2, backend='ray', sync_stats=True)
        start_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(start_val_stats)
        train_stats = estimator.fit(train_data_loader, epochs=1, batch_size=32)
        print(train_stats)
        end_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(end_val_stats)
        assert 0 < end_val_stats['Accuracy'] < 1
        assert estimator.get_model()
        dloss = end_val_stats['val_loss'] - start_val_stats['val_loss']
        dacc = end_val_stats['Accuracy'] - start_val_stats['Accuracy']
        print(f'dLoss: {dloss}, dAcc: {dacc}')
        assert dloss < 0 < dacc, 'training sanity check failed. loss increased!'
if __name__ == '__main__':
    pytest.main([__file__])