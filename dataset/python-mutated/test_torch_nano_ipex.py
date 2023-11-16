import os
import pytest
from unittest import TestCase
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from bigdl.nano.pytorch import TorchNano
from bigdl.nano.pytorch.vision.models import vision
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_2_0
from bigdl.nano.utils.common import _avx2_checker
batch_size = 256
num_workers = 0
data_dir = '/tmp/data'

class ResNet18(nn.Module):

    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        if False:
            while True:
                i = 10
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = nn.Sequential(backbone, head)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return self.model(x)

class MyNano(TorchNano):

    def train(self, optimizer_supported: bool=False):
        if False:
            print('Hello World!')
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss_func = nn.CrossEntropyLoss()
        if optimizer_supported:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        train_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)
        (model, optimizer, train_loader) = self.setup(model, optimizer, train_loader)
        model.train()
        num_epochs = 1
        for _i in range(num_epochs):
            (total_loss, num) = (0, 0)
            for (X, y) in train_loader:
                optimizer.zero_grad()
                loss = loss_func(model(X), y)
                self.backward(loss)
                optimizer.step()
                total_loss += loss.sum()
                num += 1
            print(f'avg_loss: {total_loss / num}')

class LinearModel(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.fc1 = nn.Linear(1, 1, bias=False)
        self.fc1.weight.data.fill_(1.0)

    def forward(self, input_):
        if False:
            i = 10
            return i + 15
        return self.fc1(input_)

class MyNanoCorrectness(TorchNano):

    def train(self, lr):
        if False:
            return 10
        dataset = TensorDataset(torch.tensor([[0.0], [0.0], [1.0], [1.0]]), torch.tensor([[0.0], [0.0], [0.0], [0.0]]))
        train_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=False)
        origin_model = LinearModel()
        loss_func = nn.MSELoss()
        optimizer = torch.optim.SGD(origin_model.parameters(), lr=lr)
        (model, optimizer, train_loader) = self.setup(origin_model, optimizer, train_loader)
        model.train()
        num_epochs = 2
        for _i in range(num_epochs):
            for (X, y) in train_loader:
                optimizer.zero_grad()
                loss = loss_func(model(X), y)
                self.backward(loss)
                optimizer.step()
        assert model.fc1.weight.data == 0.25, f'wrong weights: {model.fc1.weight.data}'

class MyNanoLoadStateDict(TorchNano):

    def train(self, lr):
        if False:
            for i in range(10):
                print('nop')
        dataset = TensorDataset(torch.tensor([[0.0], [0.0], [1.0], [1.0]]), torch.tensor([[0.0], [0.0], [0.0], [0.0]]))
        train_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=False)
        loss_func = nn.MSELoss()
        origin_model = LinearModel()
        origin_optimizer = torch.optim.SGD(origin_model.parameters(), lr=lr)

        def train_one_epoch(model, optimizer, loss_func, data_loader):
            if False:
                print('Hello World!')
            for (X, y) in data_loader:
                optimizer.zero_grad()
                loss = loss_func(model(X), y)
                self.backward(loss)
                optimizer.step()
        (model, optimizer, train_loader) = self.setup(origin_model, origin_optimizer, train_loader)
        model.train()
        train_one_epoch(model, optimizer, loss_func, train_loader)
        origin_model.load_state_dict(model.state_dict())
        origin_optimizer.load_state_dict(optimizer.state_dict())
        (model, optimizer) = self.setup(origin_model, origin_optimizer)
        model.train()
        train_one_epoch(model, optimizer, loss_func, train_loader)
        assert model.fc1.weight.data == 0.25, f'wrong weights: {model.fc1.weight.data}'

class Lite:

    def setUp(self):
        if False:
            return 10
        test_dir = os.path.dirname(__file__)
        project_test_dir = os.path.abspath(os.path.join(test_dir, '..', '..', '..', '..', '..'))
        os.environ['PYTHONPATH'] = project_test_dir

    def test_torch_nano(self):
        if False:
            for i in range(10):
                print('nop')
        MyNano(use_ipex=True).train()

    def test_torch_nano_spawn(self):
        if False:
            print('Hello World!')
        MyNano(use_ipex=True, num_processes=2, distributed_backend='spawn').train()

    def test_torch_nano_subprocess(self):
        if False:
            while True:
                i = 10
        MyNano(use_ipex=True, num_processes=2, distributed_backend='subprocess').train()

    def test_torch_nano_correctness(self):
        if False:
            i = 10
            return i + 15
        MyNanoCorrectness(use_ipex=True).train(0.25)

    def test_torch_nano_spawn_correctness(self):
        if False:
            while True:
                i = 10
        MyNanoCorrectness(use_ipex=True, num_processes=2, distributed_backend='spawn', auto_lr=False).train(0.5)

    def test_torch_nano_subprocess_correctness(self):
        if False:
            i = 10
            return i + 15
        MyNanoCorrectness(use_ipex=True, num_processes=2, distributed_backend='subprocess', auto_lr=False).train(0.5)

    def test_torch_nano_bf16_support_opt(self):
        if False:
            while True:
                i = 10
        MyNano(use_ipex=True, precision='bf16').train(optimizer_supported=True)

    def test_torch_nano_bf16_unsupport_opt(self):
        if False:
            return 10
        MyNano(use_ipex=True, precision='bf16').train()

    def test_torch_nano_bf16_spawn(self):
        if False:
            i = 10
            return i + 15
        MyNano(use_ipex=True, precision='bf16', num_processes=2, distributed_backend='spawn').train()

    def test_torch_nano_bf16_subprocess(self):
        if False:
            print('Hello World!')
        MyNano(use_ipex=True, precision='bf16', num_processes=2, distributed_backend='subprocess').train()

    def test_torch_nano_load_state_dict(self):
        if False:
            for i in range(10):
                print('nop')
        from bigdl.nano.utils.common import compare_version
        import operator
        if compare_version('intel_extension_for_pytorch', operator.lt, '1.13'):
            pass
        else:
            MyNanoLoadStateDict(use_ipex=True).train(0.25)
TORCH_CLS = Lite

class TestLite(TORCH_CLS, TestCase):
    pass
if __name__ == '__main__':
    pytest.main([__file__])