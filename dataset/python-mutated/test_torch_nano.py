import os
import math
import platform
import pytest
from unittest import TestCase
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from bigdl.nano.pytorch import TorchNano
from bigdl.nano.pytorch.vision.models import vision
batch_size = 256
num_workers = 0
data_dir = '/tmp/data'

class ResNet18(nn.Module):

    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        if False:
            i = 10
            return i + 15
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = nn.Sequential(backbone, head)

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.model(x)

    def do_nothing(self):
        if False:
            print('Hello World!')
        pass

class MyNano(TorchNano):

    def train(self):
        if False:
            i = 10
            return i + 15
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss_func = nn.CrossEntropyLoss()
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
            while True:
                i = 10
        super().__init__()
        self.fc1 = nn.Linear(1, 1, bias=False)
        self.fc1.weight.data.fill_(1.0)

    def forward(self, input_):
        if False:
            for i in range(10):
                print('nop')
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

class MyNanoAccess(TorchNano):

    def train(self):
        if False:
            while True:
                i = 10
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        train_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)
        (model, optimizer, train_loader) = self.setup(model, optimizer, train_loader)
        model.do_nothing()

class MyNanoMultiOptimizer(TorchNano):

    def train(self):
        if False:
            i = 10
            return i + 15
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss_func = nn.CrossEntropyLoss()
        optimizers = [torch.optim.Adam(model.parameters(), lr=0.005), torch.optim.Adam(model.parameters(), lr=0.01)]
        train_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)
        (model, optimizers, train_loader) = self.setup(model, optimizers, train_loader)
        model.train()
        num_epochs = 1
        for _i in range(num_epochs):
            (total_loss, num) = (0, 0)
            for (X, y) in train_loader:
                for optimizer in optimizers:
                    optimizer.zero_grad()
                loss = loss_func(model(X), y)
                self.backward(loss)
                for optimizer in optimizers:
                    optimizer.step()
                total_loss += loss.sum()
                num += 1
            print(f'avg_loss: {total_loss / num}')

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
                return 10
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

class MyNanoAutoLRCorrectness(TorchNano):

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
        expect_weight = torch.tensor([[1.0]])
        cur_lr_ratio = 1.0
        max_lr_ratio = self.num_processes
        cur_step = 0
        max_step = optimizer.max_step
        num_epochs = max_step + 10
        for _i in range(num_epochs):
            for (X, y) in train_loader:
                optimizer.zero_grad()
                loss = loss_func(model(X), y)
                self.backward(loss)
                optimizer.step()
                expect_weight = expect_weight - expect_weight * lr * cur_lr_ratio
                expect = expect_weight.item()
                real = model.fc1.weight.data.item()
                assert math.isclose(expect, real, rel_tol=1e-05), f'step: {_i}, expect: {expect}, real: {real}'
                if cur_step < max_step:
                    cur_step += 1
                    cur_lr_ratio = (max_lr_ratio - 1.0) * cur_step / max_step + 1

class MyNanoCUDA(TorchNano):

    def train(self):
        if False:
            i = 10
            return i + 15
        t = torch.tensor([0], device='cuda:0')
        assert t.device.type == 'cpu'

class TestLite(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        test_dir = os.path.dirname(__file__)
        project_test_dir = os.path.abspath(os.path.join(test_dir, '..', '..', '..', '..', '..'))
        os.environ['PYTHONPATH'] = project_test_dir

    def test_torch_nano(self):
        if False:
            while True:
                i = 10
        MyNano().train()

    def test_torch_nano_spawn(self):
        if False:
            print('Hello World!')
        MyNano(num_processes=2, distributed_backend='spawn').train()

    def test_torch_nano_subprocess(self):
        if False:
            print('Hello World!')
        MyNano(num_processes=2, distributed_backend='subprocess').train()

    def test_torch_nano_specify_cpu_cores(self):
        if False:
            print('Hello World!')
        MyNano(num_processes=2, cpu_for_each_process=[[0, 1], [2, 3]]).train()

    def test_torch_nano_correctness(self):
        if False:
            i = 10
            return i + 15
        MyNanoCorrectness(auto_lr=False).train(0.25)

    def test_torch_nano_spawn_correctness(self):
        if False:
            i = 10
            return i + 15
        MyNanoCorrectness(num_processes=2, distributed_backend='spawn', auto_lr=False).train(0.5)

    def test_torch_nano_subprocess_correctness(self):
        if False:
            while True:
                i = 10
        MyNanoCorrectness(num_processes=2, distributed_backend='subprocess', auto_lr=False).train(0.5)

    @pytest.mark.skipif(platform.system() != 'Linux', reason='torch_ccl is only avaiable on Linux')
    def test_torch_nano_spawn_with_ccl(self):
        if False:
            print('Hello World!')
        MyNano(num_processes=2, distributed_backend='spawn', process_group_backend='ccl').train()

    @pytest.mark.skipif(platform.system() != 'Linux', reason='torch_ccl is only avaiable on Linux')
    def test_torch_nano_subprocess_with_ccl(self):
        if False:
            return 10
        MyNano(num_processes=2, distributed_backend='subprocess', process_group_backend='ccl').train()

    def test_torch_nano_attribute_access(self):
        if False:
            for i in range(10):
                print('nop')
        MyNanoAccess().train()

    def test_torch_nano_attribute_access_ddp(self):
        if False:
            while True:
                i = 10
        MyNanoAccess(num_processes=2).train()

    def test_torch_nano_multi_optimizer(self):
        if False:
            i = 10
            return i + 15
        MyNanoMultiOptimizer().train()

    def test_torch_nano_load_state_dict(self):
        if False:
            while True:
                i = 10
        MyNanoLoadStateDict().train(0.25)

    def test_torch_nano_load_state_dict_ddp(self):
        if False:
            i = 10
            return i + 15
        MyNanoLoadStateDict(num_processes=2).train(0.5)

    def test_torch_nano_auto_lr(self):
        if False:
            while True:
                i = 10
        MyNanoAutoLRCorrectness(num_processes=2, distributed_backend='spawn', auto_lr=True).train(0.01)

    def test_torch_nano_cuda_patch_spawn(self):
        if False:
            for i in range(10):
                print('nop')
        from bigdl.nano.pytorch import patch_torch, unpatch_torch
        patch_torch(cuda_to_cpu=True)
        MyNanoCUDA(num_processes=2, distributed_backend='spawn').train()
        unpatch_torch()

    def test_torch_nano_cuda_patch_subprocess(self):
        if False:
            print('Hello World!')
        from bigdl.nano.pytorch import patch_torch, unpatch_torch
        patch_torch(cuda_to_cpu=True)
        MyNanoCUDA(num_processes=2, distributed_backend='subprocess').train()
        unpatch_torch()
if __name__ == '__main__':
    pytest.main([__file__])