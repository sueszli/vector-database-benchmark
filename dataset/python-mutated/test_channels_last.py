import os
import pytest
import torch
from unittest import TestCase
from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch import TorchNano
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_1_12, TORCH_VERSION_LESS_2_0
from torchvision.models.resnet import ResNet, BasicBlock
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
import torch.nn.functional as F
from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from test.pytorch.utils._train_torch_lightning import create_test_data_loader
num_classes = 10
batch_size = 32
dataset_size = 256
num_workers = 0
data_dir = '/tmp/data'

class CustomResNet(pl.LightningModule):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2])
        self.head = torch.nn.Linear(self.model.fc.out_features, num_classes)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if not TORCH_VERSION_LESS_1_12:
            assert x.is_contiguous(memory_format=torch.channels_last)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        if False:
            return 10
        (x, y) = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        if False:
            print('Hello World!')
        (x, y) = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, 'multiclass', num_classes=num_classes)
        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        if False:
            print('Hello World!')
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        if False:
            for i in range(10):
                print('nop')
        self.evaluate(batch, 'test')

    def configure_optimizers(self):
        if False:
            return 10
        return torch.optim.Adam(params=self.parameters(), lr=0.05)

class ConvModel(torch.nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 1, (1, 2), bias=False)
        self.conv1.weight.data.fill_(1.0)

    def forward(self, input):
        if False:
            return 10
        x = self.conv1(input)
        if not TORCH_VERSION_LESS_1_12:
            assert x.is_contiguous(memory_format=torch.channels_last)
        output = torch.flatten(x, 1)
        return output

class MyNano(TorchNano):

    def train(self):
        if False:
            return 10
        model = CustomResNet()
        loss_func = torch.nn.CrossEntropyLoss()
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

class MyNanoChannelsLastCorrectness(TorchNano):

    def train(self):
        if False:
            i = 10
            return i + 15
        x = torch.Tensor([[[[1, 0]], [[1, 0]]], [[[1, 0]], [[2, 0]]], [[[0, 3]], [[1, 0]]], [[[1, 1]], [[2, 1]]]])
        y = torch.Tensor([[0.0], [1.0], [0.0], [1.0]])
        train_dataset = torch.utils.data.TensorDataset(x, y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False)
        origin_model = ConvModel()
        loss_fuc = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(origin_model.parameters(), lr=0.25)
        (model, optimizer, train_loader) = self.setup(origin_model, optimizer, train_loader)
        model.train()
        for (X, y) in train_loader:
            optimizer.zero_grad()
            loss = loss_fuc(model(X), y)
            self.backward(loss)
            optimizer.step()
        result = torch.tensor([[[[0.0, -1.0]], [[-1.25, 0.5]]]])
        assert origin_model.conv1.weight.equal(result)

class ChannelsLast:
    data_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform, subset=dataset_size)
    test_data_loader = create_test_data_loader(data_dir, batch_size, num_workers, data_transform, subset=dataset_size)

    def setUp(self):
        if False:
            return 10
        test_dir = os.path.dirname(__file__)
        project_test_dir = os.path.abspath(os.path.join(os.path.join(os.path.join(test_dir, '..'), '..'), '..'))
        os.environ['PYTHONPATH'] = project_test_dir

    def test_trainer_lightning_channels_last(self):
        if False:
            for i in range(10):
                print('nop')
        model = CustomResNet()
        trainer = Trainer(max_epochs=1, channels_last=True)
        trainer.fit(model, self.data_loader, self.test_data_loader)
        trainer.test(model, self.test_data_loader)

    def test_trainer_channels_last_correctness(self):
        if False:
            print('Hello World!')
        model = ConvModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        loss = torch.nn.MSELoss()
        pl_module = Trainer.compile(model=model, loss=loss, optimizer=optimizer)
        trainer = Trainer(max_epochs=1, channels_last=True)
        x = torch.Tensor([[[[1, 0]], [[1, 0]]], [[[1, 0]], [[2, 0]]], [[[0, 3]], [[1, 0]]], [[[1, 1]], [[2, 1]]]])
        y = torch.Tensor([[0.0], [1.0], [0.0], [1.0]])
        dataset = torch.utils.data.TensorDataset(x, y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        trainer.fit(pl_module, data_loader)
        result = torch.tensor([[[[0.0, -1.0]], [[-1.25, 0.5]]]])
        assert pl_module.model.conv1.weight.equal(result)

    def test_trainer_lightning_channels_last_subprocess(self):
        if False:
            print('Hello World!')
        model = CustomResNet()
        trainer = Trainer(max_epochs=1, num_processes=2, distributed_backend='subprocess', channels_last=True)
        trainer.fit(model, self.data_loader, self.test_data_loader)
        trainer.test(model, self.test_data_loader)

    def test_trainer_channels_last_correctness_subprocess(self):
        if False:
            return 10
        model = ConvModel()
        model.conv1 = torch.nn.Conv2d(2, 1, (1, 2), bias=False)
        model.conv1.weight.data.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        loss = torch.nn.MSELoss()
        pl_module = Trainer.compile(model=model, loss=loss, optimizer=optimizer)
        trainer = Trainer(max_epochs=1, channels_last=True, distributed_backend='subprocess', num_processes=2)
        x = torch.Tensor([[[[1, 0]], [[1, 0]]], [[[1, 0]], [[2, 0]]], [[[0, 3]], [[1, 0]]], [[[1, 1]], [[2, 1]]]])
        y = torch.Tensor([[0], [1], [0], [1]])
        dataset = torch.utils.data.TensorDataset(x, y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        trainer.fit(pl_module, data_loader)
        result = torch.tensor([[[[0.0, -1.0]], [[-1.25, 0.5]]]])
        assert pl_module.model.conv1.weight.equal(result)

    def test_torch_nano_channels_last(self):
        if False:
            print('Hello World!')
        MyNano(channels_last=True).train()

    def test_torch_nano_channels_last_subprocess(self):
        if False:
            i = 10
            return i + 15
        MyNano(num_processes=2, strategy='subprocess', channels_last=True).train()

    def test_torch_nano_channels_last_correctness(self):
        if False:
            i = 10
            return i + 15
        MyNanoChannelsLastCorrectness(channels_last=True).train()

    def test_torch_nano_channels_last_subprocess_correctness(self):
        if False:
            return 10
        MyNanoChannelsLastCorrectness(num_processes=2, strategy='subprocess', channels_last=True).train()

class ChannelsLastSpawn:
    data_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform, subset=dataset_size)
    test_data_loader = create_test_data_loader(data_dir, batch_size, num_workers, data_transform, subset=dataset_size)

    def test_lightning_channels_last_spawn(self):
        if False:
            return 10
        model = CustomResNet()
        trainer = Trainer(max_epochs=1, num_processes=2, distributed_backend='spawn', channels_last=True)
        trainer.fit(model, self.data_loader, self.test_data_loader)
        trainer.test(model, self.test_data_loader)

    def test_trainer_channels_last_correctness_spawn(self):
        if False:
            while True:
                i = 10
        model = ConvModel()
        model.conv1 = torch.nn.Conv2d(2, 1, (1, 2), bias=False)
        model.conv1.weight.data.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        loss = torch.nn.MSELoss()
        pl_module = Trainer.compile(model=model, loss=loss, optimizer=optimizer)
        trainer = Trainer(max_epochs=1, channels_last=True, distributed_backend='spawn', num_processes=2)
        x = torch.Tensor([[[[1, 0]], [[1, 0]]], [[[1, 0]], [[2, 0]]], [[[0, 3]], [[1, 0]]], [[[1, 1]], [[2, 1]]]])
        y = torch.Tensor([[0], [1], [0], [1]])
        dataset = torch.utils.data.TensorDataset(x, y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        trainer.fit(pl_module, data_loader)
        result = torch.tensor([[[[0.0, -1.0]], [[-1.25, 0.5]]]])
        assert pl_module.model.conv1.weight.equal(result)

    def test_torch_nano_channels_last_spawn(self):
        if False:
            while True:
                i = 10
        MyNano(num_processes=2, strategy='spawn', channels_last=True).train()

    def test_torch_nano_channels_last_spawn_correctness(self):
        if False:
            while True:
                i = 10
        MyNanoChannelsLastCorrectness(num_processes=2, strategy='spawn', channels_last=True).train()
TORCH_CLS = ChannelsLast
TORCH_CLS_Spawn = ChannelsLastSpawn

class CaseWithoutscheduler:

    def test_placeholder(self):
        if False:
            print('Hello World!')
        pass
if not TORCH_VERSION_LESS_2_0:
    print('channels last for torch >= 2.0')
    TORCH_CLS = CaseWithoutscheduler
    TORCH_CLS_Spawn = CaseWithoutscheduler

class TestChannelsLast(TORCH_CLS, TestCase):
    pass

class TestChannelsLastSpawn(TORCH_CLS_Spawn, TestCase):
    pass
if __name__ == '__main__':
    pytest.main([__file__])