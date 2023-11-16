import os
from unittest import TestCase
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.plugins.precision.double import DoublePrecisionPlugin
from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch.vision.models import vision
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_1_10
from test.pytorch.utils._train_torch_lightning import create_data_loader, create_test_data_loader, data_transform
batch_size = 32
dataset_size = 256
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
            return 10
        return self.model(x)

class TestTrainer(TestCase):
    train_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform, dataset_size)
    test_loader = create_test_data_loader(data_dir, batch_size, num_workers, data_transform, dataset_size)

    def test_trainer_precision(self):
        if False:
            for i in range(10):
                print('nop')
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        pl_model = Trainer.compile(model, loss, optimizer)
        if TORCH_VERSION_LESS_1_10:
            trainer = Trainer(max_epochs=2, precision=64)
            trainer.fit(pl_model, self.train_loader)
            assert isinstance(trainer.strategy.precision_plugin, DoublePrecisionPlugin)
            opt = pl_model.optimizers()
            assert opt.param_groups[0]['params'][0].dtype is torch.float64
        else:
            trainer = Trainer(max_epochs=2, precision='bf16')
            trainer.fit(pl_model, self.train_loader)
            assert isinstance(trainer.strategy.precision_plugin, NativeMixedPrecisionPlugin)
            input = TensorDataset(torch.rand(1, 3, 32, 32))
            train_loader = DataLoader(input)
            y_hat = trainer.predict(pl_model, train_loader)
if __name__ == '__main__':
    pytest.main([__file__])