import os
import pytest
from unittest import TestCase
import torch
import torchmetrics
from torch import nn
from bigdl.nano.pytorch.lightning import LightningModule
from bigdl.nano.pytorch import Trainer
from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from test.pytorch.utils._train_torch_lightning import create_test_data_loader
from test.pytorch.utils._train_ipex_callback import CheckIPEXCallback, CheckIPEXFusedStepCallback
from test.pytorch.tests.train.trainer.test_lightning import ResNet18
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_2_0
from bigdl.nano.utils.common import _avx2_checker
num_classes = 10
batch_size = 32
dataset_size = 256
num_workers = 0
data_dir = '/tmp/data'

class Plugin:
    model = ResNet18(pretrained=False, include_top=False, freeze=True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform, subset=dataset_size)
    test_data_loader = create_test_data_loader(data_dir, batch_size, num_workers, data_transform, subset=dataset_size)

    def setUp(self):
        if False:
            return 10
        test_dir = os.path.dirname(__file__)
        project_test_dir = os.path.abspath(os.path.join(test_dir, '..', '..', '..', '..', '..'))
        os.environ['PYTHONPATH'] = project_test_dir

    def test_trainer_subprocess_plugin(self):
        if False:
            while True:
                i = 10
        pl_model = LightningModule(self.model, self.loss, self.optimizer, metrics=[torchmetrics.F1Score('multiclass', num_classes=num_classes), torchmetrics.Accuracy('multiclass', num_classes=num_classes)])
        trainer = Trainer(num_processes=2, distributed_backend='subprocess', max_epochs=4, use_ipex=True, callbacks=[CheckIPEXCallback()])
        trainer.fit(pl_model, self.data_loader, self.test_data_loader)
        trainer.test(pl_model, self.test_data_loader)

    def test_trainer_spawn_plugin_bf16(self):
        if False:
            return 10
        model = ResNet18(pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        pl_model = LightningModule(model, loss, optimizer, metrics=[torchmetrics.F1Score('multiclass', num_classes=num_classes), torchmetrics.Accuracy('multiclass', num_classes=num_classes)])
        trainer = Trainer(num_processes=2, distributed_backend='spawn', max_epochs=4, use_ipex=True, precision='bf16', callbacks=[CheckIPEXCallback(), CheckIPEXFusedStepCallback()])
        trainer.fit(pl_model, self.data_loader, self.test_data_loader)
        trainer.test(pl_model, self.test_data_loader)
TORCH_CLS = Plugin

class CasePT2:

    def test_placeholder(self):
        if False:
            return 10
        pass
if not TORCH_VERSION_LESS_2_0:
    print('Trainer Plugin with torch 2.0')
    TORCH_CLS = CasePT2

class TestPlugin(TORCH_CLS, TestCase):
    pass
if __name__ == '__main__':
    pytest.main([__file__])