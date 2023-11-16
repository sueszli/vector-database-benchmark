import os
import platform
from unittest import TestCase
import pytest
import torch
from pytorch_lightning import LightningModule
from test.pytorch.utils._train_torch_lightning import train_with_linear_top_layer
from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from bigdl.nano.utils.common import compare_version
import operator
from torch import nn
from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch.vision.models import vision
batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), '../data')

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

class LitResNet18(LightningModule):

    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.classify = nn.Sequential(backbone, head)

    def forward(self, *args):
        if False:
            return 10
        return self.classify(args[0])

class TestTrainer(TestCase):
    model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)
    user_defined_pl_model = LitResNet18(10)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        test_dir = os.path.dirname(__file__)
        project_test_dir = os.path.abspath(os.path.join(os.path.join(test_dir, '..'), '..'))
        os.environ['PYTHONPATH'] = project_test_dir

    def test_resnet18(self):
        if False:
            return 10
        resnet18 = vision.resnet18(pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(resnet18, batch_size, num_workers, data_dir)

    @pytest.mark.skipif(compare_version('torch', operator.ge, '2.0.0') and compare_version('pytorch_lightning', operator.lt, '2.0.0'), reason='We have not upgraded version of pytorch_lightning.')
    def test_trainer_ray_compile(self):
        if False:
            for i in range(10):
                print('nop')
        trainer = Trainer(max_epochs=1, num_processes=2, distributed_backend='ray')
        pl_model = Trainer.compile(self.model, self.loss, self.optimizer)
        trainer.fit(pl_model, self.train_loader)

    @pytest.mark.skipif(platform.system() != 'Linux', reason='torch_ccl is only avaiable on Linux')
    @pytest.mark.skipif(compare_version('torch', operator.ge, '2.0.0') and compare_version('pytorch_lightning', operator.lt, '2.0.0'), reason='We have not upgraded version of pytorch_lightning.')
    def test_trainer_ray_with_ccl(self):
        if False:
            i = 10
            return i + 15
        trainer = Trainer(max_epochs=1, num_processes=2, distributed_backend='ray', process_group_backend='ccl')
        pl_model = Trainer.compile(self.model, self.loss, self.optimizer)
        trainer.fit(pl_model, self.train_loader)
if __name__ == '__main__':
    pytest.main([__file__])