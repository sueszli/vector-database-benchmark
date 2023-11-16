import copy
import os
import shutil
from unittest import TestCase
import pytest
import torch
from pytorch_lightning import LightningModule
from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from test.pytorch.utils._train_torch_lightning import train_with_linear_top_layer
from torch import nn
from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch.vision.models import vision
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
            return 10
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
            print('Hello World!')
        return self.classify(args[0])

class TestTrainer(TestCase):
    model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)
    user_defined_pl_model = LitResNet18(10)

    def test_resnet18(self):
        if False:
            while True:
                i = 10
        resnet18 = vision.resnet18(pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(resnet18, batch_size, num_workers, data_dir)

    def test_trainer_compile(self):
        if False:
            print('Hello World!')
        trainer = Trainer(max_epochs=1)
        pl_model = Trainer.compile(self.model, self.loss, self.optimizer)
        trainer.fit(pl_model, self.train_loader)

    def test_trainer_save_load(self):
        if False:
            for i in range(10):
                print('nop')
        trainer = Trainer(max_epochs=1)
        pl_model = Trainer.compile(self.model, self.loss, self.optimizer)
        trainer.save(pl_model, 'saved_model')
        assert len(os.listdir('saved_model')) > 0
        original_state_dict = copy.deepcopy(pl_model.state_dict())
        trainer.fit(pl_model, self.train_loader)
        loaded_pl_model = trainer.load('saved_model', pl_model)
        loaded_state_dict = loaded_pl_model.state_dict()
        for k in original_state_dict.keys():
            assert (original_state_dict[k] == loaded_state_dict[k]).all()
        shutil.rmtree('saved_model')
if __name__ == '__main__':
    pytest.main([__file__])