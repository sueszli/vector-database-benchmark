"""Computer vision example on Transfer Learning. This computer vision example illustrates how one could fine-tune a
pre-trained network (by default, a ResNet50 is used) using pytorch-lightning. For the sake of this example, the 'cats
and dogs dataset' (~60MB, see `DATA_URL` below) and the proposed network (denoted by `TransferLearningModel`, see
below) is trained for 15 epochs.

The training consists of three stages.

From epoch 0 to 4, the feature extractor (the pre-trained network) is frozen except
maybe for the BatchNorm layers (depending on whether `train_bn = True`). The BatchNorm
layers (if `train_bn = True`) and the parameters of the classifier are trained as a
single parameters group with lr = 1e-2.

From epoch 5 to 9, the last two layer groups of the pre-trained network are unfrozen
and added to the optimizer as a new parameter group with lr = 1e-4 (while lr = 1e-3
for the first parameter group in the optimizer).

Eventually, from epoch 10, all the remaining layer groups of the pre-trained network
are unfrozen and added to the optimizer as a third parameter group. From epoch 10,
the parameters of the pre-trained network are trained with lr = 1e-5 while those of
the classifier is trained with lr = 1e-4.

Note:
    See: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

To run:
    python computer_vision_fine_tuning.py fit

"""
import logging
from pathlib import Path
from typing import Union
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo
from lightning.pytorch.callbacks.finetuning import BaseFinetuning
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.utilities.model_helpers import get_torchvision_model
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
log = logging.getLogger(__name__)
DATA_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

class MilestonesFinetuning(BaseFinetuning):

    def __init__(self, milestones: tuple=(5, 10), train_bn: bool=False):
        if False:
            while True:
                i = 10
        super().__init__()
        self.milestones = milestones
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: LightningModule):
        if False:
            return 10
        self.freeze(modules=pl_module.feature_extractor, train_bn=self.train_bn)

    def finetune_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer):
        if False:
            return 10
        if epoch == self.milestones[0]:
            self.unfreeze_and_add_param_group(modules=pl_module.feature_extractor[-5:], optimizer=optimizer, train_bn=self.train_bn)
        elif epoch == self.milestones[1]:
            self.unfreeze_and_add_param_group(modules=pl_module.feature_extractor[:-5], optimizer=optimizer, train_bn=self.train_bn)

class CatDogImageDataModule(LightningDataModule):

    def __init__(self, dl_path: Union[str, Path]='data', num_workers: int=0, batch_size: int=8):
        if False:
            while True:
                i = 10
        'CatDogImageDataModule.\n\n        Args:\n            dl_path: root directory where to download the data\n            num_workers: number of CPU workers\n            batch_size: number of sample in a batch\n\n        '
        super().__init__()
        self._dl_path = dl_path
        self._num_workers = num_workers
        self._batch_size = batch_size

    def prepare_data(self):
        if False:
            i = 10
            return i + 15
        'Download images and prepare images datasets.'
        download_and_extract_archive(url=DATA_URL, download_root=self._dl_path, remove_finished=True)

    @property
    def data_path(self):
        if False:
            for i in range(10):
                print('nop')
        return Path(self._dl_path).joinpath('cats_and_dogs_filtered')

    @property
    def normalize_transform(self):
        if False:
            while True:
                i = 10
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        if False:
            i = 10
            return i + 15
        return transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), self.normalize_transform])

    @property
    def valid_transform(self):
        if False:
            print('Hello World!')
        return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), self.normalize_transform])

    def create_dataset(self, root, transform):
        if False:
            while True:
                i = 10
        return ImageFolder(root=root, transform=transform)

    def __dataloader(self, train: bool):
        if False:
            while True:
                i = 10
        'Train/validation loaders.'
        if train:
            dataset = self.create_dataset(self.data_path.joinpath('train'), self.train_transform)
        else:
            dataset = self.create_dataset(self.data_path.joinpath('validation'), self.valid_transform)
        return DataLoader(dataset=dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=train)

    def train_dataloader(self):
        if False:
            return 10
        log.info('Training data loaded.')
        return self.__dataloader(train=True)

    def val_dataloader(self):
        if False:
            return 10
        log.info('Validation data loaded.')
        return self.__dataloader(train=False)

class TransferLearningModel(LightningModule):

    def __init__(self, backbone: str='resnet50', train_bn: bool=False, milestones: tuple=(2, 4), batch_size: int=32, lr: float=0.001, lr_scheduler_gamma: float=0.1, num_workers: int=6, **kwargs) -> None:
        if False:
            return 10
        'TransferLearningModel.\n\n        Args:\n            backbone: Name (as in ``torchvision.models``) of the feature extractor\n            train_bn: Whether the BatchNorm layers should be trainable\n            milestones: List of two epochs milestones\n            lr: Initial learning rate\n            lr_scheduler_gamma: Factor by which the learning rate is reduced at each milestone\n\n        '
        super().__init__()
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers
        self.__build_model()
        self.train_acc = Accuracy(task='binary')
        self.valid_acc = Accuracy(task='binary')
        self.save_hyperparameters()

    def __build_model(self):
        if False:
            for i in range(10):
                print('nop')
        'Define model layers & loss.'
        backbone = get_torchvision_model(self.backbone, weights='DEFAULT')
        _layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)
        _fc_layers = [nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 32), nn.Linear(32, 1)]
        self.fc = nn.Sequential(*_fc_layers)
        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x):
        if False:
            print('Hello World!')
        'Forward pass.\n\n        Returns logits.\n\n        '
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)
        return self.fc(x)

    def loss(self, logits, labels):
        if False:
            while True:
                i = 10
        return self.loss_func(input=logits, target=labels)

    def training_step(self, batch, batch_idx):
        if False:
            print('Hello World!')
        (x, y) = batch
        y_logits = self.forward(x)
        y_scores = torch.sigmoid(y_logits)
        y_true = y.view((-1, 1)).type_as(x)
        train_loss = self.loss(y_logits, y_true)
        self.log('train_acc', self.train_acc(y_scores, y_true.int()), prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        if False:
            for i in range(10):
                print('nop')
        (x, y) = batch
        y_logits = self.forward(x)
        y_scores = torch.sigmoid(y_logits)
        y_true = y.view((-1, 1)).type_as(x)
        self.log('val_loss', self.loss(y_logits, y_true), prog_bar=True)
        self.log('val_acc', self.valid_acc(y_scores, y_true.int()), prog_bar=True)

    def configure_optimizers(self):
        if False:
            for i in range(10):
                print('nop')
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(f'The model will start training with only {len(trainable_parameters)} trainable parameters out of {len(parameters)}.')
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        return ([optimizer], [scheduler])

class MyLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        if False:
            return 10
        parser.add_lightning_class_args(MilestonesFinetuning, 'finetuning')
        parser.link_arguments('data.batch_size', 'model.batch_size')
        parser.link_arguments('finetuning.milestones', 'model.milestones')
        parser.link_arguments('finetuning.train_bn', 'model.train_bn')
        parser.set_defaults({'trainer.max_epochs': 15, 'trainer.enable_model_summary': False, 'trainer.num_sanity_val_steps': 0})

def cli_main():
    if False:
        while True:
            i = 10
    MyLightningCLI(TransferLearningModel, CatDogImageDataModule, seed_everything_default=1234)
if __name__ == '__main__':
    cli_lightning_logo()
    cli_main()