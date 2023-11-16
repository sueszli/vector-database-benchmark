#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is adapted from PyTorch Lightning.
# https://github.com/Lightning-AI/lightning/blob/master/examples/
# pl_domain_templates/computer_vision_fine_tuning.py
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Computer vision example on Transfer Learning. This computer vision example illustrates how one could fine-tune a
pre-trained network (by default, a ResNet50 is used) using pytorch-lightning. For the sake of this example, the
'cats and dogs dataset' (~60MB, see `DATA_URL` below) and the proposed network (denoted by `TransferLearningModel`,
see below) is trained for 15 epochs.

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
"""


import logging
from pathlib import Path
from typing import Union
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassAccuracy
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities.rank_zero import rank_zero_info


log = logging.getLogger(__name__)
DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"


class TransferLearningModel(LightningModule):
    def __init__(
        self,
        backbone: str = "resnet18",
        milestones: tuple = (5, 10),
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        **kwargs,
    ) -> None:
        """TransferLearningModel.

        Args:
            backbone: Name (as in ``torchvision.models``) of the feature extractor
            train_bn: Whether the BatchNorm layers should be trainable
            milestones: List of two epochs milestones
            lr: Initial learning rate
            lr_scheduler_gamma: Factor by which the learning rate is reduced at each milestone
        """
        super().__init__()
        self.backbone = backbone
        self.milestones = milestones
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers

        self.__build_model()

        self.train_acc = MulticlassAccuracy(num_classes=2)
        self.valid_acc = MulticlassAccuracy(num_classes=2)
        self.save_hyperparameters()

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True)

        _layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)

        # 2. Classifier:
        _fc_layers = [nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 32), nn.Linear(32, 1)]
        self.fc = nn.Sequential(*_fc_layers)

        # 3. Loss:
        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x):
        """Forward pass.

        Returns logits.
        """

        # 1. Feature extraction:
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)

        # 2. Classifier (returns logits):
        x = self.fc(x)

        return x

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_scores = torch.sigmoid(y_logits)
        y_true = y.view((-1, 1)).type_as(x)

        # 2. Compute loss
        train_loss = self.loss(y_logits, y_true)

        # 3. Compute accuracy:
        self.log("train_acc", self.train_acc(y_scores, y_true.int()), prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_scores = torch.sigmoid(y_logits)
        y_true = y.view((-1, 1)).type_as(x)

        # 2. Compute loss
        self.log("val_loss", self.loss(y_logits, y_true), prog_bar=True)

        # 3. Compute accuracy:
        self.log("val_acc", self.valid_acc(y_scores, y_true.int()), prog_bar=True)

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        return [optimizer], [scheduler]


class CatDogImageDataModule(LightningDataModule):
    def __init__(self, dl_path: Union[str, Path] = "data", num_workers: int = 0, batch_size: int = 8):
        """CatDogImageDataModule.

        Args:
            dl_path: root directory where to download the data
            num_workers: number of CPU workers
            batch_size: number of sample in a batch
        """
        super().__init__()

        self._dl_path = dl_path
        self._num_workers = num_workers
        self._batch_size = batch_size

    def prepare_data(self):
        """Download images and prepare images datasets."""
        download_and_extract_archive(url=DATA_URL, download_root=self._dl_path,
                                     remove_finished=True)

    @property
    def data_path(self):
        return Path(self._dl_path).joinpath("cats_and_dogs_filtered")

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    @property
    def valid_transform(self):
        return transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(), self.normalize_transform])

    def create_dataset(self, root, transform):
        return ImageFolder(root=root, transform=transform)

    def __dataloader(self, train: bool, batch_size=None, limit_num_samples=None):
        """Train/validation loaders."""
        if batch_size is None:
            batch_size = self._batch_size
        if train:
            dataset = self.create_dataset(self.data_path.joinpath("train"),
                                          self.train_transform)
            return DataLoader(dataset=dataset, batch_size=batch_size, 
                              num_workers=self._num_workers, shuffle=True)
        else:
            dataset = self.create_dataset(self.data_path.joinpath("validation"),
                                          self.valid_transform)
            if limit_num_samples is not None:
                indices = np.random.permutation(len(dataset))[:limit_num_samples]
                dataset = Subset(dataset, indices)
            return DataLoader(dataset=dataset, batch_size=batch_size,
                              num_workers=self._num_workers, shuffle=False)

    def train_dataloader(self, batch_size=None):
        log.info("Training data loaded.")
        return self.__dataloader(train=True, batch_size=batch_size)

    def val_dataloader(self, batch_size=None, limit_num_samples=None):
        log.info("Validation data loaded.")
        return self.__dataloader(train=False, batch_size=batch_size,
                                 limit_num_samples=limit_num_samples)


class MilestonesFinetuning(BaseFinetuning):
    def __init__(self, milestones: tuple = (5, 10), train_bn: bool = False):
        super().__init__()
        self.milestones = milestones
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: LightningModule):
        self.freeze(modules=pl_module.feature_extractor, train_bn=self.train_bn)

    def finetune_function(self, pl_module: LightningModule, epoch: int, 
                          optimizer: Optimizer, opt_idx: int):
        if epoch == self.milestones[0]:
            # unfreeze 5 last layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[-5:],  # type: ignore
                optimizer=optimizer, train_bn=self.train_bn
            )

        elif epoch == self.milestones[1]:
            # unfreeze remaining layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[:-5],  # type: ignore
                optimizer=optimizer, train_bn=self.train_bn
            )
