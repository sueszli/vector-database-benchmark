import os
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule, seed_everything
from bigdl.nano.pytorch import Trainer, InferenceOptimizer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
seed_everything(7)
PATH_DATASETS = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 64
NUM_WORKERS = int(os.cpu_count() / 2)
train_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), cifar10_normalization()])
test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), cifar10_normalization()])
cifar10_dm = CIFAR10DataModule(data_dir=PATH_DATASETS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, train_transforms=train_transforms, test_transforms=test_transforms, val_transforms=test_transforms, pin_memory=False)

def create_model():
    if False:
        for i in range(10):
            print('nop')
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

class LitResnet(LightningModule):

    def __init__(self, lr=0.05):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        if False:
            print('Hello World!')
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
        acc = accuracy(preds, y, 'multiclass', num_classes=10)
        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        if False:
            return 10
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        if False:
            print('Hello World!')
        self.evaluate(batch, 'test')

    def configure_optimizers(self):
        if False:
            for i in range(10):
                print('nop')
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=0.0005)
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {'scheduler': OneCycleLR(optimizer, 0.1, epochs=self.trainer.max_epochs, steps_per_epoch=steps_per_epoch), 'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}
model = LitResnet(lr=0.05)
model.datamodule = cifar10_dm
trainer = Trainer(progress_bar_refresh_rate=10, max_epochs=30, logger=TensorBoardLogger('lightning_logs/', name='resnet'), callbacks=[LearningRateMonitor(logging_interval='step')])
trainer.fit(model, cifar10_dm)
start = time()
outputs = trainer.test(model, datamodule=cifar10_dm)
fp32_infer_time = time() - start
fp32_acc = outputs[0]['test_acc'] * 100
i8_model = InferenceOptimizer.quantize(model, calib_dataloader=cifar10_dm.train_dataloader())
with InferenceOptimizer.get_context(i8_model):
    start = time()
    outputs = trainer.test(i8_model, cifar10_dm)
    i8_infer_time = time() - start
    i8_acc = outputs[0]['test_acc'] * 100
summary = '\n|    Precision   | Inference Time(s) | Model Size(MB) | Accuracy(%) |\n|      FP32      |       {:5.2f}       |      {:5.2f}     |    {:5.2f}    |\n|      INT8      |       {:5.2f}       |      {:5.2f}     |    {:5.2f}    |\n| Improvement(%) |       {:5.2f}       |      {:5.2f}     |    {:5.2f}    |\n'
summary = summary.format(fp32_infer_time, model.model_size, fp32_acc, i8_infer_time, i8_model.quantized_model_size, i8_acc, (1 - i8_infer_time / fp32_infer_time) * 100, (1 - i8_model.quantized_model_size / model.model_size) * 100, i8_acc - fp32_acc)
print(summary)