"""
Searching in DARTS search space
===============================

In this tutorial, we demonstrate how to search in the famous model space proposed in `DARTS`_.

Through this process, you will learn:

* How to use the built-in model spaces from NNI's model space hub.
* How to use one-shot exploration strategies to explore a model space.
* How to customize evaluators to achieve the best performance.

In the end, we get a strong-performing model on CIFAR-10 dataset, which achieves up to 97.28% accuracy.

.. attention::

   Running this tutorial requires a GPU.
   If you don't have one, you can set ``gpus`` in :class:`~nni.nas.evaluator.pytorch.Classification` to be 0,
   but do note that it will be much slower.

.. _DARTS: https://arxiv.org/abs/1806.09055

Use a pre-searched DARTS model
------------------------------

Similar to `the beginner tutorial of PyTorch <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`__,
we begin with CIFAR-10 dataset, which is a image classification dataset of 10 categories.
The images in CIFAR-10 are of size 3x32x32, i.e., RGB-colored images of 32x32 pixels in size.

We first load the CIFAR-10 dataset with torchvision.
"""
import nni
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from nni.nas.evaluator.pytorch import DataLoader
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
transform_valid = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
valid_data = nni.trace(CIFAR10)(root='./data', train=False, download=True, transform=transform_valid)
valid_loader = DataLoader(valid_data, batch_size=256, num_workers=6)
from nni.nas.hub.pytorch import DARTS as DartsSpace
darts_v2_model = DartsSpace.load_searched_model('darts-v2', pretrained=True, download=True)

def evaluate_model(model, cuda=False):
    if False:
        while True:
            i = 10
    device = torch.device('cuda' if cuda else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for (inputs, targets) in valid_loader:
            (inputs, targets) = (inputs.to(device), targets.to(device))
            logits = model(inputs)
            (_, predict) = torch.max(logits, 1)
            correct += (predict == targets).sum().cpu().item()
            total += targets.size(0)
    print('Accuracy:', correct / total)
    return correct / total
evaluate_model(darts_v2_model, cuda=True)
model_space = DartsSpace(width=16, num_cells=8, dataset='cifar')
fast_dev_run = True
import numpy as np
from nni.nas.evaluator.pytorch import Classification
from torch.utils.data import SubsetRandomSampler
transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
train_data = nni.trace(CIFAR10)(root='./data', train=True, download=True, transform=transform)
num_samples = len(train_data)
indices = np.random.permutation(num_samples)
split = num_samples // 2
search_train_loader = DataLoader(train_data, batch_size=64, num_workers=6, sampler=SubsetRandomSampler(indices[:split]))
search_valid_loader = DataLoader(train_data, batch_size=64, num_workers=6, sampler=SubsetRandomSampler(indices[split:]))
evaluator = Classification(learning_rate=0.001, weight_decay=0.0001, train_dataloaders=search_train_loader, val_dataloaders=search_valid_loader, max_epochs=10, gpus=1, fast_dev_run=fast_dev_run)
from nni.nas.strategy import DARTS as DartsStrategy
strategy = DartsStrategy()
from nni.nas.experiment import NasExperiment
experiment = NasExperiment(model_space, evaluator, strategy)
experiment.run()
exported_arch = experiment.export_top_models(formatter='dict')[0]
exported_arch
import io
import graphviz
import matplotlib.pyplot as plt
from PIL import Image

def plot_single_cell(arch_dict, cell_name):
    if False:
        return 10
    g = graphviz.Digraph(node_attr=dict(style='filled', shape='rect', align='center'), format='png')
    g.body.extend(['rankdir=LR'])
    g.node('c_{k-2}', fillcolor='darkseagreen2')
    g.node('c_{k-1}', fillcolor='darkseagreen2')
    assert len(arch_dict) % 2 == 0
    for i in range(2, 6):
        g.node(str(i), fillcolor='lightblue')
    for i in range(2, 6):
        for j in range(2):
            op = arch_dict[f'{cell_name}/op_{i}_{j}']
            from_ = arch_dict[f'{cell_name}/input_{i}_{j}']
            if from_ == 0:
                u = 'c_{k-2}'
            elif from_ == 1:
                u = 'c_{k-1}'
            else:
                u = str(from_)
            v = str(i)
            g.edge(u, v, label=op, fillcolor='gray')
    g.node('c_{k}', fillcolor='palegoldenrod')
    for i in range(2, 6):
        g.edge(str(i), 'c_{k}', fillcolor='gray')
    g.attr(label=f'{cell_name.capitalize()} cell')
    image = Image.open(io.BytesIO(g.pipe()))
    return image

def plot_double_cells(arch_dict):
    if False:
        return 10
    image1 = plot_single_cell(arch_dict, 'normal')
    image2 = plot_single_cell(arch_dict, 'reduce')
    height_ratio = max(image1.size[1] / image1.size[0], image2.size[1] / image2.size[0])
    (_, axs) = plt.subplots(1, 2, figsize=(20, 10 * height_ratio))
    axs[0].imshow(image1)
    axs[1].imshow(image2)
    axs[0].axis('off')
    axs[1].axis('off')
    plt.show()
plot_double_cells(exported_arch)
plot_double_cells({'normal/op_2_0': 'sep_conv_3x3', 'normal/input_2_0': 1, 'normal/op_2_1': 'sep_conv_3x3', 'normal/input_2_1': 0, 'normal/op_3_0': 'sep_conv_3x3', 'normal/input_3_0': 1, 'normal/op_3_1': 'sep_conv_3x3', 'normal/input_3_1': 2, 'normal/op_4_0': 'sep_conv_3x3', 'normal/input_4_0': 1, 'normal/op_4_1': 'sep_conv_3x3', 'normal/input_4_1': 0, 'normal/op_5_0': 'sep_conv_3x3', 'normal/input_5_0': 1, 'normal/op_5_1': 'max_pool_3x3', 'normal/input_5_1': 0, 'reduce/op_2_0': 'sep_conv_3x3', 'reduce/input_2_0': 0, 'reduce/op_2_1': 'sep_conv_3x3', 'reduce/input_2_1': 1, 'reduce/op_3_0': 'dil_conv_5x5', 'reduce/input_3_0': 2, 'reduce/op_3_1': 'sep_conv_3x3', 'reduce/input_3_1': 0, 'reduce/op_4_0': 'dil_conv_5x5', 'reduce/input_4_0': 2, 'reduce/op_4_1': 'sep_conv_5x5', 'reduce/input_4_1': 1, 'reduce/op_5_0': 'sep_conv_5x5', 'reduce/input_5_0': 4, 'reduce/op_5_1': 'dil_conv_5x5', 'reduce/input_5_1': 2})
from nni.nas.space import model_context
with model_context(exported_arch):
    final_model = DartsSpace(width=16, num_cells=8, dataset='cifar')
train_loader = DataLoader(train_data, batch_size=96, num_workers=6)
valid_loader
max_epochs = 100
evaluator = Classification(learning_rate=0.001, weight_decay=0.0001, train_dataloaders=train_loader, val_dataloaders=valid_loader, max_epochs=max_epochs, gpus=1, export_onnx=False, fast_dev_run=fast_dev_run)
evaluator.fit(final_model)
import torch
from nni.nas.evaluator.pytorch import ClassificationModule

class DartsClassificationModule(ClassificationModule):

    def __init__(self, learning_rate: float=0.001, weight_decay: float=0.0, auxiliary_loss_weight: float=0.4, max_epochs: int=600):
        if False:
            print('Hello World!')
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.max_epochs = max_epochs
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, export_onnx=False)

    def configure_optimizers(self):
        if False:
            i = 10
            return i + 15
        'Customized optimizer with momentum, as well as a scheduler.'
        optimizer = torch.optim.SGD(self.parameters(), momentum=0.9, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return {'optimizer': optimizer, 'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=0.001)}

    def training_step(self, batch, batch_idx):
        if False:
            return 10
        'Training step, customized with auxiliary loss.'
        (x, y) = batch
        if self.auxiliary_loss_weight:
            (y_hat, y_aux) = self(x)
            loss_main = self.criterion(y_hat, y)
            loss_aux = self.criterion(y_aux, y)
            self.log('train_loss_main', loss_main)
            self.log('train_loss_aux', loss_aux)
            loss = loss_main + self.auxiliary_loss_weight * loss_aux
        else:
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        for (name, metric) in self.metrics.items():
            self.log('train_' + name, metric(y_hat, y), prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        if False:
            for i in range(10):
                print('nop')
        self.model.set_drop_path_prob(self.model.drop_path_prob * self.current_epoch / self.max_epochs)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])
from nni.nas.evaluator.pytorch import Lightning, Trainer
max_epochs = 50
evaluator = Lightning(DartsClassificationModule(0.025, 0.0003, 0.0, max_epochs), Trainer(gpus=1, max_epochs=max_epochs, fast_dev_run=fast_dev_run), train_dataloaders=search_train_loader, val_dataloaders=search_valid_loader)
strategy = DartsStrategy(gradient_clip_val=5.0)
model_space = DartsSpace(width=16, num_cells=8, dataset='cifar')
experiment = NasExperiment(model_space, evaluator, strategy)
experiment.run()
exported_arch = experiment.export_top_models(formatter='dict')[0]
exported_arch
plot_double_cells({'normal/op_2_0': 'sep_conv_3x3', 'normal/input_2_0': 0, 'normal/op_2_1': 'sep_conv_3x3', 'normal/input_2_1': 1, 'normal/op_3_0': 'sep_conv_3x3', 'normal/input_3_0': 1, 'normal/op_3_1': 'skip_connect', 'normal/input_3_1': 0, 'normal/op_4_0': 'sep_conv_3x3', 'normal/input_4_0': 0, 'normal/op_4_1': 'max_pool_3x3', 'normal/input_4_1': 1, 'normal/op_5_0': 'sep_conv_3x3', 'normal/input_5_0': 0, 'normal/op_5_1': 'sep_conv_3x3', 'normal/input_5_1': 1, 'reduce/op_2_0': 'max_pool_3x3', 'reduce/input_2_0': 0, 'reduce/op_2_1': 'sep_conv_5x5', 'reduce/input_2_1': 1, 'reduce/op_3_0': 'dil_conv_5x5', 'reduce/input_3_0': 2, 'reduce/op_3_1': 'max_pool_3x3', 'reduce/input_3_1': 0, 'reduce/op_4_0': 'max_pool_3x3', 'reduce/input_4_0': 0, 'reduce/op_4_1': 'sep_conv_3x3', 'reduce/input_4_1': 2, 'reduce/op_5_0': 'max_pool_3x3', 'reduce/input_5_0': 0, 'reduce/op_5_1': 'skip_connect', 'reduce/input_5_1': 2})

def cutout_transform(img, length: int=16):
    if False:
        print('Hello World!')
    (h, w) = (img.size(1), img.size(2))
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)
    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)
    mask[y1:y2, x1:x2] = 0.0
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img
transform_with_cutout = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD), cutout_transform])
train_data_cutout = nni.trace(CIFAR10)(root='./data', train=True, download=True, transform=transform_with_cutout)
train_loader_cutout = DataLoader(train_data_cutout, batch_size=96)
with model_context(exported_arch):
    final_model = DartsSpace(width=36, num_cells=20, dataset='cifar', auxiliary_loss=True, drop_path_prob=0.2)
max_epochs = 600
evaluator = Lightning(DartsClassificationModule(0.025, 0.0003, 0.4, max_epochs), trainer=Trainer(gpus=1, gradient_clip_val=5.0, max_epochs=max_epochs, fast_dev_run=fast_dev_run), train_dataloaders=train_loader_cutout, val_dataloaders=valid_loader)
evaluator.fit(final_model)