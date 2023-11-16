"""
Hello, NAS!
===========

This is the 101 tutorial of Neural Architecture Search (NAS) on NNI.
In this tutorial, we will search for a neural architecture on MNIST dataset with the help of NAS framework of NNI, i.e., *Retiarii*.
We use multi-trial NAS as an example to show how to construct and explore a model space.

There are mainly three crucial components for a neural architecture search task, namely,

* Model search space that defines a set of models to explore.
* A proper strategy as the method to explore this model space.
* A model evaluator that reports the performance of every model in the space.

Currently, PyTorch is the only supported framework by Retiarii, and we have only tested **PyTorch 1.9 to 1.13**.
This tutorial assumes PyTorch context but it should also apply to other frameworks, which is in our future plan.

Define your Model Space
-----------------------

Model space is defined by users to express a set of models that users want to explore, which contains potentially good-performing models.
In this framework, a model space is defined with two parts: a base model and possible mutations on the base model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import nni
from nni.nas.nn.pytorch import LayerChoice, ModelSpace, MutableDropout, MutableLinear

class Net(ModelSpace):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if False:
            print('Hello World!')
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
        output = F.log_softmax(x, dim=1)
        return output

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        if False:
            return 10
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self.pointwise(self.depthwise(x))

class MyModelSpace(ModelSpace):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = LayerChoice([nn.Conv2d(32, 64, 3, 1), DepthwiseSeparableConv(32, 64)], label='conv2')
        self.dropout1 = MutableDropout(nni.choice('dropout', [0.25, 0.5, 0.75]))
        self.dropout2 = nn.Dropout(0.5)
        feature = nni.choice('feature', [64, 128, 256])
        self.fc1 = MutableLinear(9216, feature)
        self.fc2 = MutableLinear(feature, 10)

    def forward(self, x):
        if False:
            print('Hello World!')
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
        output = F.log_softmax(x, dim=1)
        return output
model_space = MyModelSpace()
model_space
import nni.nas.strategy as strategy
search_strategy = strategy.Random()
import nni
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def train_epoch(model, device, train_loader, optimizer, epoch):
    if False:
        return 10
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for (batch_idx, (data, target)) in enumerate(train_loader):
        (data, target) = (data.to(device), target.to(device))
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), loss.item()))

def test_epoch(model, device, test_loader):
    if False:
        while True:
            i = 10
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (data, target) in test_loader:
            (data, target) = (data.to(device), target.to(device))
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), accuracy))
    return accuracy

def evaluate_model(model):
    if False:
        print('Hello World!')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(MNIST('data/mnist', download=True, transform=transf), batch_size=64, shuffle=True)
    test_loader = DataLoader(MNIST('data/mnist', download=True, train=False, transform=transf), batch_size=64)
    for epoch in range(3):
        train_epoch(model, device, train_loader, optimizer, epoch)
        accuracy = test_epoch(model, device, test_loader)
        nni.report_intermediate_result(accuracy)
    nni.report_final_result(accuracy)
from nni.nas.evaluator import FunctionalEvaluator
evaluator = FunctionalEvaluator(evaluate_model)
from nni.nas.experiment import NasExperiment
exp = NasExperiment(model_space, evaluator, search_strategy)
exp.config.max_trial_number = 3
exp.config.trial_concurrency = 1
exp.config.trial_gpu_number = 0
exp.run(port=8081)
import os
from pathlib import Path

def evaluate_model_with_visualization(model):
    if False:
        while True:
            i = 10
    if 'NNI_OUTPUT_DIR' in os.environ:
        dummy_input = torch.zeros(1, 3, 32, 32)
        torch.onnx.export(model, (dummy_input,), Path(os.environ['NNI_OUTPUT_DIR']) / 'model.onnx')
    evaluate_model(model)
for model_dict in exp.export_top_models(formatter='dict'):
    print(model_dict)