"""
Hyperparameter tuning with Ray Tune
===================================

Hyperparameter tuning can make the difference between an average model and a highly
accurate one. Often simple things like choosing a different learning rate or changing
a network layer size can have a dramatic impact on your model performance.

Fortunately, there are tools that help with finding the best combination of parameters.
`Ray Tune <https://docs.ray.io/en/latest/tune.html>`_ is an industry standard tool for
distributed hyperparameter tuning. Ray Tune includes the latest hyperparameter search
algorithms, integrates with TensorBoard and other analysis libraries, and natively
supports distributed training through `Ray's distributed machine learning engine
<https://ray.io/>`_.

In this tutorial, we will show you how to integrate Ray Tune into your PyTorch
training workflow. We will extend `this tutorial from the PyTorch documentation
<https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_ for training
a CIFAR10 image classifier.

As you will see, we only need to add some slight modifications. In particular, we
need to

1. wrap data loading and training in functions,
2. make some network parameters configurable,
3. add checkpointing (optional),
4. and define the search space for the model tuning

|

To run this tutorial, please make sure the following packages are
installed:

-  ``ray[tune]``: Distributed hyperparameter tuning library
-  ``torchvision``: For the data transformers

Setup / Imports
---------------
Let's start with the imports:
"""
from functools import partial
import os
from tempfile import TemporaryDirectory
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

def load_data(data_dir='./data'):
    if False:
        i = 10
        return i + 15
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return (trainset, testset)

class Net(nn.Module):

    def __init__(self, l1=120, l2=84):
        if False:
            print('Hello World!')
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        if False:
            return 10
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_cifar(config, data_dir=None):
    if False:
        while True:
            i = 10
    net = Net(config['l1'], config['l2'])
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9)
    checkpoint = train.get_checkpoint()
    if checkpoint:
        checkpoint_dir = checkpoint.to_directory()
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
        checkpoint_state = torch.load(checkpoint_path)
        start_epoch = checkpoint_state['epoch']
        net.load_state_dict(checkpoint_state['net_state_dict'])
        optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
    else:
        start_epoch = 0
    (trainset, testset) = load_data(data_dir)
    test_abs = int(len(trainset) * 0.8)
    (train_subset, val_subset) = random_split(trainset, [test_abs, len(trainset) - test_abs])
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=int(config['batch_size']), shuffle=True, num_workers=8)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=int(config['batch_size']), shuffle=True, num_workers=8)
    for epoch in range(start_epoch, 10):
        running_loss = 0.0
        epoch_steps = 0
        for (i, data) in enumerate(trainloader, 0):
            (inputs, labels) = data
            (inputs, labels) = (inputs.to(device), labels.to(device))
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / epoch_steps))
                running_loss = 0.0
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for (i, data) in enumerate(valloader, 0):
            with torch.no_grad():
                (inputs, labels) = data
                (inputs, labels) = (inputs.to(device), labels.to(device))
                outputs = net(inputs)
                (_, predicted) = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        checkpoint_data = {'epoch': epoch, 'net_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        with TemporaryDirectory() as tmpdir:
            torch.save(checkpoint_data, os.path.join(tmpdir, 'checkpoint.pt'))
            train.report({'loss': val_loss / val_steps, 'accuracy': correct / total}, checkpoint=Checkpoint.from_directory(tmpdir))
    print('Finished Training')

def test_accuracy(net, device='cpu'):
    if False:
        while True:
            i = 10
    (trainset, testset) = load_data()
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            (images, labels) = data
            (images, labels) = (images.to(device), labels.to(device))
            outputs = net(images)
            (_, predicted) = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    if False:
        i = 10
        return i + 15
    data_dir = os.path.abspath('./data')
    load_data(data_dir)
    config = {'l1': tune.choice([2 ** i for i in range(9)]), 'l2': tune.choice([2 ** i for i in range(9)]), 'lr': tune.loguniform(0.0001, 0.1), 'batch_size': tune.choice([2, 4, 8, 16])}
    scheduler = ASHAScheduler(metric='loss', mode='min', max_t=max_num_epochs, grace_period=1, reduction_factor=2)
    result = tune.run(partial(train_cifar, data_dir=data_dir), resources_per_trial={'cpu': 2, 'gpu': gpus_per_trial}, config=config, num_samples=num_samples, scheduler=scheduler)
    best_trial = result.get_best_trial('loss', 'min', 'last')
    print(f'Best trial config: {best_trial.config}')
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    best_trained_model = Net(best_trial.config['l1'], best_trial.config['l2'])
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)
    best_checkpoint = best_trial.checkpoint
    best_checkpoint_dir = best_checkpoint.to_directory()
    best_checkpoint_path = os.path.join(best_checkpoint_dir, 'checkpoint.pt')
    best_checkpoint_data = torch.load(best_checkpoint_path)
    best_trained_model.load_state_dict(best_checkpoint_data['net_state_dict'])
    test_acc = test_accuracy(best_trained_model, device)
    print('Best trial test set accuracy: {}'.format(test_acc))
if __name__ == '__main__':
    import sys
    sys.stdout.fileno = lambda : False
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)