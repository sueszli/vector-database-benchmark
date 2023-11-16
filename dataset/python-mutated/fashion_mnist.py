from __future__ import print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy
from bigdl.orca.learn.trigger import EveryEpoch

def train_data_creator(config={}, batch_size=4, download=True, data_dir='./data'):
    if False:
        i = 10
        return i + 15
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.FashionMNIST(root=data_dir, download=download, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    return trainloader

def validation_data_creator(config={}, batch_size=4, download=True, data_dir='./data'):
    if False:
        for i in range(10):
            print('nop')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return testloader

def matplotlib_imshow(img, one_channel=False):
    if False:
        print('Hello World!')
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

class Net(nn.Module):

    def __init__(self):
        if False:
            return 10
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        if False:
            return 10
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def model_creator(config):
    if False:
        while True:
            i = 10
    model = Net()
    return model

def optimizer_creator(model, config):
    if False:
        return 10
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return optimizer

def main():
    if False:
        return 10
    parser = argparse.ArgumentParser(description='PyTorch Tensorboard Example')
    parser.add_argument('--cluster_mode', type=str, default='local', help='The cluster mode, such as local, yarn, spark-submit or k8s.')
    parser.add_argument('--runtime', type=str, default='spark', help='The runtime backend, one of spark or ray.')
    parser.add_argument('--address', type=str, default='', help='The cluster address if the driver connects to an existing ray cluster. If it is empty, a new Ray cluster will be created.')
    parser.add_argument('--backend', type=str, default='spark', help='The backend of PyTorch Estimator; spark, ray and bigdl are supported.')
    parser.add_argument('--batch_size', type=int, default=4, help='The training batch size')
    parser.add_argument('--epochs', type=int, default=2, help='The number of epochs to train for')
    parser.add_argument('--data_dir', type=str, default='./data', help='The path of dataset')
    parser.add_argument('--download', type=bool, default=True, help='Download dataset or not')
    args = parser.parse_args()
    if args.runtime == 'ray':
        init_orca_context(runtime=args.runtime, address=args.address)
    elif args.cluster_mode == 'local':
        init_orca_context()
    elif args.cluster_mode.startswith('yarn'):
        init_orca_context(cluster_mode=args.cluster_mode, cores=4, num_nodes=2)
    elif args.cluster_mode == 'spark-submit':
        init_orca_context(cluster_mode=args.cluster_mode)
    tensorboard_dir = args.data_dir + 'runs'
    writer = SummaryWriter(tensorboard_dir + '/fashion_mnist_experiment_1')
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
    dataiter = iter(train_data_creator(config={}, batch_size=4, download=args.download, data_dir=args.data_dir))
    (images, labels) = dataiter.next()
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)
    writer.add_image('four_fashion_mnist_images', img_grid)
    writer.add_graph(model_creator(config={}), images)
    writer.close()
    criterion = nn.CrossEntropyLoss()
    batch_size = args.batch_size
    epochs = args.epochs
    if args.backend in ['ray', 'spark']:
        orca_estimator = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator, loss=criterion, metrics=[Accuracy()], model_dir=os.getcwd(), use_tqdm=True, backend=args.backend)
        stats = orca_estimator.fit(train_data_creator, epochs=epochs, batch_size=batch_size)
        for stat in stats:
            writer.add_scalar('training_loss', stat['train_loss'], stat['epoch'])
        print('Train stats: {}'.format(stats))
        val_stats = orca_estimator.evaluate(validation_data_creator, batch_size=batch_size)
        print('Validation stats: {}'.format(val_stats))
        orca_estimator.shutdown()
    else:
        invalidInputError(False, 'Only ray, and spark are supported as the backend, but got {}'.format(args.backend))
    stop_orca_context()
if __name__ == '__main__':
    main()