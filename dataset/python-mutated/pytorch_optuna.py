import os
from filelock import FileLock
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
EPOCH_SIZE = 512
TEST_SIZE = 256

def train(model, optimizer, train_loader, device=None):
    if False:
        return 10
    device = device or torch.device('cpu')
    model.train()
    for (batch_idx, (data, target)) in enumerate(train_loader):
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        (data, target) = (data.to(device), target.to(device))
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def test(model, data_loader, device=None):
    if False:
        while True:
            i = 10
    device = device or torch.device('cpu')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (batch_idx, (data, target)) in enumerate(data_loader):
            if batch_idx * len(data) > TEST_SIZE:
                break
            (data, target) = (data.to(device), target.to(device))
            outputs = model(data)
            (_, predicted) = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

def load_data():
    if False:
        print('Hello World!')
    mnist_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    with FileLock(os.path.expanduser('~/data.lock')):
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('~/data', train=True, download=True, transform=mnist_transforms), batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('~/data', train=False, download=True, transform=mnist_transforms), batch_size=64, shuffle=True)
    return (train_loader, test_loader)

class ConvNet(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
import torch
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch

def objective(config):
    if False:
        i = 10
        return i + 15
    (train_loader, test_loader) = load_data()
    model = ConvNet().to('cpu')
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    while True:
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)
        train.report({'mean_accuracy': acc})
search_space = {'lr': tune.loguniform(0.0001, 0.01), 'momentum': tune.uniform(0.1, 0.9)}
algo = OptunaSearch()
tuner = tune.Tuner(objective, tune_config=tune.TuneConfig(metric='mean_accuracy', mode='max', search_alg=algo), run_config=train.RunConfig(stop={'training_iteration': 5}), param_space=search_space)
results = tuner.fit()
print('Best config is:', results.get_best_result().config)