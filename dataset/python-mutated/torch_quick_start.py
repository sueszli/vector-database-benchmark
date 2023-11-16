import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def get_dataset():
    if False:
        while True:
            i = 10
    return datasets.FashionMNIST(root='/tmp/data', train=True, download=True, transform=ToTensor())

class NeuralNetwork(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(28 * 28, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        inputs = self.flatten(inputs)
        logits = self.linear_relu_stack(inputs)
        return logits

def train_func():
    if False:
        return 10
    num_epochs = 3
    batch_size = 64
    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = NeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        for (inputs, labels) in dataloader:
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch}, loss: {loss.item()}')
from ray import train

def train_func_distributed():
    if False:
        print('Hello World!')
    num_epochs = 3
    batch_size = 64
    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader = train.torch.prepare_data_loader(dataloader)
    model = NeuralNetwork()
    model = train.torch.prepare_model(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        for (inputs, labels) in dataloader:
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch}, loss: {loss.item()}')
if __name__ == '__main__':
    train_func()
    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig
    use_gpu = False
    trainer = TorchTrainer(train_func_distributed, scaling_config=ScalingConfig(num_workers=4, use_gpu=use_gpu))
    results = trainer.fit()