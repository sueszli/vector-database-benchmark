import ray.train.torch
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn
import torchmetrics
from torch.optim import Adam
import numpy as np

def train_func(config):
    if False:
        i = 10
        return i + 15
    n = 100
    X = torch.Tensor(np.random.normal(0, 1, size=(n, 4)))
    X_valid = torch.Tensor(np.random.normal(0, 1, size=(n, 4)))
    Y = torch.Tensor(np.random.uniform(0, 1, size=(n, 1)))
    Y_valid = torch.Tensor(np.random.uniform(0, 1, size=(n, 1)))
    model = ray.train.torch.prepare_model(nn.Linear(4, 1))
    criterion = nn.MSELoss()
    mape = torchmetrics.MeanAbsolutePercentageError()
    mean_valid_loss = torchmetrics.MeanMetric()
    optimizer = Adam(model.parameters(), lr=0.0003)
    for epoch in range(config['num_epochs']):
        model.train()
        y = model.forward(X)
        loss = criterion(y, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(X_valid)
            valid_loss = criterion(pred, Y_valid)
            mean_valid_loss(valid_loss)
            mape(pred, Y_valid)
        valid_loss = valid_loss.item()
        mape_collected = mape.compute().item()
        mean_valid_loss_collected = mean_valid_loss.compute().item()
        train.report({'mape_collected': mape_collected, 'valid_loss': valid_loss, 'mean_valid_loss_collected': mean_valid_loss_collected})
        mape.reset()
        mean_valid_loss.reset()
trainer = TorchTrainer(train_func, train_loop_config={'num_epochs': 5}, scaling_config=ScalingConfig(num_workers=2))
result = trainer.fit()
print(result.metrics['valid_loss'], result.metrics['mean_valid_loss_collected'])