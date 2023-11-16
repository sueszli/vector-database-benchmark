import os
os.environ['KERAS_BACKEND'] = 'torch'
import torch
import torch.nn as nn
import torch.optim as optim
from keras import layers
import keras
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
num_classes = 10
input_shape = (28, 28, 1)
learning_rate = 0.01
batch_size = 128
num_epochs = 1

def get_data():
    if False:
        for i in range(10):
            print('nop')
    ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    return dataset

def get_model():
    if False:
        print('Hello World!')
    model = keras.Sequential([layers.Input(shape=(28, 28, 1)), layers.Conv2D(32, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)), layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)), layers.Flatten(), layers.Dropout(0.5), layers.Dense(num_classes)])
    return model

class MyModel(nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.model = keras.Sequential([layers.Input(shape=(28, 28, 1)), layers.Conv2D(32, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)), layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)), layers.Flatten(), layers.Dropout(0.5), layers.Dense(num_classes)])

    def forward(self, x):
        if False:
            while True:
                i = 10
        return self.model(x)

def train(model, train_loader, num_epochs, optimizer, loss_fn):
    if False:
        i = 10
        return i + 15
    for epoch in range(num_epochs):
        running_loss = 0.0
        for (batch_idx, (inputs, targets)) in enumerate(train_loader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / 10}')
                running_loss = 0.0

def setup(current_gpu_index, num_gpu):
    if False:
        i = 10
        return i + 15
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '56492'
    device = torch.device('cuda:{}'.format(current_gpu_index))
    dist.init_process_group(backend='nccl', init_method='env://', world_size=num_gpu, rank=current_gpu_index)
    torch.cuda.set_device(device)

def prepare(dataset, current_gpu_index, num_gpu, batch_size):
    if False:
        while True:
            i = 10
    sampler = DistributedSampler(dataset, num_replicas=num_gpu, rank=current_gpu_index, shuffle=False)
    train_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)
    return train_loader

def cleanup():
    if False:
        i = 10
        return i + 15
    dist.destroy_process_group()

def main(current_gpu_index, num_gpu):
    if False:
        print('Hello World!')
    setup(current_gpu_index, num_gpu)
    dataset = get_data()
    model = get_model()
    dataloader = prepare(dataset, current_gpu_index, num_gpu, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(current_gpu_index)
    ddp_model = DDP(model, device_ids=[current_gpu_index], output_device=current_gpu_index)
    train(ddp_model, dataloader, num_epochs, optimizer, loss_fn)
    torch_module = MyModel().to(current_gpu_index)
    ddp_torch_module = DDP(torch_module, device_ids=[current_gpu_index], output_device=current_gpu_index)
    optimizer = optim.Adam(torch_module.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train(ddp_torch_module, dataloader, num_epochs, optimizer, loss_fn)
    cleanup()
if __name__ == '__main__':
    num_gpu = torch.cuda.device_count()
    print(f'Running on {num_gpu} GPUs')
    torch.multiprocessing.spawn(main, args=(num_gpu,), nprocs=num_gpu, join=True)