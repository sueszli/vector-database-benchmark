import os
os.environ['KERAS_BACKEND'] = 'torch'
import torch
import torch.nn as nn
import torch.optim as optim
from keras import layers
import keras
import numpy as np
num_classes = 10
input_shape = (28, 28, 1)
learning_rate = 0.01
batch_size = 64
num_epochs = 1
((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
model = keras.Sequential([layers.Input(shape=(28, 28, 1)), layers.Conv2D(32, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)), layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)), layers.Flatten(), layers.Dropout(0.5), layers.Dense(num_classes)])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

def train(model, train_loader, num_epochs, optimizer, loss_fn):
    if False:
        i = 10
        return i + 15
    for epoch in range(num_epochs):
        running_loss = 0.0
        for (batch_idx, (inputs, targets)) in enumerate(train_loader):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / 10}')
                running_loss = 0.0
dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
train(model, train_loader, num_epochs, optimizer, loss_fn)

class MyModel(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.model = keras.Sequential([layers.Input(shape=(28, 28, 1)), layers.Conv2D(32, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)), layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)), layers.Flatten(), layers.Dropout(0.5), layers.Dense(num_classes)])

    def forward(self, x):
        if False:
            while True:
                i = 10
        return self.model(x)
torch_module = MyModel()
optimizer = optim.Adam(torch_module.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
train(torch_module, train_loader, num_epochs, optimizer, loss_fn)