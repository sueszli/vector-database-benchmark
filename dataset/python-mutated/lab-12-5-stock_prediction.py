"""
This script shows how to predict stock prices using a basic RNN
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import matplotlib
torch.manual_seed(777)
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def MinMaxScaler(data):
    if False:
        i = 10
        return i + 15
    ' Min Max Normalization\n\n    Parameters\n    ----------\n    data : numpy.ndarray\n        input data to be normalized\n        shape: [Batch size, dimension]\n\n    Returns\n    ----------\n    data : numpy.ndarry\n        normalized data\n        shape: [Batch size, dimension]\n\n    References\n    ----------\n    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html\n\n    '
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-07)
learning_rate = 0.01
num_epochs = 500
input_size = 5
hidden_size = 5
num_classes = 1
timesteps = seq_length = 7
num_layers = 1
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]]
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]
    print(_x, '->', _y)
    dataX.append(_x)
    dataY.append(_y)
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX = torch.Tensor(np.array(dataX[0:train_size]))
trainX = Variable(trainX)
testX = torch.Tensor(np.array(dataX[train_size:len(dataX)]))
testX = Variable(testX)
trainY = torch.Tensor(np.array(dataY[0:train_size]))
trainY = Variable(trainY)
testY = torch.Tensor(np.array(dataY[train_size:len(dataY)]))
testY = Variable(testY)

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        if False:
            i = 10
            return i + 15
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        (_, (h_out, _)) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out
lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    print('Epoch: %d, loss: %1.5f' % (epoch, loss.data[0]))
print('Learning finished!')
lstm.eval()
test_predict = lstm(testX)
test_predict = test_predict.data.numpy()
testY = testY.data.numpy()
plt.plot(testY)
plt.plot(test_predict)
plt.xlabel('Time Period')
plt.ylabel('Stock Price')
plt.show()