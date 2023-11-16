from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from util import get_normalized_data
import torch
from torch.autograd import Variable
from torch import optim
(Xtrain, Xtest, Ytrain, Ytest) = get_normalized_data()
(_, D) = Xtrain.shape
K = len(set(Ytrain))
model = torch.nn.Sequential()
model.add_module('dense1', torch.nn.Linear(D, 500))
model.add_module('bn1', torch.nn.BatchNorm1d(500))
model.add_module('relu1', torch.nn.ReLU())
model.add_module('dense2', torch.nn.Linear(500, 300))
model.add_module('bn2', torch.nn.BatchNorm1d(300))
model.add_module('relu2', torch.nn.ReLU())
model.add_module('dense3', torch.nn.Linear(300, K))
loss = torch.nn.CrossEntropyLoss(size_average=True)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(model, loss, optimizer, inputs, labels):
    if False:
        for i in range(10):
            print('nop')
    model.train()
    inputs = Variable(inputs, requires_grad=False)
    labels = Variable(labels, requires_grad=False)
    optimizer.zero_grad()
    logits = model.forward(inputs)
    output = loss.forward(logits, labels)
    output.backward()
    optimizer.step()
    return output.item()

def get_cost(model, loss, inputs, labels):
    if False:
        i = 10
        return i + 15
    model.eval()
    inputs = Variable(inputs, requires_grad=False)
    labels = Variable(labels, requires_grad=False)
    logits = model.forward(inputs)
    output = loss.forward(logits, labels)
    return output.item()

def predict(model, inputs):
    if False:
        i = 10
        return i + 15
    model.eval()
    inputs = Variable(inputs, requires_grad=False)
    logits = model.forward(inputs)
    return logits.data.numpy().argmax(axis=1)

def score(model, inputs, labels):
    if False:
        return 10
    predictions = predict(model, inputs)
    return np.mean(labels.numpy() == predictions)
Xtrain = torch.from_numpy(Xtrain).float()
Ytrain = torch.from_numpy(Ytrain).long()
Xtest = torch.from_numpy(Xtest).float()
Ytest = torch.from_numpy(Ytest).long()
epochs = 15
batch_size = 32
n_batches = Xtrain.size()[0] // batch_size
train_costs = []
test_costs = []
train_accuracies = []
test_accuracies = []
for i in range(epochs):
    cost = 0
    test_cost = 0
    for j in range(n_batches):
        Xbatch = Xtrain[j * batch_size:(j + 1) * batch_size]
        Ybatch = Ytrain[j * batch_size:(j + 1) * batch_size]
        cost += train(model, loss, optimizer, Xbatch, Ybatch)
    train_acc = score(model, Xtrain, Ytrain)
    test_acc = score(model, Xtest, Ytest)
    test_cost = get_cost(model, loss, Xtest, Ytest)
    print('Epoch: %d, cost: %f, acc: %.2f' % (i, test_cost, test_acc))
    train_costs.append(cost / n_batches)
    train_accuracies.append(train_acc)
    test_costs.append(test_cost)
    test_accuracies.append(test_acc)
plt.plot(train_costs, label='Train cost')
plt.plot(test_costs, label='Test cost')
plt.title('Cost')
plt.legend()
plt.show()
plt.plot(train_accuracies, label='Train accuracy')
plt.plot(test_accuracies, label='Test accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()