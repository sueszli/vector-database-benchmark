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
model.add_module('relu1', torch.nn.ReLU())
model.add_module('dense2', torch.nn.Linear(500, 300))
model.add_module('relu2', torch.nn.ReLU())
model.add_module('dense3', torch.nn.Linear(300, K))
loss = torch.nn.CrossEntropyLoss(size_average=True)
optimizer = optim.Adam(model.parameters())

def train(model, loss, optimizer, inputs, labels):
    if False:
        print('Hello World!')
    inputs = Variable(inputs, requires_grad=False)
    labels = Variable(labels, requires_grad=False)
    optimizer.zero_grad()
    logits = model.forward(inputs)
    output = loss.forward(logits, labels)
    output.backward()
    optimizer.step()
    return output.item()

def predict(model, inputs):
    if False:
        i = 10
        return i + 15
    inputs = Variable(inputs, requires_grad=False)
    logits = model.forward(inputs)
    return logits.data.numpy().argmax(axis=1)
Xtrain = torch.from_numpy(Xtrain).float()
Ytrain = torch.from_numpy(Ytrain).long()
Xtest = torch.from_numpy(Xtest).float()
epochs = 15
batch_size = 32
n_batches = Xtrain.size()[0] // batch_size
costs = []
test_accuracies = []
for i in range(epochs):
    cost = 0.0
    for j in range(n_batches):
        Xbatch = Xtrain[j * batch_size:(j + 1) * batch_size]
        Ybatch = Ytrain[j * batch_size:(j + 1) * batch_size]
        cost += train(model, loss, optimizer, Xbatch, Ybatch)
    Ypred = predict(model, Xtest)
    acc = np.mean(Ytest == Ypred)
    print('Epoch: %d, cost: %f, acc: %.2f' % (i, cost / n_batches, acc))
    costs.append(cost / n_batches)
    test_accuracies.append(acc)
plt.plot(costs)
plt.title('Training cost')
plt.show()
plt.plot(test_accuracies)
plt.title('Test accuracies')
plt.show()