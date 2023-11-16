import unittest
import jittor as jt
import numpy as np
import torch
import torch.nn as tnn

class Net(tnn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        super(Net, self).__init__()
        self.conv1 = tnn.Conv2d(3, 6, 5)
        self.pool = tnn.MaxPool2d(2, 2)
        self.conv2 = tnn.Conv2d(6, 16, 5)
        self.fc1 = tnn.Linear(16 * 5 * 5, 120)
        self.fc2 = tnn.Linear(120, 84)
        self.fc3 = tnn.Linear(84, 10)

    def forward(self, x):
        if False:
            while True:
                i = 10
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class TestOptStateDict(unittest.TestCase):

    def test_opt_state_dict(self):
        if False:
            return 10
        return
        net = Net()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        img = torch.rand((2, 3, 40, 40))
        pred = net(img)
        optim.zero_grad()
        pred.sum().backward()
        optim.step()
if __name__ == '__main__':
    unittest.main()