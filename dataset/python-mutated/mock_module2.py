import torch
from . import mock_module3

class Class1:

    def __init__(self, x, y):
        if False:
            while True:
                i = 10
        self.x = x
        self.y = y

    def method2(self, x):
        if False:
            while True:
                i = 10
        return mock_module3.method1([], x)

def method1(x, y):
    if False:
        print('Hello World!')
    torch.ones(1, 1)
    x.append(y)
    return x