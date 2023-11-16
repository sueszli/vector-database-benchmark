import torch

def method1(x, y):
    if False:
        for i in range(10):
            print('nop')
    torch.ones(1, 1)
    x.append(y)
    return x