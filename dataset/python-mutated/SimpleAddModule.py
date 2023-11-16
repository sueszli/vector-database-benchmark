import torch
from utils import NUM_LOOP_ITERS

def add_tensors_loop(x, y):
    if False:
        print('Hello World!')
    z = torch.add(x, y)
    for i in range(NUM_LOOP_ITERS):
        z = torch.add(z, x)
    return z

class SimpleAddModule(torch.nn.Module):

    def __init__(self, add_op):
        if False:
            while True:
                i = 10
        super().__init__()
        self.add_op = add_op

    def forward(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        return self.add_op(x, y)