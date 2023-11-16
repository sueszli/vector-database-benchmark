import torch.nn as nn
import numpy as np

class linear(nn.Module):
    """
    This is an nn.Module implementation to op.linear.
    The difference between this implementation and
    torch.nn.linear is that this implementation take input
    shape (batch_size, *) and output shape is (batch_size, output_size)
    """

    def __init__(self, input_size, output_size):
        if False:
            return 10
        super().__init__()
        self.linear_torch = nn.Linear(input_size, output_size)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.linear_torch(x)
        return x

def flatten(x):
    if False:
        print('Hello World!')
    '\n    This is an torch implementation to op.flatten.\n    '
    return x.view(-1, np.prod(x.shape[1:]))