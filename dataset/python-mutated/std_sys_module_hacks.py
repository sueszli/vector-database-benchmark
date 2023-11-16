import os
import os.path
import typing
import typing.io
import typing.re
import torch

class Module(torch.nn.Module):

    def forward(self):
        if False:
            for i in range(10):
                print('nop')
        return os.path.abspath('test')