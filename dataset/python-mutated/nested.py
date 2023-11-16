import torch

@torch.jit.script
class FooUniqueName:

    def __init__(self, y):
        if False:
            print('Hello World!')
        self.y = y