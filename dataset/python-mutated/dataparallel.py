import torch
import torch.nn as nn

class DataParallel(nn.DataParallel):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        if False:
            i = 10
            return i + 15
        super().__init__(module, device_ids=None, output_device=None, dim=0)
        self.module = module

    def parameters(self, recurse: bool=True):
        if False:
            i = 10
            return i + 15
        return self.module.parameters(recurse=True)