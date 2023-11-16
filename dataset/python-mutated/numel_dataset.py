import numpy as np
import torch
from . import BaseWrapperDataset

class NumelDataset(BaseWrapperDataset):

    def __init__(self, dataset, reduce=False):
        if False:
            i = 10
            return i + 15
        super().__init__(dataset)
        self.reduce = reduce

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        item = self.dataset[index]
        if torch.is_tensor(item):
            return torch.numel(item)
        else:
            return np.size(item)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.dataset)

    def collater(self, samples):
        if False:
            for i in range(10):
                print('nop')
        if self.reduce:
            return sum(samples)
        else:
            return torch.tensor(samples)