import numpy as np
import torch
from . import BaseWrapperDataset

class AppendTokenDataset(BaseWrapperDataset):

    def __init__(self, dataset, token=None):
        if False:
            i = 10
            return i + 15
        super().__init__(dataset)
        self.token = token
        if token is not None:
            self._sizes = np.array(dataset.sizes) + 1
        else:
            self._sizes = dataset.sizes

    def __getitem__(self, idx):
        if False:
            i = 10
            return i + 15
        item = self.dataset[idx]
        if self.token is not None:
            item = torch.cat([item, item.new([self.token])])
        return item

    @property
    def sizes(self):
        if False:
            print('Hello World!')
        return self._sizes

    def num_tokens(self, index):
        if False:
            for i in range(10):
                print('nop')
        n = self.dataset.num_tokens(index)
        if self.token is not None:
            n += 1
        return n

    def size(self, index):
        if False:
            while True:
                i = 10
        n = self.dataset.size(index)
        if self.token is not None:
            n += 1
        return n