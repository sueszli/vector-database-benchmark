import numpy as np
from fairseq.data import FairseqDataset

class DummyDataset(FairseqDataset):

    def __init__(self, batch, num_items, item_size):
        if False:
            print('Hello World!')
        super().__init__()
        self.batch = batch
        self.num_items = num_items
        self.item_size = item_size

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        return index

    def __len__(self):
        if False:
            return 10
        return self.num_items

    def collater(self, samples):
        if False:
            return 10
        return self.batch

    @property
    def sizes(self):
        if False:
            print('Hello World!')
        return np.array([self.item_size] * self.num_items)

    def num_tokens(self, index):
        if False:
            while True:
                i = 10
        return self.item_size

    def size(self, index):
        if False:
            i = 10
            return i + 15
        return self.item_size

    def ordered_indices(self):
        if False:
            print('Hello World!')
        return np.arange(self.num_items)

    @property
    def supports_prefetch(self):
        if False:
            for i in range(10):
                print('nop')
        return False