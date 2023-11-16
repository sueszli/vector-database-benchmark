import torch
from . import FairseqDataset

class IdDataset(FairseqDataset):

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        return index

    def __len__(self):
        if False:
            return 10
        return 0

    def collater(self, samples):
        if False:
            i = 10
            return i + 15
        return torch.tensor(samples)