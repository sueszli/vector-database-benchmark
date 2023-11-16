import torch
from . import FairseqDataset

class RawLabelDataset(FairseqDataset):

    def __init__(self, labels):
        if False:
            while True:
                i = 10
        super().__init__()
        self.labels = labels

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        return self.labels[index]

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.labels)

    def collater(self, samples):
        if False:
            while True:
                i = 10
        return torch.tensor(samples)