import torch
from . import FairseqDataset

class ConcatSentencesDataset(FairseqDataset):

    def __init__(self, *datasets):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.datasets = datasets
        assert all((len(ds) == len(datasets[0]) for ds in datasets)), 'datasets must have the same length'

    def __getitem__(self, index):
        if False:
            return 10
        return torch.cat([ds[index] for ds in self.datasets])

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.datasets[0])

    def collater(self, samples):
        if False:
            i = 10
            return i + 15
        return self.datasets[0].collater(samples)

    @property
    def sizes(self):
        if False:
            i = 10
            return i + 15
        return sum((ds.sizes for ds in self.datasets))

    def num_tokens(self, index):
        if False:
            while True:
                i = 10
        return sum((ds.num_tokens(index) for ds in self.datasets))

    def size(self, index):
        if False:
            for i in range(10):
                print('nop')
        return sum((ds.size(index) for ds in self.datasets))

    def ordered_indices(self):
        if False:
            print('Hello World!')
        return self.datasets[0].ordered_indices()

    @property
    def supports_prefetch(self):
        if False:
            return 10
        return any((getattr(ds, 'supports_prefetch', False) for ds in self.datasets))

    def prefetch(self, indices):
        if False:
            print('Hello World!')
        for ds in self.datasets:
            if getattr(ds, 'supports_prefetch', False):
                ds.prefetch(indices)

    def set_epoch(self, epoch):
        if False:
            return 10
        super().set_epoch(epoch)
        for ds in self.datasets:
            if hasattr(ds, 'set_epoch'):
                ds.set_epoch(epoch)