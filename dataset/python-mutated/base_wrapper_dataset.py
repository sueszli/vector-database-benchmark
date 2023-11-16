from torch.utils.data.dataloader import default_collate
from . import FairseqDataset

class BaseWrapperDataset(FairseqDataset):

    def __init__(self, dataset):
        if False:
            return 10
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        return self.dataset[index]

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.dataset)

    def collater(self, samples):
        if False:
            return 10
        if hasattr(self.dataset, 'collater'):
            return self.dataset.collater(samples)
        else:
            return default_collate(samples)

    @property
    def sizes(self):
        if False:
            while True:
                i = 10
        return self.dataset.sizes

    def num_tokens(self, index):
        if False:
            while True:
                i = 10
        return self.dataset.num_tokens(index)

    def size(self, index):
        if False:
            return 10
        return self.dataset.size(index)

    def ordered_indices(self):
        if False:
            print('Hello World!')
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self.dataset, 'supports_prefetch', False)

    def attr(self, attr: str, index: int):
        if False:
            i = 10
            return i + 15
        return self.dataset.attr(attr, index)

    def prefetch(self, indices):
        if False:
            i = 10
            return i + 15
        self.dataset.prefetch(indices)

    def get_batch_shapes(self):
        if False:
            print('Hello World!')
        return self.dataset.get_batch_shapes()

    def batch_by_size(self, indices, max_tokens=None, max_sentences=None, required_batch_size_multiple=1):
        if False:
            print('Hello World!')
        return self.dataset.batch_by_size(indices, max_tokens=max_tokens, max_sentences=max_sentences, required_batch_size_multiple=required_batch_size_multiple)

    def filter_indices_by_size(self, indices, max_sizes):
        if False:
            i = 10
            return i + 15
        return self.dataset.filter_indices_by_size(indices, max_sizes)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        if False:
            for i in range(10):
                print('nop')
        return self.dataset.can_reuse_epoch_itr_across_epochs

    def set_epoch(self, epoch):
        if False:
            i = 10
            return i + 15
        super().set_epoch(epoch)
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)