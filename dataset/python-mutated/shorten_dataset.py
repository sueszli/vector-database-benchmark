import numpy as np
from fairseq.data import data_utils
from . import BaseWrapperDataset

class TruncateDataset(BaseWrapperDataset):
    """Truncate a sequence by returning the first truncation_length tokens"""

    def __init__(self, dataset, truncation_length):
        if False:
            i = 10
            return i + 15
        super().__init__(dataset)
        assert truncation_length is not None
        self.truncation_length = truncation_length
        self.dataset = dataset

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        item = self.dataset[index]
        item_len = item.size(0)
        if item_len > self.truncation_length:
            item = item[:self.truncation_length]
        return item

    @property
    def sizes(self):
        if False:
            for i in range(10):
                print('nop')
        return np.minimum(self.dataset.sizes, self.truncation_length)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.dataset)

class RandomCropDataset(TruncateDataset):
    """Truncate a sequence by returning a random crop of truncation_length tokens"""

    def __init__(self, dataset, truncation_length, seed=1):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(dataset, truncation_length)
        self.seed = seed
        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        if False:
            print('Hello World!')
        return True

    def set_epoch(self, epoch, **unused):
        if False:
            while True:
                i = 10
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            item_len = item.size(0)
            excess = item_len - self.truncation_length
            if excess > 0:
                start_idx = np.random.randint(0, excess)
                item = item[start_idx:start_idx + self.truncation_length]
            return item

def maybe_shorten_dataset(dataset, split, shorten_data_split_list, shorten_method, tokens_per_sample, seed):
    if False:
        for i in range(10):
            print('nop')
    truncate_split = split in shorten_data_split_list.split(',') or len(shorten_data_split_list) == 0
    if shorten_method == 'truncate' and truncate_split:
        dataset = TruncateDataset(dataset, tokens_per_sample)
    elif shorten_method == 'random_crop' and truncate_split:
        dataset = RandomCropDataset(dataset, tokens_per_sample, seed)
    return dataset