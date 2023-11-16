from functools import lru_cache
from . import BaseWrapperDataset

class LRUCacheDataset(BaseWrapperDataset):

    def __init__(self, dataset, token=None):
        if False:
            print('Hello World!')
        super().__init__(dataset)

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        if False:
            return 10
        return self.dataset[index]

    @lru_cache(maxsize=8)
    def collater(self, samples):
        if False:
            for i in range(10):
                print('nop')
        return self.dataset.collater(samples)