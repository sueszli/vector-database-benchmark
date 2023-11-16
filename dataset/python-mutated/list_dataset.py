from . import BaseWrapperDataset

class ListDataset(BaseWrapperDataset):

    def __init__(self, dataset, sizes=None):
        if False:
            i = 10
            return i + 15
        super().__init__(dataset)
        self._sizes = sizes

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        for x in self.dataset:
            yield x

    def collater(self, samples):
        if False:
            while True:
                i = 10
        return samples

    @property
    def sizes(self):
        if False:
            i = 10
            return i + 15
        return self._sizes

    def num_tokens(self, index):
        if False:
            return 10
        return self.sizes[index]

    def size(self, index):
        if False:
            i = 10
            return i + 15
        return self.sizes[index]

    def set_epoch(self, epoch):
        if False:
            for i in range(10):
                print('nop')
        pass