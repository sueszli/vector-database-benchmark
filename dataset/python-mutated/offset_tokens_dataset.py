from . import BaseWrapperDataset

class OffsetTokensDataset(BaseWrapperDataset):

    def __init__(self, dataset, offset):
        if False:
            while True:
                i = 10
        super().__init__(dataset)
        self.offset = offset

    def __getitem__(self, idx):
        if False:
            for i in range(10):
                print('nop')
        return self.dataset[idx] + self.offset