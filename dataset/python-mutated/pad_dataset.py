from fairseq.data import data_utils
from . import BaseWrapperDataset

class PadDataset(BaseWrapperDataset):

    def __init__(self, dataset, pad_idx, left_pad, pad_length=None):
        if False:
            print('Hello World!')
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
        self.pad_length = pad_length

    def collater(self, samples):
        if False:
            while True:
                i = 10
        return data_utils.collate_tokens(samples, self.pad_idx, left_pad=self.left_pad, pad_to_length=self.pad_length)

class LeftPadDataset(PadDataset):

    def __init__(self, dataset, pad_idx):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(dataset, pad_idx, left_pad=True)

class RightPadDataset(PadDataset):

    def __init__(self, dataset, pad_idx):
        if False:
            while True:
                i = 10
        super().__init__(dataset, pad_idx, left_pad=False)