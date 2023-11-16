import torch
from fairseq.data import data_utils
from . import BaseWrapperDataset

class PaddingMaskDataset(BaseWrapperDataset):

    def __init__(self, dataset, left_pad, pad_length=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(dataset)
        self.left_pad = left_pad
        self.pad_length = pad_length

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        item = self.dataset[index]
        return torch.zeros_like(item).bool()

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.dataset)

    def collater(self, samples):
        if False:
            print('Hello World!')
        return data_utils.collate_tokens(samples, True, left_pad=self.left_pad, pad_to_length=self.pad_length)

class LeftPaddingMaskDataset(PaddingMaskDataset):

    def __init__(self, dataset):
        if False:
            return 10
        super().__init__(dataset, left_pad=True)

class RightPaddingMaskDataset(PaddingMaskDataset):

    def __init__(self, dataset):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(dataset, left_pad=False)