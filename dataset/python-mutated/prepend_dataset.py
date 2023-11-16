import numpy as np
import torch
from . import BaseWrapperDataset

class PrependDataset(BaseWrapperDataset):

    def __init__(self, dataset, prepend_getter, ensure_first_token_is=None):
        if False:
            print('Hello World!')
        super().__init__(dataset)
        self.prepend_getter = prepend_getter
        self.ensure_first_token = ensure_first_token_is

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        item = self.dataset[idx]
        is_tuple = isinstance(item, tuple)
        src = item[0] if is_tuple else item
        assert self.ensure_first_token is None or src[0] == self.ensure_first_token
        prepend_idx = self.prepend_getter(self.dataset, idx)
        assert isinstance(prepend_idx, int)
        src[0] = prepend_idx
        item = tuple((src,) + item[1:]) if is_tuple else src
        return item