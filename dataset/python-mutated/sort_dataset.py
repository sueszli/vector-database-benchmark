import numpy as np
from . import BaseWrapperDataset

class SortDataset(BaseWrapperDataset):

    def __init__(self, dataset, sort_order):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(dataset)
        if not isinstance(sort_order, (list, tuple)):
            sort_order = [sort_order]
        self.sort_order = sort_order
        assert all((len(so) == len(dataset) for so in sort_order))

    def ordered_indices(self):
        if False:
            while True:
                i = 10
        return np.lexsort(self.sort_order)