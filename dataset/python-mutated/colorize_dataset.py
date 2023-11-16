import torch
from . import BaseWrapperDataset

class ColorizeDataset(BaseWrapperDataset):
    """Adds 'colors' property to net input that is obtained from the provided color getter for use by models"""

    def __init__(self, dataset, color_getter):
        if False:
            while True:
                i = 10
        super().__init__(dataset)
        self.color_getter = color_getter

    def collater(self, samples):
        if False:
            while True:
                i = 10
        base_collate = super().collater(samples)
        if len(base_collate) > 0:
            base_collate['net_input']['colors'] = torch.tensor(list((self.color_getter(self.dataset, s['id']) for s in samples)), dtype=torch.long)
        return base_collate