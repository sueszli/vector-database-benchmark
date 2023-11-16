"""
TODO (huxu): fairseq wrapper class for all dataset you defined: mostly MMDataset.
"""
from collections import OrderedDict
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from fairseq.data import FairseqDataset, data_utils

class FairseqMMDataset(FairseqDataset):
    """
    A wrapper class for MMDataset for fairseq.
    """

    def __init__(self, mmdataset):
        if False:
            print('Hello World!')
        if not isinstance(mmdataset, Dataset):
            raise TypeError('mmdataset must be of type `torch.utils.data.dataset`.')
        self.mmdataset = mmdataset

    def set_epoch(self, epoch, **unused):
        if False:
            i = 10
            return i + 15
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx):
        if False:
            return 10
        with data_utils.numpy_seed(43211, self.epoch, idx):
            return self.mmdataset[idx]

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.mmdataset)

    def collater(self, samples):
        if False:
            while True:
                i = 10
        if hasattr(self.mmdataset, 'collator'):
            return self.mmdataset.collator(samples)
        if len(samples) == 0:
            return {}
        if isinstance(samples[0], dict):
            batch = OrderedDict()
            for key in samples[0]:
                if samples[0][key] is not None:
                    batch[key] = default_collate([sample[key] for sample in samples])
            return batch
        else:
            return default_collate(samples)

    def size(self, index):
        if False:
            for i in range(10):
                print('nop')
        "dummy implementation: we don't use --max-tokens"
        return 1

    def num_tokens(self, index):
        if False:
            print('Hello World!')
        "dummy implementation: we don't use --max-tokens"
        return 1