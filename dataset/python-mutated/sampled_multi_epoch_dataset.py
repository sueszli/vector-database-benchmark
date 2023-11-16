import hashlib
import logging
import math
import numpy as np
from fairseq.data import SampledMultiDataset
from .sampled_multi_dataset import CollateFormat, default_virtual_size_func
logger = logging.getLogger(__name__)

class SampledMultiEpochDataset(SampledMultiDataset):
    """Samples from multiple sub-datasets according to sampling ratios
       using virtual epoch sizes to speed up dataloading.
    Args:
        datasets (
            List[~torch.utils.data.Dataset]
            or OrderedDict[str, ~torch.utils.data.Dataset]
        ): datasets
        sampling_ratios (List[float]): list of probability of each dataset to be sampled
            (default: None, which corresponds to concating all dataset together).
        seed (int): RNG seed to use (default: 2).
        epoch (int): starting epoch number (default: 1).
        eval_key (str, optional): a key used at evaluation time that causes
            this instance to pass-through batches from *datasets[eval_key]*.
        collate_format (CollateFormat):  collater output format, either CollateFormat.ordered_dict or
            CollateFormat.single (default: CollateFormat.single) where CollateFormat.single configures
            the collater to output batches of data mixed from all sub-datasets,
            and CollateFormat.ordered_dict configures the collater to output a dictionary of batches indexed by keys
            of sub-datasets.
            Note that not all sub-datasets will present in a single batch in both formats.
        virtual_size (int, or callable): the expected virtual size of the dataset (default: default_virtual_size_func).
        split (str): the split of the data, e.g. 'train', 'valid' or 'test'.
        virtual_epoch_size (int): virtual epoch size, the dataset will go through the data by
            this virtual epoch size one by one to speed up data loading, e.g. indicing and filtering
            can be performed whenever a virtual epoch is loaded without waiting for the whole dataset to be loaded.
        shared_collater (bool): whether or not to all sub-datasets have the same collater.
        shard_epoch (int): the real epoch number for shard selection.
        shuffle (bool): whether or not to shuffle data (default: True).
    """

    def __init__(self, datasets, sampling_ratios=None, seed=2, epoch=1, eval_key=None, collate_format=CollateFormat.single, virtual_size=default_virtual_size_func, split='', virtual_epoch_size=None, shared_collater=False, shard_epoch=1, shuffle=True):
        if False:
            while True:
                i = 10
        self.virtual_epoch_size = virtual_epoch_size
        self._current_epoch_start_index = None
        self._random_global_indices = None
        self.shard_epoch = shard_epoch if shard_epoch is not None else 1
        self.load_next_shard = None
        self._epoch_sizes = None
        super().__init__(datasets=datasets, sampling_ratios=sampling_ratios, seed=seed, epoch=epoch, eval_key=eval_key, collate_format=collate_format, virtual_size=virtual_size, split=split, shared_collater=shared_collater, shuffle=shuffle)

    def _setup(self, epoch):
        if False:
            return 10
        self.virtual_epoch_size = self.virtual_epoch_size if self.virtual_epoch_size is not None else self.virtual_size
        if self.virtual_epoch_size > self.virtual_size:
            logger.warning(f'virtual epoch size {self.virtual_epoch_size} is greater than virtual dataset size {self.virtual_size}')
            self.virtual_epoch_size = self.virtual_size
        self.num_virtual_epochs = math.ceil(self.virtual_size / self.virtual_epoch_size)
        self._current_epoch_start_index = self._get_epoch_start_index(epoch)
        logger.info(f'virtual epoch size {self.virtual_epoch_size}; virtual dataset size {self.virtual_size}')

    def _map_epoch_index_to_global(self, index):
        if False:
            while True:
                i = 10
        index = self._current_epoch_start_index + index
        return self._random_global_indices[index]

    @property
    def sizes(self):
        if False:
            i = 10
            return i + 15
        if self._epoch_sizes is not None:
            return self._epoch_sizes
        _sizes = super().sizes
        indices = self._random_global_indices[self._current_epoch_start_index:self._current_epoch_start_index + len(self)]
        self._epoch_sizes = _sizes[indices]
        del self._sizes
        self._sizes = None
        return self._epoch_sizes

    def _get_dataset_and_index(self, index):
        if False:
            i = 10
            return i + 15
        i = self._map_epoch_index_to_global(index)
        return super()._get_dataset_and_index(i)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.virtual_epoch_size if self._current_epoch_start_index + self.virtual_epoch_size < self.virtual_size else self.virtual_size - self._current_epoch_start_index

    def set_epoch(self, epoch):
        if False:
            for i in range(10):
                print('nop')
        if self._current_epoch_start_index is None:
            self._setup(epoch)
            self._next_virtual_epoch(epoch)
        else:
            if epoch == self._cur_epoch:
                return
            self._next_virtual_epoch(epoch)

    def _get_epoch_start_index(self, epoch):
        if False:
            return 10
        assert epoch >= 1
        return (epoch - 1) % self.num_virtual_epochs * self.virtual_epoch_size

    def _next_global_indices(self, epoch):
        if False:
            print('Hello World!')
        rng = np.random.RandomState([int(hashlib.sha1(str(self.__class__.__name__).encode('utf-8')).hexdigest(), 16) % 2 ** 32, self.seed % 2 ** 32, epoch])
        del self._random_global_indices
        self._random_global_indices = rng.choice(self.virtual_size, self.virtual_size, replace=False)
        if self.load_next_shard is None:
            self.load_next_shard = False
        else:
            self.shard_epoch += 1
            self.load_next_shard = True
            logger.info(f'to load next epoch/shard in next load_dataset: epoch={epoch}/shard_epoch={self.shard_epoch}')

    def _next_virtual_epoch(self, epoch):
        if False:
            i = 10
            return i + 15
        index = self._get_epoch_start_index(epoch)
        if index == 0 or self._random_global_indices is None:
            logger.info(f'establishing a new set of global virtual indices for epoch={epoch}/shard_epoch={self.shard_epoch}')
            super().set_epoch(epoch)
            self._next_global_indices(epoch)
        else:
            self._cur_epoch = epoch
        self._clean_if_not_none([self._epoch_sizes])
        self._epoch_sizes = None
        self._current_epoch_start_index = index