import logging
from collections import OrderedDict
from typing import Dict, Sequence
import numpy as np
from . import FairseqDataset, LanguagePairDataset
logger = logging.getLogger(__name__)

class RoundRobinZipDatasets(FairseqDataset):
    """Zip multiple :class:`~fairseq.data.FairseqDataset` instances together.

    Shorter datasets are repeated in a round-robin fashion to match the length
    of the longest one.

    Args:
        datasets (Dict[~fairseq.data.FairseqDataset]): a dictionary of
            :class:`~fairseq.data.FairseqDataset` instances.
        eval_key (str, optional): a key used at evaluation time that causes
            this instance to pass-through batches from *datasets[eval_key]*.
    """

    def __init__(self, datasets, eval_key=None):
        if False:
            while True:
                i = 10
        super().__init__()
        if isinstance(datasets, dict):
            datasets = OrderedDict(datasets)
        assert isinstance(datasets, OrderedDict)
        assert datasets, "Can't make a RoundRobinZipDatasets out of nothing"
        for dataset in datasets.values():
            assert isinstance(dataset, FairseqDataset)
        self.datasets = datasets
        self.eval_key = eval_key
        self.longest_dataset_key = max(datasets, key=lambda k: len(datasets[k]))
        self.longest_dataset = datasets[self.longest_dataset_key]
        self._ordered_indices: Dict[str, Sequence[int]] = None

    def _map_index(self, key, index):
        if False:
            for i in range(10):
                print('nop')
        assert self._ordered_indices is not None, 'Must call RoundRobinZipDatasets.ordered_indices() first'
        o = self._ordered_indices[key]
        return o[index % len(o)]

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        if self.eval_key is None:
            return OrderedDict([(key, dataset[self._map_index(key, index)]) for (key, dataset) in self.datasets.items()])
        else:
            return self.datasets[self.eval_key][self._map_index(self.eval_key, index)]

    def __len__(self):
        if False:
            print('Hello World!')
        if self._ordered_indices is not None:
            return len(self._ordered_indices[self.longest_dataset_key])
        return len(self.longest_dataset)

    def collater(self, samples):
        if False:
            while True:
                i = 10
        'Merge a list of samples to form a mini-batch.'
        if len(samples) == 0:
            return None
        if self.eval_key is None:
            return OrderedDict([(key, dataset.collater([sample[key] for sample in samples])) for (key, dataset) in self.datasets.items()])
        else:
            return self.datasets[self.eval_key].collater(samples)

    def num_tokens(self, index):
        if False:
            i = 10
            return i + 15
        "Return an example's length (number of tokens), used for batching."
        return max((dataset.num_tokens(self._map_index(key, index)) for (key, dataset) in self.datasets.items()))

    def size(self, index):
        if False:
            i = 10
            return i + 15
        "Return an example's size as a float or tuple. This value is used when\n        filtering a dataset with ``--max-positions``."
        return {key: dataset.size(self._map_index(key, index)) for (key, dataset) in self.datasets.items()}

    def ordered_indices(self):
        if False:
            for i in range(10):
                print('nop')
        'Ordered indices for batching.'
        if self._ordered_indices is None:
            self._ordered_indices = OrderedDict([(key, dataset.ordered_indices()) for (key, dataset) in self.datasets.items()])
        return np.arange(len(self))

    def filter_indices_by_size(self, indices, max_positions=None):
        if False:
            return 10
        '\n        Filter each sub-dataset independently, then update the round robin to work\n        on the filtered sub-datasets.\n        '

        def _deep_until_language_pair(dataset):
            if False:
                i = 10
                return i + 15
            if isinstance(dataset, LanguagePairDataset):
                return dataset
            if hasattr(dataset, 'tgt_dataset'):
                return _deep_until_language_pair(dataset.tgt_dataset)
            if hasattr(dataset, 'dataset'):
                return _deep_until_language_pair(dataset.dataset)
            raise Exception(f"Don't know how to unwrap this dataset: {dataset}")
        if not isinstance(max_positions, dict):
            max_positions = {k: max_positions for k in self.datasets.keys()}
        ignored_some = False
        for (key, dataset) in self.datasets.items():
            dataset = _deep_until_language_pair(dataset)
            (self._ordered_indices[key], ignored) = dataset.filter_indices_by_size(self._ordered_indices[key], max_positions[key])
            if len(ignored) > 0:
                ignored_some = True
                logger.warning(f'{len(ignored)} samples from {key} have invalid sizes and will be skipped, max_positions={max_positions[key]}, first few sample ids={ignored[:10]}')
        return (np.arange(len(self)), [0] if ignored_some else [])

    @property
    def supports_prefetch(self):
        if False:
            i = 10
            return i + 15
        return all((getattr(dataset, 'supports_prefetch', False) for dataset in self.datasets.values()))

    def prefetch(self, indices):
        if False:
            for i in range(10):
                print('nop')
        for (key, dataset) in self.datasets.items():
            dataset.prefetch([self._map_index(key, index) for index in indices])