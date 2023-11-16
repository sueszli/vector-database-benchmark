from collections import OrderedDict
from typing import Callable, Dict, List
import numpy as np
from . import FairseqDataset

def uniform_sampler(x):
    if False:
        print('Hello World!')
    return np.random.choice(x, 1).item()

class MultiCorpusSampledDataset(FairseqDataset):
    """
    Stores multiple instances of FairseqDataset together and in every iteration
    creates a batch by first sampling a dataset according to a specified
    probability distribution and then getting instances from that dataset.

    Args:
        datasets: an OrderedDict of FairseqDataset instances.
        sampling_func: A function for sampling over list of dataset keys.
            The default strategy is to sample uniformly.
    """

    def __init__(self, datasets: Dict[str, FairseqDataset], sampling_func: Callable[[List], int]=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        assert isinstance(datasets, OrderedDict)
        self.datasets = datasets
        if sampling_func is None:
            sampling_func = uniform_sampler
        self.sampling_func = sampling_func
        self.total_num_instances = 0
        for (_, dataset) in datasets.items():
            assert isinstance(dataset, FairseqDataset)
            self.total_num_instances += len(dataset)
        self._ordered_indices = None

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Length of this dataset is the sum of individual datasets\n        '
        return self.total_num_instances

    def ordered_indices(self):
        if False:
            print('Hello World!')
        "\n        Ordered indices for batching. Here we call the underlying\n        dataset's ordered_indices() so that we get the same random ordering\n        as we would have from using the underlying dataset directly.\n        "
        if self._ordered_indices is None:
            self._ordered_indices = OrderedDict([(key, dataset.ordered_indices()) for (key, dataset) in self.datasets.items()])
        return np.arange(len(self))

    def _map_index_to_dataset(self, key: int, index: int):
        if False:
            while True:
                i = 10
        '\n        Different underlying datasets have different lengths. In order to ensure\n        we are not accessing an index outside the range of the current dataset\n        size, we wrap around. This function should be called after we have\n        created an ordering for this and all underlying datasets.\n        '
        assert self._ordered_indices is not None, 'Must call MultiCorpusSampledDataset.ordered_indices() first'
        mapped_index = index % len(self.datasets[key])
        return self._ordered_indices[key][mapped_index]

    def __getitem__(self, index: int):
        if False:
            i = 10
            return i + 15
        '\n        Get the item associated with index from each underlying dataset.\n        Since index is in the range of [0, TotalNumInstances], we need to\n        map the index to the dataset before retrieving the item.\n        '
        return OrderedDict([(key, dataset[self._map_index_to_dataset(key, index)]) for (key, dataset) in self.datasets.items()])

    def collater(self, samples: List[Dict]):
        if False:
            i = 10
            return i + 15
        '\n        Generate a mini-batch for this dataset.\n        To convert this into a regular mini-batch we use the following\n        logic:\n            1. Select a dataset using the specified probability distribution.\n            2. Call the collater function of the selected dataset.\n        '
        if len(samples) == 0:
            return None
        selected_key = self.sampling_func(list(self.datasets.keys()))
        selected_samples = [sample[selected_key] for sample in samples]
        return self.datasets[selected_key].collater(selected_samples)

    def num_tokens(self, index: int):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return an example's length (number of tokens), used for batching. Here\n        we return the max across all examples at index across all underlying\n        datasets.\n        "
        return max((dataset.num_tokens(self._map_index_to_dataset(key, index)) for (key, dataset) in self.datasets.items()))

    def size(self, index: int):
        if False:
            while True:
                i = 10
        "\n        Return an example's size as a float or tuple. Here we return the max\n        across all underlying datasets. This value is used when filtering a\n        dataset with max-positions.\n        "
        return max((dataset.size(self._map_index_to_dataset(key, index)) for (key, dataset) in self.datasets.items()))

    @property
    def supports_prefetch(self):
        if False:
            print('Hello World!')
        return all((getattr(dataset, 'supports_prefetch', False) for dataset in self.datasets.values()))

    def prefetch(self, indices):
        if False:
            return 10
        for (key, dataset) in self.datasets.items():
            dataset.prefetch([self._map_index_to_dataset(key, index) for index in indices])

    @property
    def supports_fetch_outside_dataloader(self):
        if False:
            i = 10
            return i + 15
        return all((self.datasets[key].supports_fetch_outside_dataloader for key in self.datasets))