from typing import Any, List, Union
import numpy as np
from datasets import Dataset, IterableDataset, concatenate_datasets
from modelscope.metainfo import Models
from modelscope.msdatasets.dataset_cls.custom_datasets import CUSTOM_DATASETS, TorchCustomDataset
from modelscope.utils.constant import Tasks

@CUSTOM_DATASETS.register_module(module_name=Models.veco, group_key=Tasks.nli)
class VecoDataset(TorchCustomDataset):

    def __init__(self, datasets: Union[Any, List[Any]], mode, preprocessor=None, **kwargs):
        if False:
            print('Hello World!')
        self.seed = kwargs.get('seed', 42)
        self.permutation = None
        self.datasets = None
        super().__init__(datasets, mode, preprocessor, **kwargs)

    def switch_dataset(self, idx):
        if False:
            while True:
                i = 10
        'Switch dataset in evaluation.\n\n        Veco evaluates dataset one by one.\n\n        Args:\n            idx: The index of the dataset\n        '
        if self.mode == 'train':
            raise ValueError('Only support switch dataset in the evaluation loop')
        if idx >= len(self.datasets):
            raise ValueError('Index is bigger than the number of the datasets.')
        self._inner_dataset = self.datasets[idx]

    def __getitem__(self, item):
        if False:
            i = 10
            return i + 15
        if self.permutation is not None:
            item = self.permutation[item]
        return super().__getitem__(item)

    def prepare_dataset(self, datasets: Union[Any, List[Any]]) -> Any:
        if False:
            i = 10
            return i + 15
        "Compose all the datasets.\n\n        If the mode is 'train', all datasets will be mixed together, if the mode is 'eval',\n        the datasets will be kept and returns the first one.\n\n        Args:\n            datasets: The datasets to be composed.\n\n        Returns: The final dataset.\n        "
        if not isinstance(datasets, (list, tuple)):
            datasets = [datasets]
        if self.mode == 'train':
            if len(datasets) == 1:
                return datasets[0]
            elif all([isinstance(dataset, (Dataset, IterableDataset)) for dataset in datasets]):
                dataset = concatenate_datasets(list(datasets))
                return dataset.shuffle(seed=self.seed)
            else:
                generator = np.random.default_rng(self.seed)
                _len = sum([len(dataset) for dataset in datasets])
                self.permutation = generator.permutation(_len)
            return super().prepare_dataset(datasets)
        else:
            self.datasets = datasets
            return self.datasets[0]