from typing import Any, List, Union
import torch.utils.data
from torch.utils.data import ConcatDataset as TorchConcatDataset
from modelscope.utils.constant import ModeKeys

class TorchCustomDataset(torch.utils.data.Dataset):
    """The custom dataset base class for all the torch-based task processors.
    """

    def __init__(self, datasets: Union[Any, List[Any]], mode=ModeKeys.TRAIN, preprocessor=None, **kwargs):
        if False:
            return 10
        self.trainer = None
        self.mode = mode
        self.preprocessor = preprocessor
        self._inner_dataset = self.prepare_dataset(datasets)

    def __getitem__(self, index) -> Any:
        if False:
            i = 10
            return i + 15
        return self.preprocessor(self._inner_dataset[index]) if self.preprocessor else self._inner_dataset[index]

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._inner_dataset)

    def prepare_dataset(self, datasets: Union[Any, List[Any]]) -> Any:
        if False:
            for i in range(10):
                print('nop')
        'Prepare a dataset.\n\n        User can process the input datasets in a whole dataset perspective.\n        This method gives a default implementation of datasets merging, user can override this\n        method to write custom logics.\n\n        Args:\n            datasets: The original dataset(s)\n\n        Returns: A single dataset, which may be created after merging.\n\n        '
        if isinstance(datasets, List):
            if len(datasets) == 1:
                return datasets[0]
            elif len(datasets) > 1:
                return TorchConcatDataset(datasets)
        else:
            return datasets