from typing import Dict, List
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset

class MetricLearningTrainDataset(Dataset, ABC):
    """
    Base class for datasets adapted for
    metric learning train stage.
    """

    @abstractmethod
    def get_labels(self) -> List[int]:
        if False:
            i = 10
            return i + 15
        '\n        Dataset for metric learning must provide\n        label of each sample for forming positive\n        and negative pairs during the training\n        based on these labels.\n\n        Raises:\n            NotImplementedError: You should implement it  # noqa: DAR402\n        '
        raise NotImplementedError()

class QueryGalleryDataset(Dataset, ABC):
    """
    QueryGallleryDataset for CMCScoreCallback
    """

    @abstractmethod
    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        if False:
            return 10
        '\n        Dataset for query/gallery split should\n        return dict with `feature`, `targets` and\n        `is_query` key. Value by key `is_query` should\n        be boolean and indicate whether current object\n        is in query or in gallery.\n\n        Args:\n            item: Item\n\n        Raises:\n            NotImplementedError: You should implement it  # noqa: DAR402\n        '
        raise NotImplementedError()

    @property
    @abstractmethod
    def query_size(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Query/Gallery dataset should have property\n        query size.\n\n        Returns:\n            query size  # noqa: DAR202\n\n        Raises:\n            NotImplementedError: You should implement it  # noqa: DAR402\n        '
        raise NotImplementedError()

    @property
    @abstractmethod
    def gallery_size(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Query/Gallery dataset should have property\n        gallery size.\n\n        Returns:\n            gallery size  # noqa: DAR202\n\n        Raises:\n            NotImplementedError: You should implement it  # noqa: DAR402\n        '
        raise NotImplementedError()
__all__ = ['MetricLearningTrainDataset', 'QueryGalleryDataset']