from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
from catalyst.contrib.data.dataset_ml import MetricLearningTrainDataset, QueryGalleryDataset
from catalyst.contrib.utils.image import imread

class Market1501MLDataset(MetricLearningTrainDataset):
    """
    Market1501 train dataset.
    This dataset should be used for training stage of the reid pipeline.

    .. _Scalable Person Re-identification\\: A Benchmark:
        https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf  # noqa: E501, W505
    """

    def __init__(self, root: str, transform: Optional[Callable[[torch.Tensor], torch.Tensor]]=None):
        if False:
            return 10
        '\n        Market1501 dataset for train stage of reid task.\n\n        Args:\n            root: path to a directory that contains Market-1501-v15.09.15\n            transform: transformation that should be applied to images\n        '
        self.root = Path(root)
        self._data_dir = self.root / 'Market-1501-v15.09.15/bounding_box_train'
        self.transform = transform
        (self.images, self.pids) = self._load_data(self._data_dir)

    @staticmethod
    def _load_data(data_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            return 10
        '\n        Load data from train directory of the dataset.\n        Parse names of images to get person id as labels.\n\n        Args:\n            data_dir: path to directory that contains training data\n\n        Returns:\n            images for training and their labels\n        '
        filenames = list(data_dir.glob('*.jpg'))
        data = torch.from_numpy(np.array([imread(filename) for filename in filenames])).float()
        targets = torch.from_numpy(np.array([int(filename.name.split('_')[0]) for filename in filenames]))
        return (data, targets)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if False:
            while True:
                i = 10
        'Get item from dataset.\n\n        Args:\n            index: index of the element\n\n        Returns:\n            dict of image and its pid\n        '
        (image, pid) = (self.images[index], self.pids[index])
        if self.transform is not None:
            image = self.transform(image)
        return {'image': image, 'pid': pid}

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        'Get len of the dataset'
        return len(self.pids)

    def get_labels(self) -> List[int]:
        if False:
            while True:
                i = 10
        'Get list of labels of dataset'
        return self.pids.tolist()

class Market1501QGDataset(QueryGalleryDataset):
    """Market1501QGDataset is a dataset for test stage of reid pipeline"""

    def __init__(self, root: str, transform: Optional[Callable[[torch.Tensor], torch.Tensor]]=None):
        if False:
            return 10
        '\n        Market1501 dataset for testing stage of reid task.\n\n        Args:\n            root: path to a directory that contains Market-1501-v15.09.15\n            transform: transformation that should be applied to images\n        '
        self.root = Path(root)
        self._gallery_dir = self.root / 'Market-1501-v15.09.15/bounding_box_test'
        self._query_dir = self.root / 'Market-1501-v15.09.15/query'
        self.transform = transform
        (query_data, query_pids, query_cids) = self._load_data(self._query_dir)
        (gallery_data, gallery_pids, gallery_cids) = self._load_data(self._gallery_dir)
        self._query_size = query_data.shape[0]
        self._gallery_size = gallery_data.shape[0]
        self.data = torch.cat([gallery_data, query_data])
        self.pids = np.concatenate([gallery_pids, query_pids])
        self.cids = np.concatenate([gallery_cids, query_cids])
        self._is_query = torch.cat([torch.zeros(size=(self._gallery_size,)), torch.ones(size=(self._query_size,))])

    @property
    def query_size(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Length of query part of the dataset\n\n        Returns:\n            query size\n        '
        return self._query_size

    @property
    def gallery_size(self) -> int:
        if False:
            print('Hello World!')
        '\n        Length of gallery part of the dataset\n\n        Returns:\n            gallery size\n        '
        return self._gallery_size

    @staticmethod
    def _load_data(data_dir: Path) -> Tuple[torch.Tensor, Iterable, Iterable]:
        if False:
            while True:
                i = 10
        'Load data from directory.\n\n        Parse names of images to get person ids as labels and camera ids.\n\n        Args:\n            data_dir: path to directory that contains data\n\n        Returns:\n            images, their labels and ids of the cameras that made the photos\n        '
        filenames = list(data_dir.glob('[!-]*.jpg'))
        data = torch.from_numpy(np.array([imread(filename) for filename in filenames])).float()
        pids = np.array([int(filename.name.split('_')[0]) for filename in filenames])
        cids = np.array([int(filename.name.split('_')[1][1:2]) for filename in filenames])
        return (data, pids, cids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if False:
            return 10
        'Get an item from dataset\n\n        Args:\n            index: index of the item to get\n\n        Returns:\n            dict of image, pid, cid and is_query flag\n            that shows if the image should be used as query or gallery sample.\n        '
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        item = {'image': img, 'pid': self.pids[index], 'cid': self.cids[index], 'is_query': self._is_query[index]}
        return item

    def __len__(self):
        if False:
            while True:
                i = 10
        'Get len of the dataset'
        return len(self.pids)
__all__ = ['Market1501MLDataset', 'Market1501QGDataset']