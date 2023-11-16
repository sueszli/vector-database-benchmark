from typing import Iterable, Tuple
import os
from catalyst.contrib.data.dataset_cv import ImageFolderDataset
from catalyst.contrib.datasets.misc import download_and_extract_archive

class ImageClassificationDataset(ImageFolderDataset):
    """
    Base class for datasets with the following structure:

    .. code-block:: bash

        path/to/dataset/
        |-- train/
        |   |-- class1/  # folder of N images
        |   |   |-- train_image11
        |   |   |-- train_image12
        |   |   ...
        |   |   `-- train_image1N
        |   ...
        |   `-- classM/  # folder of K images
        |       |-- train_imageM1
        |       |-- train_imageM2
        |       ...
        |       `-- train_imageMK
        `-- val/
            |-- class1/  # folder of P images
            |   |-- val_image11
            |   |-- val_image12
            |   ...
            |   `-- val_image1P
            ...
            `-- classM/  # folder of T images
                |-- val_imageT1
                |-- val_imageT2
                ...
                `-- val_imageMT

    """
    name: str
    resources: Iterable[Tuple[str, str]] = None

    def __init__(self, root: str, train: bool=True, download: bool=False, **kwargs):
        if False:
            return 10
        'Constructor method for the ``ImageClassificationDataset`` class.\n\n        Args:\n            root: root directory of dataset\n            train: if ``True``, creates dataset from ``train/``\n                subfolder, otherwise from ``val/``\n            download: if ``True``, downloads the dataset from\n                the internet and puts it in root directory. If dataset\n                is already downloaded, it is not downloaded again\n            **kwargs: Keyword-arguments passed to ``super().__init__`` method.\n        '
        if download and (not os.path.exists(os.path.join(root, self.name))):
            os.makedirs(root, exist_ok=True)
            for (url, md5) in self.resources:
                filename = url.rpartition('/')[2]
                download_and_extract_archive(url, download_root=root, filename=filename, md5=md5)
        rootpath = os.path.join(root, self.name, 'train' if train else 'val')
        super().__init__(rootpath=rootpath, **kwargs)
__all__ = ['ImageClassificationDataset']