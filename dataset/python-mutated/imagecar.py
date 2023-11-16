from typing import Callable, Optional
from pathlib import Path
import requests
from torch.utils.data import Dataset
from catalyst.contrib.datasets.misc import _extract_archive
from catalyst.settings import SETTINGS
if SETTINGS.cv_required:
    import cv2
DATASET_IDX = '1lq6wOcxtIR3LnIARvlIBZBwJzL7h0FYc'
CHUNK_SIZE = 32768

def _download_file_from_google_drive(id, destination):
    if False:
        while True:
            i = 10
    url = 'https://docs.google.com/uc?export=download'
    session = requests.Session()
    response = session.get(url, params={'id': id}, stream=True)
    token = _get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
    _save_response_content(response, destination)

def _get_confirm_token(response):
    if False:
        print('Hello World!')
    for (key, value) in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def _save_response_content(response, destination):
    if False:
        i = 10
        return i + 15
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

class CarvanaOneCarDataset(Dataset):
    """
    The dataset contains images of cars and the corresponding binary masks for them
    """

    def __init__(self, root: str, train: bool=True, download: bool=False, transforms: Optional[Callable]=None):
        if False:
            print('Hello World!')
        "\n        Args:\n            root: str: root directory of dataset where\n            ``CarvanaOneCarDataset/`` exist.\n            train: (bool, optional): If True, creates dataset from\n                training part, otherwise from test part\n            download: (bool, optional): If true, downloads the dataset from\n                the internet and puts it in root directory. If dataset\n                is already downloaded, it is not downloaded again.\n            transforms: (callable, optional): A function/transform that\n                takes in an image and returns a transformed version.\n\n        Raises:\n            RuntimeError: If ``download is False`` and the dataset not found.\n\n        Examples:\n            >>> from catalyst.contrib.datasets import CarvanaOneCarDataset\n            >>> dataset = CarvanaOneCarDataset(root='./',\n            >>>                                train=True,\n            >>>                                download=True,\n            >>>                                transforms=None)\n            >>> image = dataset[0]['image']\n            >>> mask = dataset[0]['mask']\n        "
        directory = Path(root) / 'CarvanaOneCarDataset'
        if download and (not directory.exists()):
            _download_file_from_google_drive(DATASET_IDX, f'{root}/CarvanaOneCarDataset.zip')
            _extract_archive(f'{root}/CarvanaOneCarDataset.zip', f'{root}/', True)
        if not directory.exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
        split = 'train' if train else 'test'
        mask_path = directory / f'{split}_masks'
        image_path = directory / f'{split}_images'
        self.image_paths = sorted(image_path.glob('*.jpg'))
        self.mask_paths = sorted(mask_path.glob('*.png'))
        self.transforms = transforms

    def __len__(self) -> int:
        if False:
            return 10
        '\n        Returns:\n            int, dataset length\n        '
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            idx: Index\n\n        Returns:\n             Dict with 2 fields: ``image`` and ``mask``\n        '
        image_path = str(self.image_paths[idx])
        mask_path = str(self.mask_paths[idx])
        result = {'image': cv2.imread(image_path), 'mask': cv2.imread(mask_path, 2)}
        if self.transforms is not None:
            result = self.transforms(**result)
        return result
__all__ = ['CarvanaOneCarDataset']