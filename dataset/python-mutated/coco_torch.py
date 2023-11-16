"""Module for loading a sample of the COCO dataset and the yolov5s model."""
try:
    from torchvision.datasets import VisionDataset
except ImportError as error:
    raise ImportError('torchvision is not installed. Please install torchvision>=0.11.3 in order to use the selected dataset.') from error
import logging
import typing as t
import warnings
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from typing_extensions import Literal
from deepchecks.vision.datasets.assets.coco_detection.static_predictions_yolo import coco_detections_static_predictions_dict
from deepchecks.vision.datasets.detection.coco_utils import COCO_DIR, LABEL_MAP, download_coco128, get_image_and_label
from deepchecks.vision.utils.test_utils import get_data_loader_sequential, hash_image
from deepchecks.vision.vision_data import BatchOutputFormat, VisionData
__all__ = ['load_dataset', 'load_model', 'CocoDataset']
LOCAL_MODEL_PATH = COCO_DIR / 'yolov5-6.1'

def load_model(pretrained: bool=True, device: t.Union[str, torch.device]='cpu'):
    if False:
        i = 10
        return i + 15
    'Load the yolov5s (version 6.1)  model and return it.'
    dev = torch.device(device) if isinstance(device, str) else device
    torch.hub._validate_not_a_forked_repo = lambda *_: True
    logger = logging.getLogger('yolov5')
    logger.disabled = True
    if not LOCAL_MODEL_PATH.exists():
        repo = 'https://github.com/ultralytics/yolov5/archive/v6.1.zip'
        with urlopen(repo) as f:
            with zipfile.ZipFile(BytesIO(f.read())) as myzip:
                myzip.extractall(COCO_DIR)
    model = torch.hub.load(str(LOCAL_MODEL_PATH), 'yolov5s', source='local', pretrained=pretrained, verbose=False, device=dev)
    model.eval()
    logger.disabled = False
    return MockModel(model, dev)

def _batch_collate(batch):
    if False:
        i = 10
        return i + 15
    (imgs, labels) = zip(*batch)
    return (list(imgs), list(labels))

def collate_without_model(data) -> t.Tuple[t.List[np.ndarray], t.List[torch.Tensor]]:
    if False:
        return 10
    'Collate function for the coco dataset returning images and labels in correct format as tuple.'
    raw_images = [x[0] for x in data]
    images = [np.array(x) for x in raw_images]

    def move_class(tensor):
        if False:
            print('Hello World!')
        return torch.index_select(tensor, 1, torch.LongTensor([4, 0, 1, 2, 3]).to(tensor.device)) if len(tensor) > 0 else tensor
    labels = [move_class(x[1]) for x in data]
    return (images, labels)

def deepchecks_collate(model) -> t.Callable:
    if False:
        print('Hello World!')
    'Process batch to deepchecks format.\n\n    Parameters\n    ----------\n    model\n        model to predict with\n    Returns\n    -------\n    BatchOutputFormat\n        batch of data in deepchecks format\n    '

    def _process_batch_to_deepchecks_format(data) -> BatchOutputFormat:
        if False:
            while True:
                i = 10
        raw_images = [x[0] for x in data]
        images = [np.array(x) for x in raw_images]

        def move_class(tensor):
            if False:
                return 10
            return torch.index_select(tensor, 1, torch.LongTensor([4, 0, 1, 2, 3]).to(tensor.device)) if len(tensor) > 0 else tensor
        labels = [move_class(x[1]) for x in data]
        predictions = []
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
            raw_predictions: 'yolov5.models.common.Detections' = model(raw_images)
            for single_image_tensor in raw_predictions.pred:
                pred_modified = torch.clone(single_image_tensor)
                pred_modified[:, 2] = pred_modified[:, 2] - pred_modified[:, 0]
                pred_modified[:, 3] = pred_modified[:, 3] - pred_modified[:, 1]
                predictions.append(pred_modified)
        return BatchOutputFormat(images=images, labels=labels, predictions=predictions)
    return _process_batch_to_deepchecks_format

def load_dataset(train: bool=True, batch_size: int=32, num_workers: int=0, shuffle: bool=False, pin_memory: bool=True, object_type: Literal['VisionData', 'DataLoader']='DataLoader', n_samples: t.Optional[int]=None, device: t.Union[str, torch.device]='cpu') -> t.Union[DataLoader, VisionData]:
    if False:
        i = 10
        return i + 15
    "Get the COCO128 dataset and return a dataloader.\n\n    Parameters\n    ----------\n    train : bool, default: True\n        if `True` train dataset, otherwise test dataset\n    batch_size : int, default: 32\n        Batch size for the dataloader.\n    num_workers : int, default: 0\n        Number of workers for the dataloader.\n    shuffle : bool, default: False\n        Whether to shuffle the dataset.\n    pin_memory : bool, default: True\n        If ``True``, the data loader will copy Tensors\n        into CUDA pinned memory before returning them.\n    object_type : Literal['Dataset', 'DataLoader'], default: 'DataLoader'\n        type of the return value. If 'Dataset', :obj:`deepchecks.vision.VisionData`\n        will be returned, otherwise :obj:`torch.utils.data.DataLoader`\n    n_samples : int, optional\n        Only relevant for loading a VisionData. Number of samples to load. Return the first n_samples if shuffle\n        is False otherwise selects n_samples at random. If None, returns all samples.\n    device : t.Union[str, torch.device], default : 'cpu'\n        device to use in tensor calculations\n\n    Returns\n    -------\n    Union[DataLoader, VisionData]\n        A DataLoader or VisionData instance representing COCO128 dataset\n    "
    (coco_dir, dataset_name) = CocoDataset.download_coco128(COCO_DIR)
    dataset = CocoDataset(root=str(coco_dir), name=dataset_name, train=train, transforms=A.Compose([A.NoOp()], bbox_params=A.BboxParams(format='coco')))
    if object_type == 'DataLoader':
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=_batch_collate, pin_memory=pin_memory, generator=torch.Generator())
    elif object_type == 'VisionData':
        model = load_model(device=device)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=deepchecks_collate(model), pin_memory=pin_memory, generator=torch.Generator())
        dataloader = get_data_loader_sequential(dataloader, shuffle=shuffle, n_samples=n_samples)
        return VisionData(batch_loader=dataloader, label_map=LABEL_MAP, task_type='object_detection', reshuffle_data=False)
    else:
        raise TypeError(f'Unknown value of object_type - {object_type}')

class MockDetections:
    """Class which mocks YOLOv5 predictions object."""

    def __init__(self, dets):
        if False:
            i = 10
            return i + 15
        self.pred = dets

class MockModel:
    """Class of COCO model that returns cached predictions."""

    def __init__(self, real_model, device):
        if False:
            for i in range(10):
                print('nop')
        self.real_model = real_model
        self.device = device
        self.cache = coco_detections_static_predictions_dict

    def __call__(self, batch):
        if False:
            i = 10
            return i + 15
        results = []
        for img in batch:
            hash_key = hash_image(img)
            if hash_key not in self.cache:
                self.cache[hash_key] = self.real_model([img]).pred[0]
            results.append(self.cache[hash_key].to(self.device))
        return MockDetections(results)

class CocoDataset(VisionDataset):
    """An instance of PyTorch VisionData the represents the COCO128 dataset.

    Parameters
    ----------
    root : str
        Path to the root directory of the dataset.
    name : str
        Name of the dataset.
    train : bool
        if `True` train dataset, otherwise test dataset
    transform : Callable, optional
        A function/transforms that takes in an image and a label and returns the
        transformed versions of both.
        E.g, ``transforms.Rotate``
    target_transform : Callable, optional
        A function/transform that takes in the target and transforms it.
    transforms : Callable, optional
        A function/transform that takes in an PIL image and returns a transformed version.
        E.g, transforms.RandomCrop
    """
    TRAIN_FRACTION = 0.5

    def __init__(self, root: str, name: str, train: bool=True, transform: t.Optional[t.Callable]=None, target_transform: t.Optional[t.Callable]=None, transforms: t.Optional[t.Callable]=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(root, transforms, transform, target_transform)
        self.train = train
        self.root = Path(root).absolute()
        self.images_dir = Path(root) / 'images' / name
        self.labels_dir = Path(root) / 'labels' / name
        images: t.List[Path] = sorted(self.images_dir.glob('./*.jpg'))
        labels: t.List[t.Optional[Path]] = []
        for image in images:
            label = self.labels_dir / f'{image.stem}.txt'
            labels.append(label if label.exists() else None)
        assert len(images) != 0, 'Did not find folder with images or it was empty'
        assert not all((l is None for l in labels)), 'Did not find folder with labels or it was empty'
        train_len = int(self.TRAIN_FRACTION * len(images))
        if self.train is True:
            self.images = images[0:train_len]
            self.labels = labels[0:train_len]
        else:
            self.images = images[train_len:]
            self.labels = labels[train_len:]

    def __getitem__(self, idx: int) -> t.Tuple[Image.Image, torch.Tensor]:
        if False:
            while True:
                i = 10
        'Get the image and label at the given index.'
        (img, bboxes) = get_image_and_label(self.images[idx], self.labels[idx], self.transforms)
        if bboxes:
            bboxes = torch.stack([torch.tensor(x) for x in bboxes])
        else:
            bboxes = torch.tensor([])
        return (img, bboxes)

    def __len__(self) -> int:
        if False:
            return 10
        'Return the number of images in the dataset.'
        return len(self.images)

    @classmethod
    def download_coco128(cls, root: t.Union[str, Path]) -> t.Tuple[Path, str]:
        if False:
            while True:
                i = 10
        'Download coco128 and returns the root path and folder name.'
        return download_coco128(root)