"""Module for loading a sample of the COCO dataset and the yolov5s model."""
try:
    from torchvision.datasets import VisionDataset
    from torchvision.datasets.utils import download_and_extract_archive
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large
except ImportError as error:
    raise ImportError('torchvision is not installed. Please install torchvision>=0.11.3 in order to use the selected dataset.') from error
import contextlib
import os
import pathlib
import typing as t
from pathlib import Path
import albumentations as A
import numpy as np
import torch
import torchvision.transforms.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader
from typing_extensions import Literal
from deepchecks.vision.utils.test_utils import get_data_loader_sequential
from deepchecks.vision.vision_data import BatchOutputFormat, VisionData
__all__ = ['load_dataset', 'load_model', 'CocoSegmentationDataset']
from deepchecks.vision.vision_data.utils import object_to_numpy
DATA_DIR = pathlib.Path(__file__).absolute().parent.parent / 'assets' / 'coco_segmentation'

def load_model(pretrained: bool=True) -> nn.Module:
    if False:
        print('Hello World!')
    'Load the lraspp_mobilenet_v3_large model and return it.'
    model = lraspp_mobilenet_v3_large(pretrained=pretrained, progress=False)
    _ = model.eval()
    return model

def _batch_collate(batch):
    if False:
        for i in range(10):
            print('nop')
    'Get list of samples from `CocoSegmentDataset` and combine them to a batch.'
    (images, masks) = zip(*batch)
    return (list(images), list(masks))

def deepchecks_collate(model) -> t.Callable:
    if False:
        return 10
    'Process batch to deepchecks format.\n\n    Parameters\n    ----------\n    model : nn.Module\n        model to predict with\n    Returns\n    -------\n    BatchOutputFormat\n        batch of data in deepchecks format\n    '

    def _process_batch_to_deepchecks_format(data) -> BatchOutputFormat:
        if False:
            return 10
        raw_images = [x[0] for x in data]
        images = [object_to_numpy(tensor).transpose((1, 2, 0)) for tensor in raw_images]
        labels = [x[1] for x in data]
        normalized_batch = [F.normalize(img.unsqueeze(0).float() / 255, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for img in raw_images]
        predictions = [model(img)['out'].squeeze(0).detach() for img in normalized_batch]
        predictions = [torch.nn.functional.softmax(pred, dim=0) for pred in predictions]
        return {'images': images, 'labels': labels, 'predictions': predictions}
    return _process_batch_to_deepchecks_format

def load_dataset(train: bool=True, batch_size: int=32, num_workers: int=0, shuffle: bool=True, pin_memory: bool=True, object_type: Literal['VisionData', 'DataLoader']='VisionData', test_mode: bool=False) -> t.Union[DataLoader, VisionData]:
    if False:
        for i in range(10):
            print('nop')
    'Get the COCO128 dataset and return a dataloader.\n\n    Parameters\n    ----------\n    train : bool, default: True\n        if `True` train dataset, otherwise test dataset\n    batch_size : int, default: 32\n        Batch size for the dataloader.\n    num_workers : int, default: 0\n        Number of workers for the dataloader.\n    shuffle : bool, default: True\n        Whether to shuffle the dataset.\n    pin_memory : bool, default: True\n        If ``True``, the data loader will copy Tensors\n        into CUDA pinned memory before returning them.\n    object_type : Literal[\'Dataset\', \'DataLoader\'], default: \'DataLoader\'\n        type of the return value. If \'Dataset\', :obj:`deepchecks.vision.VisionDataset`\n        will be returned, otherwise :obj:`torch.utils.data.DataLoader`\n    test_mode: bool, default False\n        whether to load this dataset in "test_mode", meaning very minimal number of images in order to use for\n        unittests.\n\n    Returns\n    -------\n    Union[DataLoader, VisionDataset]\n\n        A DataLoader or VisionDataset instance representing COCO128 dataset\n    '
    root = DATA_DIR
    dataset = CocoSegmentationDataset.load_or_download(root=root, train=train, test_mode=test_mode)
    if object_type == 'DataLoader':
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=_batch_collate, pin_memory=pin_memory, generator=torch.Generator())
    elif object_type == 'VisionData':
        model = load_model()
        loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=deepchecks_collate(model), pin_memory=pin_memory, generator=torch.Generator())
        loader = get_data_loader_sequential(loader, shuffle=shuffle)
        return VisionData(batch_loader=loader, task_type='semantic_segmentation', label_map=LABEL_MAP, reshuffle_data=False)
    else:
        raise TypeError(f'Unknown value of object_type - {object_type}')

class CocoSegmentationDataset(VisionDataset):
    """An instance of PyTorch VisionData the represents the COCO128-segments dataset.

    Uses only the 21 categories used also by Pascal-VOC, in order to match the model supplied in this file,
    torchvision's deeplabv3_mobilenet_v3_large.

    Parameters
    ----------
    root : str
        Path to the root directory of the dataset.
    name : str
        Name of the dataset.
    train : bool
        if `True` train dataset, otherwise test dataset
    transforms : Callable, optional
        A function/transform that takes in an PIL image and returns a transformed version.
        E.g, transforms.RandomCrop
    """
    TRAIN_FRACTION = 0.5

    def __init__(self, root: str, name: str, train: bool=True, transforms: t.Optional[t.Callable]=None, test_mode: bool=False) -> None:
        if False:
            return 10
        super().__init__(root, transforms=transforms)
        self.train = train
        self.root = Path(root).absolute()
        self.images_dir = Path(root) / 'images' / name
        self.labels_dir = Path(root) / 'labels' / name
        all_images: t.List[Path] = sorted(self.images_dir.glob('./*.jpg'))
        images: t.List[Path] = []
        labels: t.List[t.Optional[Path]] = []
        for i in range(len(all_images)):
            label = self.labels_dir / f'{all_images[i].stem}.txt'
            if label.exists():
                polygons = label.open('r').read().strip().splitlines()
                relevant_labels = [polygon.split()[0] for polygon in polygons]
                relevant_labels = [class_id for class_id in relevant_labels if int(class_id) in COCO_TO_PASCAL_VOC]
                if len(relevant_labels) > 0:
                    images.append(all_images[i])
                    labels.append(label)
        assert len(images) != 0, 'Did not find folder with images or it was empty'
        assert not all((l is None for l in labels)), 'Did not find folder with labels or it was empty'
        train_len = int(self.TRAIN_FRACTION * len(images))
        if test_mode is True:
            if self.train is True:
                self.images = images[0:5] * 2
                self.labels = labels[0:5] * 2
            else:
                self.images = images[1:6] * 2
                self.labels = labels[1:6] * 2
        elif self.train is True:
            self.images = images[0:train_len]
            self.labels = labels[0:train_len]
        else:
            self.images = images[train_len:]
            self.labels = labels[train_len:]

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        'Get the image and label at the given index.'
        image = Image.open(str(self.images[idx]))
        label_file = self.labels[idx]
        masks = []
        classes = []
        if label_file is not None:
            for label_str in label_file.open('r').read().strip().splitlines():
                label = np.array(label_str.split(), dtype=np.float32)
                class_id = int(label[0])
                if class_id in COCO_TO_PASCAL_VOC:
                    coordinates = (label[1:].reshape(-1, 2) * np.array([image.width, image.height])).reshape(-1).tolist()
                    mask = Image.new('L', (image.width, image.height), 0)
                    ImageDraw.Draw(mask).polygon(coordinates, outline=1, fill=1)
                    masks.append(np.array(mask, dtype=bool))
                    classes.append(COCO_TO_PASCAL_VOC[class_id])
        if self.transforms is not None:
            transformed = self.transforms(image=np.array(image), masks=masks)
            image = transformed['image']
            masks = transformed['masks']
            if masks:
                if isinstance(masks[0], np.ndarray):
                    masks = [torch.from_numpy(m) for m in masks]
                masks = torch.stack(masks)
            else:
                masks = torch.empty((0, 3))
        if image.shape[0] == 1:
            image = torch.stack([image[0], image[0], image[0]])
        ret_label = np.zeros((image.shape[1], image.shape[2]))
        ret_label_mask = np.zeros(ret_label.shape)
        for i in range(len(classes)):
            mask = np.logical_and(np.logical_not(ret_label_mask), np.array(masks[i]))
            ret_label_mask = np.logical_or(ret_label_mask, mask)
            ret_label += classes[i] * mask
        return (image, torch.as_tensor(ret_label))

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the number of images in the dataset.'
        return len(self.images)

    @classmethod
    def load_or_download(cls, train: bool, root: Path=DATA_DIR, test_mode: bool=False) -> 'CocoSegmentationDataset':
        if False:
            while True:
                i = 10
        'Load or download the coco128 dataset with segment annotations.'
        extract_dir = root / 'coco128segments'
        coco_dir = root / 'coco128segments' / 'coco128-seg'
        folder = 'train2017'
        if not coco_dir.exists():
            url = 'https://ndownloader.figshare.com/files/37650656'
            with open(os.devnull, 'w', encoding='utf8') as f, contextlib.redirect_stdout(f):
                download_and_extract_archive(url, download_root=str(root), extract_root=str(extract_dir), filename='coco128-segments.zip')
            try:
                os.remove('coco128segments/coco128/README.txt')
            except:
                pass
        return CocoSegmentationDataset(coco_dir, folder, train=train, transforms=A.Compose([ToTensorV2()]), test_mode=test_mode)
_ORIG_LABEL_MAP = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
LABEL_MAP = {0: 'background', 1: 'airplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'dining table', 12: 'dog', 13: 'horse', 14: 'motorcycle', 15: 'person', 16: 'potted plant', 17: 'sheep', 18: 'couch', 19: 'train', 20: 'tv'}
COCO_TO_PASCAL_VOC = {4: 1, 1: 2, 14: 3, 8: 4, 39: 5, 5: 6, 2: 7, 15: 8, 56: 9, 19: 10, 60: 11, 16: 12, 17: 13, 3: 14, 0: 15, 58: 16, 18: 17, 57: 18, 6: 19, 62: 20}