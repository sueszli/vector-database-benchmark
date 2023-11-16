"""
====================
Custom Task Tutorial
====================

Computer vision is an umbrella term for a wide spectrum of objectives models are trained for. These objective reflect
on the structure of the data and the possible actions on it.

The first step before running any Deepchecks checks is to create an implementation of
:class:`deepchecks.vision.vision_data.vision_data.VisionData`. Each implementation represents and standardize a computer vision task
and allows to run a more complex checks which relates to the given task's characteristics. There are default
base classes for a few known tasks like classification, object detection, and semantic segmentation however not all
tasks have a base implementation, meaning you will have to create your own task.

When creating your own task you will be limited to run checks which are agnostic to the specific task type.
For example performance checks that uses IOU works only on object detection tasks, since they need to know
the exact bounding box format in order to run, while other checks that uses
:ref:`vision__properties_guide` or custom metrics are agnostic to the task type.

In this guide we will implement a custom instance segmentation task and run checks on it.
Note that instance segmentation is different from semantic segmentation, which is currently supported in Deepchecks.

1. `Defining the Data <#defining-the-data>`__
2. `Implement Custom Task <#implement-custom-task>`__
3. `Implement Custom Properties <#implement-custom-properties>`__
4. `Implement Custom Metric <#implement-custom-metric>`__
"""
import contextlib
import os
import typing as t
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.utils import draw_segmentation_masks

class CocoInstanceSegmentationDataset(VisionDataset):
    """An instance of PyTorch VisionData the represents the COCO128-segments dataset.

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

    def __init__(self, root: str, name: str, train: bool=True, transforms: t.Optional[t.Callable]=None) -> None:
        if False:
            return 10
        super().__init__(root, transforms=transforms)
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

    def __getitem__(self, idx: int) -> t.Tuple[Image.Image, np.ndarray]:
        if False:
            while True:
                i = 10
        'Get the image and label at the given index.'
        image = Image.open(str(self.images[idx]))
        label_file = self.labels[idx]
        masks = []
        if label_file is not None:
            for label_str in label_file.open('r').read().strip().splitlines():
                label = np.array(label_str.split(), dtype=np.float32)
                class_id = int(label[0])
                coordinates = (label[1:].reshape(-1, 2) * np.array([image.width, image.height])).reshape(-1).tolist()
                mask = Image.new('L', (image.width, image.height), 0)
                ImageDraw.Draw(mask).polygon(coordinates, outline=1, fill=1)
                masks.append(np.array(mask, dtype=bool))
        if self.transforms is not None:
            transformed = self.transforms(image=np.array(image), masks=masks if masks else None)
            image = transformed['image']
            masks = transformed['masks']
            if masks:
                if isinstance(masks[0], np.ndarray):
                    masks = [torch.from_numpy(m) for m in masks]
                masks = torch.stack(masks)
            else:
                masks = torch.empty((0, 3))
        return (image, masks)

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.images)

    @classmethod
    def load_or_download(cls, root: Path, train: bool) -> 'CocoInstanceSegmentationDataset':
        if False:
            for i in range(10):
                print('nop')
        extract_dir = root / 'coco128segments'
        coco_dir = root / 'coco128segments' / 'coco128-seg'
        folder = 'train2017'
        if not coco_dir.exists():
            url = 'https://ultralytics.com/assets/coco128-segments.zip'
            with open(os.devnull, 'w', encoding='utf8') as f, contextlib.redirect_stdout(f):
                download_and_extract_archive(url, download_root=str(root), extract_root=str(extract_dir))
            try:
                os.remove('coco128/README.txt')
            except:
                pass
        return CocoInstanceSegmentationDataset(coco_dir, folder, train=train, transforms=A.Compose([ToTensorV2()]))
train_ds = CocoInstanceSegmentationDataset.load_or_download(Path('..'), train=True)
test_ds = CocoInstanceSegmentationDataset.load_or_download(Path('..'), train=False)
masked_images = [draw_segmentation_masks(train_ds[i][0], masks=train_ds[i][1], alpha=0.7) for i in range(5)]
(fig, axs) = plt.subplots(ncols=len(masked_images), figsize=(20, 20))
for (i, img) in enumerate(masked_images):
    img = img.detach()
    img = F.to_pil_image(img)
    axs[i].imshow(np.asarray(img))
    axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
fig.show()
from deepchecks.vision import VisionData, BatchOutputFormat

def deepchecks_collate_fn(batch) -> BatchOutputFormat:
    if False:
        i = 10
        return i + 15
    "Return a batch of images, labels and predictions for a batch of data. The expected format is a dictionary with\n    the following keys: 'images', 'labels' and 'predictions', each value is in the deepchecks format for the task.\n    You can also use the BatchOutputFormat class to create the output.\n    "
    batch = tuple(zip(*batch))
    images = [tensor.numpy().transpose((1, 2, 0)) for tensor in batch[0]]
    labels = batch[1]
    return BatchOutputFormat(images=images, labels=labels)
LABEL_MAP = {0: 'background', 1: 'airplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'dining table', 12: 'dog', 13: 'horse', 14: 'motorcycle', 15: 'person', 16: 'potted plant', 17: 'sheep', 18: 'couch', 19: 'train', 20: 'tv'}
train_loader = DataLoader(dataset=train_ds, batch_size=16, shuffle=False, collate_fn=deepchecks_collate_fn)
test_loader = DataLoader(dataset=test_ds, batch_size=16, shuffle=False, collate_fn=deepchecks_collate_fn)
train_vision_data = VisionData(batch_loader=train_loader, task_type='other', label_map=LABEL_MAP)
test_vision_data = VisionData(batch_loader=test_loader, task_type='other', label_map=LABEL_MAP)
from deepchecks.vision.checks import ImagePropertyDrift
result = ImagePropertyDrift().run(train_vision_data, test_vision_data)
result.show()
from deepchecks.vision.checks import LabelDrift

def number_of_detections(labels) -> t.List[int]:
    if False:
        while True:
            i = 10
    'Return a list containing the number of detections per sample in batch.'
    return [masks_per_image.shape[0] for masks_per_image in labels]
label_properties = [{'name': '# Detections per image', 'method': number_of_detections, 'output_type': 'numerical'}]
check = LabelDrift(label_properties=label_properties)
result = check.run(train_vision_data, test_vision_data)
result.show()