"""
.. _vision__classification_tutorial:

==============================================
Image Classification Tutorial
==============================================

In this tutorial, you will learn how to validate your **classification model** using deepchecks test suites.
You can read more about the different checks and suites for computer vision use cases at the
:ref:`examples section  <vision__checks_gallery>`.

A classification model is usually used to classify an image into one of a number of classes. Although there are
multi label use-cases, in which the model is used to classify an image into multiple classes, most use-cases
require the model to classify images into a single class.
Currently, deepchecks supports only single label classification (either binary or multi-class).

.. code-block:: bash

    # Before we start, if you don't have deepchecks vision package installed yet, run:
    import sys
    !{sys.executable} -m pip install "deepchecks[vision]" --quiet --upgrade # --user

    # or install using pip from your python environment
"""
import os
import urllib.request
import zipfile
url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
urllib.request.urlretrieve(url, './hymenoptera_data.zip')
with zipfile.ZipFile('./hymenoptera_data.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
import albumentations as A
import numpy as np
import PIL.Image
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader

class AntsBeesDataset(torchvision.datasets.ImageFolder):

    def __getitem__(self, index: int):
        if False:
            return 10
        'overrides __getitem__ to be compatible to albumentations'
        (path, target) = self.samples[index]
        sample = self.loader(path)
        sample = self.get_cv2_image(sample)
        if self.transforms is not None:
            transformed = self.transforms(image=sample, target=target)
            (sample, target) = (transformed['image'], transformed['target'])
        else:
            if self.transform is not None:
                sample = self.transform(image=sample)['image']
            if self.target_transform is not None:
                target = self.target_transform(target)
        return (sample, target)

    def get_cv2_image(self, image):
        if False:
            print('Hello World!')
        if isinstance(image, PIL.Image.Image):
            return np.array(image).astype('uint8')
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise RuntimeError('Only PIL.Image and CV2 loaders currently supported!')
data_dir = './hymenoptera_data'
data_transforms = A.Compose([A.Resize(height=256, width=256), A.CenterCrop(height=224, width=224), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
train_dataset = AntsBeesDataset(root=os.path.join(data_dir, 'train'))
train_dataset.transforms = data_transforms
test_dataset = AntsBeesDataset(root=os.path.join(data_dir, 'val'))
test_dataset.transforms = data_transforms
print(f'Number of training images: {len(train_dataset)}')
print(f'Number of validation images: {len(test_dataset)}')
print(f'Example output of an image shape: {train_dataset[0][0].shape}')
print(f'Example output of a label: {train_dataset[0][1]}')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)
_ = model.eval()
from deepchecks.vision.vision_data import BatchOutputFormat

def deepchecks_collate_fn(batch) -> BatchOutputFormat:
    if False:
        return 10
    "Return a batch of images, labels and predictions for a batch of data. The expected format is a dictionary with\n    the following keys: 'images', 'labels' and 'predictions', each value is in the deepchecks format for the task.\n    You can also use the BatchOutputFormat class to create the output.\n    "
    batch = tuple(zip(*batch))
    inp = torch.stack(batch[0]).detach().numpy().transpose((0, 2, 3, 1))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inp = std * inp + mean
    images = np.clip(inp, 0, 1) * 255
    labels = batch[1]
    logits = model.to(device)(torch.stack(batch[0]).to(device))
    predictions = nn.Softmax(dim=1)(logits)
    return BatchOutputFormat(images=images, labels=labels, predictions=predictions)
LABEL_MAP = {0: 'ants', 1: 'bees'}
from deepchecks.vision import VisionData
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=deepchecks_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=deepchecks_collate_fn)
training_data = VisionData(batch_loader=train_loader, task_type='classification', label_map=LABEL_MAP)
test_data = VisionData(batch_loader=test_loader, task_type='classification', label_map=LABEL_MAP)
training_data.head()
from deepchecks.vision.suites import train_test_validation
suite = train_test_validation()
result = suite.run(training_data, test_data, max_samples=5000)
result.save_as_html('output.html')
result