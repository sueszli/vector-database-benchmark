"""
.. _vision__detection_tutorial:

==========================
Object Detection Tutorial
==========================

In this tutorial, you will learn how to validate your **object detection model** using deepchecks test suites.
You can read more about the different checks and suites for computer vision use cases at the
:ref:`examples section <vision__checks_gallery>`.

If you just want to see the output of this tutorial, jump to the :ref:`observing the results <vision_segmentation_tutorial__observing_the_result>` section.

An object detection tasks usually consist of two parts:

- Object Localization, where the model predicts the location of an object in the image,
- Object Classification, where the model predicts the class of the detected object.

The common output of an object detection model is a list of bounding boxes around the objects, and
their classes.

.. code-block:: bash

    # Before we start, if you don't have deepchecks vision package installed yet, run:
    import sys
    !{sys.executable} -m pip install "deepchecks[vision]" --quiet --upgrade # --user

    # or install using pip from your python environment
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import xml.etree.ElementTree as ET
import urllib.request
import zipfile
url = 'https://figshare.com/ndownloader/files/34488599'
urllib.request.urlretrieve(url, './tomato-detection.zip')
with zipfile.ZipFile('./tomato-detection.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

class TomatoDataset(Dataset):

    def __init__(self, root, transforms):
        if False:
            return 10
        self.root = root
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(root, 'images'))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, 'annotations'))))

    def __getitem__(self, idx):
        if False:
            for i in range(10):
                print('nop')
        img_path = os.path.join(self.root, 'images', self.images[idx])
        ann_path = os.path.join(self.root, 'annotations', self.annotations[idx])
        img = Image.open(img_path).convert('RGB')
        (bboxes, labels) = ([], [])
        with open(ann_path, 'r') as f:
            root = ET.parse(f).getroot()
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                if int(difficult) == 1:
                    continue
                cls_id = 1
                xmlbox = obj.find('bndbox')
                b = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text)]
                bboxes.append(b)
                labels.append(cls_id)
        bboxes = torch.as_tensor(np.array(bboxes), dtype=torch.float32)
        labels = torch.as_tensor(np.array(labels), dtype=torch.int64)
        if self.transforms is not None:
            res = self.transforms(image=np.array(img), bboxes=bboxes, class_labels=labels)
        target = {'boxes': [torch.Tensor(x) for x in res['bboxes']], 'labels': res['class_labels']}
        img = res['image']
        return (img, target)

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.images)
data_transforms = A.Compose([A.Resize(height=256, width=256), A.CenterCrop(height=224, width=224), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
dataset = TomatoDataset(root='./tomato-detection/data', transforms=data_transforms)
(train_dataset, test_dataset) = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)], generator=torch.Generator().manual_seed(42))
test_dataset.transforms = A.Compose([ToTensorV2()])
print(f'Number of training images: {len(train_dataset)}')
print(f'Number of test images: {len(test_dataset)}')
print(f'Example output of an image shape: {train_dataset[0][0].shape}')
print(f'Example output of a label: {train_dataset[0][1]}')
from functools import partial
from torch import nn
import torchvision
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
num_anchors = model.anchor_generator.num_anchors_per_location()
norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, 2, norm_layer)
_ = model.to(device)
model.load_state_dict(torch.load('./tomato-detection/ssd_model.pth'))
_ = model.eval()

def get_untransformed_images(original_images):
    if False:
        return 10
    '\n    Convert a batch of data to images in the expected format. The expected format is an iterable of images,\n    where each image is a numpy array of shape (height, width, channels). The numbers in the array should be in the\n    range [0, 255] in a uint8 format.\n    '
    inp = torch.stack(list(original_images)).cpu().detach().numpy().transpose((0, 2, 3, 1))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp * 255

def transform_labels_to_cxywh(original_labels):
    if False:
        print('Hello World!')
    '\n    Convert a batch of data to labels in the expected format. The expected format is an iterator of arrays, each array\n    corresponding to a sample. Each array element is in a shape of [B, 5], where B is the number of bboxes\n    in the image, and each bounding box is in the structure of [class_id, x, y, w, h].\n    '
    label = []
    for annotation in original_labels:
        if len(annotation['boxes']):
            bbox = torch.stack(annotation['boxes'])
            bbox[:, 2:] = bbox[:, 2:] - bbox[:, :2]
            label.append(torch.concat([torch.stack(annotation['labels']).reshape((-1, 1)), bbox], dim=1))
        else:
            label.append(torch.tensor([]))
    return label

def infer_on_images(original_images):
    if False:
        i = 10
        return i + 15
    '\n    Returns the predictions for a batch of data. The expected format is an iterator of arrays, each array\n    corresponding to a sample. Each array element is in a shape of [B, 6], where B is the number of bboxes in the\n    predictions, and each bounding box is in the structure of [x, y, w, h, score, class_id].\n\n    Note that model and device here are global variables, and are defined in the previous code block, as the collate\n    function cannot recieve other arguments than the batch.\n    '
    nm_thrs = 0.2
    score_thrs = 0.7
    imgs = list((img.to(device) for img in original_images))
    with torch.no_grad():
        preds = model(imgs)
    processed_pred = []
    for pred in preds:
        keep_boxes = torchvision.ops.nms(pred['boxes'], pred['scores'], nm_thrs)
        score_filter = pred['scores'][keep_boxes] > score_thrs
        test_boxes = pred['boxes'][keep_boxes][score_filter].reshape((-1, 4))
        test_boxes[:, 2:] = test_boxes[:, 2:] - test_boxes[:, :2]
        test_labels = pred['labels'][keep_boxes][score_filter]
        test_scores = pred['scores'][keep_boxes][score_filter]
        processed_pred.append(torch.concat([test_boxes, test_scores.reshape((-1, 1)), test_labels.reshape((-1, 1))], dim=1))
    return processed_pred
from deepchecks.vision.vision_data import BatchOutputFormat

def deepchecks_collate_fn(batch) -> BatchOutputFormat:
    if False:
        for i in range(10):
            print('nop')
    'Return a batch of images, labels and predictions in the deepchecks format.'
    batch = tuple(zip(*batch))
    images = get_untransformed_images(batch[0])
    labels = transform_labels_to_cxywh(batch[1])
    predictions = infer_on_images(batch[0])
    return BatchOutputFormat(images=images, labels=labels, predictions=predictions)
LABEL_MAP = {1: 'Tomato'}
from deepchecks.vision.vision_data import VisionData
train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=deepchecks_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=deepchecks_collate_fn)
training_data = VisionData(batch_loader=train_loader, task_type='object_detection', label_map=LABEL_MAP)
test_data = VisionData(batch_loader=test_loader, task_type='object_detection', label_map=LABEL_MAP)
training_data.head()
from deepchecks.vision.suites import model_evaluation
suite = model_evaluation()
result = suite.run(training_data, test_data)
result.save_as_html('output.html')
result