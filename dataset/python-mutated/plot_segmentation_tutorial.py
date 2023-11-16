"""
.. _vision__segmentation_tutorial:

===============================
Semantic Segmentation Tutorial
===============================

In this tutorial, you will learn how to validate your **semantic segmentation model** using deepchecks test suites.
You can read more about the different checks and suites for computer vision use cases at the
:ref:`examples section <vision__checks_gallery>`.

If you just want to see the output of this tutorial, jump to :ref:`observing_the_result` section.

A semantic segmentation task is a task where every pixel of the image is labeled with a single class.
Therefore, a common output of these tasks is an image of identical size to the input, with a vector for each pixel
of the probability for each class.

.. code-block:: bash

    # Before we start, if you don't have deepchecks vision package installed yet, run:
    import sys
    !{sys.executable} -m pip install "deepchecks[vision]" --quiet --upgrade # --user

    # or install using pip from your python environment
"""
from deepchecks.vision.datasets.segmentation.segmentation_coco import CocoSegmentationDataset, load_model
train_dataset = CocoSegmentationDataset.load_or_download(train=True)
test_dataset = CocoSegmentationDataset.load_or_download(train=False)
print(f'Number of training images: {len(train_dataset)}')
print(f'Number of test images: {len(test_dataset)}')
print(f'Example output of an image shape: {train_dataset[0][0].shape}')
print(f'Example output of a label shape: {train_dataset[0][1].shape}')
model = load_model(pretrained=True)
import torch
import torchvision.transforms.functional as F
from deepchecks.vision.vision_data import BatchOutputFormat

def deepchecks_collate_fn(batch) -> BatchOutputFormat:
    if False:
        i = 10
        return i + 15
    "Return a batch of images, labels and predictions for a batch of data. The expected format is a dictionary with\n    the following keys: 'images', 'labels' and 'predictions', each value is in the deepchecks format for the task.\n    You can also use the BatchOutputFormat class to create the output.\n    "
    batch = tuple(zip(*batch))
    images = [tensor.numpy().transpose((1, 2, 0)) for tensor in batch[0]]
    labels = batch[1]
    normalized_batch = [F.normalize(img.unsqueeze(0).float() / 255, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for img in batch[0]]
    predictions = [model(img)['out'].squeeze(0).detach() for img in normalized_batch]
    predictions = [torch.nn.functional.softmax(pred, dim=0) for pred in predictions]
    return BatchOutputFormat(images=images, labels=labels, predictions=predictions)
LABEL_MAP = {0: 'background', 1: 'airplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'dining table', 12: 'dog', 13: 'horse', 14: 'motorcycle', 15: 'person', 16: 'potted plant', 17: 'sheep', 18: 'couch', 19: 'train', 20: 'tv'}
from torch.utils.data import DataLoader
from deepchecks.vision import VisionData
train_loader = DataLoader(dataset=train_dataset, shuffle=True, collate_fn=deepchecks_collate_fn)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, collate_fn=deepchecks_collate_fn)
training_data = VisionData(batch_loader=train_loader, task_type='semantic_segmentation', label_map=LABEL_MAP)
test_data = VisionData(batch_loader=test_loader, task_type='semantic_segmentation', label_map=LABEL_MAP)
training_data.head()
from deepchecks.vision.suites import model_evaluation
suite = model_evaluation()
result = suite.run(training_data, test_data)
result.save_as_html('output.html')
result.show()