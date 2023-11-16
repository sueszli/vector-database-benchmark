"""
.. _vision__property_label_correlation_change:

Property Label Correlation Change
***********************************

This notebook provides an overview for using and understanding the "Property Label Correlation Change" check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Run check on a Classification task <#run-the-check-on-a-classification-task>`__
* `Run check on an Object Detection task <#run-the-check-on-an-object-detection-task>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
=================================
The check estimates for every image :ref:` property <vision__properties_guide>`
(such as brightness, contrast etc.) its ability to predict the label by itself. This check can help find:

* A potential bias in one or both datasets, that leads to the labels being
  strongly correlated with simple image properties such as color, brightness,
  aspect ratio and more. This is a critical problem, that will likely stay hidden
  without this check (as it won't pop up when comparing model performance on train
  and test).

The check is based on calculating the predictive power score (PPS) of each image
property. For more details you can read here `how the PPS is calculated
<#how-is-the-predictive-power-score-pps-calculated>`__.

What is a problematic result?
-----------------------------

1. Image properties with a high predictive score - can indicate that there is a
   bias in the dataset, as a single property can predict the label successfully,
   using simple classic ML algorithms.

   This means that a deep learning algorithm may accidentally learn these properties
   instead of more accurate complex abstractions. For example, in a classification
   dataset of wolves and dogs photographs, if only wolves are photographed in the
   snow, the brightness of the image may be used to predict the label "wolf" easily.

   In this case, a model might not learn to discern wolf from dog by the animal's
   characteristics, but by using the background color.
2. A high difference between the PPS scores of a certain image property in the
   train and in the test datasets - this is an indication for a drift between
   the relation of the property and the label and a possible bias in one of
   the datasets.

   For example: an object detection dataset that identifies household items.
   In it, a pen would usually create a long and thin rectangle bounding box.
   If in the test dataset the pens would be angled differently, or other object
   are mistakenly identified as pens, the bounding boxes may have a different
   aspect ratio. In this case, the PPS of the train dataset will be high, while
   the PPS of the test dataset would be low, indicating that a bias in the train
   dataset does not appear in the test dataset, and could indicate the model will
   not be able to infer correctly on the test (or any other) dataset due to drift.

How do we calculate for different vision tasks?
-----------------------------------------------

* For classification tasks, this check uses PPS to predict the class by image properties.
* For object detection tasks, this check uses PPS to predict the class of each
  bounding box, by the image properties of that specific bounding box.
  This means that for each image, this check crops all the sub-images defined by bounding
  boxes, and uses them as inputs as though they were regular classification dataset images.

How is the Predictive Power Score (PPS) calculated?
---------------------------------------------------
The properties' predictive score results in a numeric score between 0 (feature has
no predictive power) and 1 (feature can fully predict the label alone).

The process of calculating the PPS is the following:
"""
import numpy as np
from deepchecks.vision.checks import PropertyLabelCorrelationChange
from deepchecks.vision.datasets.classification.mnist_torch import load_dataset
train_ds = load_dataset(train=True, object_type='VisionData')
test_ds = load_dataset(train=False, object_type='VisionData')

def generate_collate_function_with_leakage(collate_fn, mod):
    if False:
        i = 10
        return i + 15

    def collate_function_with_leakage(batch):
        if False:
            for i in range(10):
                print('nop')
        'Create function which inverse the data normalization.'
        batch_dict = collate_fn(batch)
        images = batch_dict['images']
        labels = batch_dict['labels']
        for (i, label) in enumerate(labels):
            if i % mod != 0:
                images[i] = np.ones(images[i].shape) * int(i % 3 + 1) * int(label)
        batch_dict['images'] = images
        return batch_dict
    return collate_function_with_leakage
train_ds._batch_loader.collate_fn = generate_collate_function_with_leakage(train_ds._batch_loader.collate_fn, 9)
test_ds._batch_loader.collate_fn = generate_collate_function_with_leakage(test_ds._batch_loader.collate_fn, 2)
check = PropertyLabelCorrelationChange()
result = check.run(train_ds, test_ds)
result.show()
from deepchecks.vision.datasets.detection.coco_torch import load_dataset
train_ds = load_dataset(train=True, object_type='VisionData')
test_ds = load_dataset(train=False, object_type='VisionData')

def generate_collate_function_with_leakage_coco(collate_fn, mod):
    if False:
        while True:
            i = 10

    def collate_function_with_leakage_coco(batch):
        if False:
            for i in range(10):
                print('nop')
        import numpy as np
        batch_dict = collate_fn(batch)
        images = batch_dict['images']
        labels = batch_dict['labels']
        ret = [np.array(x) for x in images]
        for (i, labels) in enumerate(labels):
            if i % mod != 0:
                for label in labels:
                    (x, y, w, h) = np.array(label[1:]).astype(int)
                    ret[i][y:y + h, x:x + w] = (ret[i][y:y + h, x:x + w] * int(label[0])).clip(min=200, max=255)
        batch_dict['images'] = ret
        return batch_dict
    return collate_function_with_leakage_coco
train_ds._batch_loader.collate_fn = generate_collate_function_with_leakage_coco(train_ds._batch_loader.collate_fn, 12)
test_ds._batch_loader.collate_fn = generate_collate_function_with_leakage_coco(test_ds._batch_loader.collate_fn, 2)
check = PropertyLabelCorrelationChange(per_class=False)
result = check.run(train_ds, test_ds)
result.show()
check = PropertyLabelCorrelationChange(per_class=False).add_condition_property_pps_difference_less_than(0.1).add_condition_property_pps_in_train_less_than()
result = check.run(train_dataset=train_ds, test_dataset=test_ds)
result.show(show_additional_outputs=False)