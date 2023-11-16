"""
.. _vision__label_drift:

Label Drift
**********************

This notebooks provides an overview for using and understanding label drift check.

**Structure:**

* `What Is Label Drift? <#what-is-label-drift>`__
* `Which Label Properties Are Used? <#which-label-properties-are-used>`__
* `Run check on a Classification task <#run-the-check-on-a-classification-task-mnist>`__
* `Run check on an Object Detection task <#run-the-check-on-an-object-detection-task-coco>`__

What Is Label Drift?
========================
Drift is simply a change in the distribution of data over time, and it is
also one of the top reasons why machine learning model's performance degrades
over time.

Label drift is when drift occurs in the label itself.

For more information on drift, please visit our :ref:`drift_user_guide`.

How Deepchecks Detects Label Drift
------------------------------------

This check detects label drift by using :ref:`univariate measures <drift_detection_by_univariate_measure>`
on the label properties.

Using Label Properties to Detect Label Drift
--------------------------------------------
In computer vision specifically, our labels may be complex, and measuring their drift
is not a straightforward task. Therefore, we calculate drift on different
:ref:`properties of the label<vision__properties_guide>`,
on which we can directly measure drift.

Which Label Properties Are Used?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
================  ===================================  ==========
Task Type         Property name                        What is it
================  ===================================  ==========
Classification    Samples Per Class                    Number of images per class
Object Detection  Samples Per Class                    Number of bounding boxes per class
Object Detection  Bounding Box Area                    Area of bounding box (height * width)
Object Detection  Number of Bounding Boxes Per Image   Number of bounding box objects in each image
================  ===================================  ==========


Run the check on a Classification task (MNIST)
==============================================
Imports
-------
"""
from deepchecks.vision.checks import LabelDrift
from deepchecks.vision.datasets.classification.mnist_torch import load_dataset
train_ds = load_dataset(train=True, batch_size=64, object_type='VisionData')
test_ds = load_dataset(train=False, batch_size=1000, object_type='VisionData')
check = LabelDrift()
result = check.run(train_ds, test_ds)
result.show()
from deepchecks.vision.checks import ClassPerformance
ClassPerformance().run(train_ds, test_ds)
import numpy as np
np.random.seed(42)

def generate_collate_fn_with_label_drift(collate_fn):
    if False:
        print('Hello World!')

    def collate_fn_with_label_drift(batch):
        if False:
            while True:
                i = 10
        batch_dict = collate_fn(batch)
        images = batch_dict['images']
        labels = batch_dict['labels']
        for i in range(len(images)):
            (image, label) = (images[i], labels[i])
            if label == 0:
                if np.random.randint(5) != 0:
                    batch_dict['labels'][i] = 1
        return batch_dict
    return collate_fn_with_label_drift
mod_test_ds = load_dataset(train=False, batch_size=1000, object_type='VisionData')
mod_test_ds._batch_loader.collate_fn = generate_collate_fn_with_label_drift(mod_test_ds._batch_loader.collate_fn)
check = LabelDrift()
check.run(train_ds, mod_test_ds)
check = LabelDrift().add_condition_drift_score_less_than()
check.run(train_ds, mod_test_ds)
ClassPerformance().run(train_ds, mod_test_ds)
from deepchecks.vision.datasets.detection.coco_torch import load_dataset
train_ds = load_dataset(train=True, object_type='VisionData')
test_ds = load_dataset(train=False, object_type='VisionData')
check = LabelDrift()
check.run(train_ds, test_ds)