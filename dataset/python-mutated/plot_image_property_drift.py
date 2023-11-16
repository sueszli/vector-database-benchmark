"""
.. _vision__image_property_drift:

Image Property Drift
********************

This notebooks provides an overview for using and understanding the image property drift check.

**Structure:**

* `What Is Image Drift? <#what-is-image-drift>`__
* `Which Image Properties Are Used? <#which-image-properties-are-used>`__
* `Prepare data <#prepare-data>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__
* `Check Parameters <#check-parameters>`__

What Is Image Drift?
=================================
Drift is simply a change in the distribution of data over time, and it is
also one of the top reasons why machine learning model's performance degrades
over time.

Image drift is a data drift that occurs in images in the dataset.

For more information on drift, please visit our :ref:`drift_user_guide`.

How Deepchecks Detects Image Drift
------------------------------------

This check detects image property drift by using :ref:`univariate measures <drift_detection_by_univariate_measure>`
on each image property separately.
Another possible method for drift detection is by :ref:`a domain classifier <drift_detection_by_domain_classifier>`
which is used in the :ref:`Image Dataset Drift check <vision__image_dataset_drift>`.

Using Properties to Detect Image Drift
--------------------------------------------
In computer vision specifically, we can't measure drift on images directly, as the individual pixel has little
value when estimating drift. Therefore, we calculate drift on different
:ref:`properties of the image<vision__properties_guide>`,
on which we can directly measure drift.


Which Image Properties Are Used?
=================================
==============================  ==========
Property name                   What is it
==============================  ==========
Aspect Ratio                    Ratio between height and width of image (height / width)
Area                            Area of image in pixels (height * width)
Brightness                      Average intensity of image pixels. Color channels have different weights according to
                                RGB-to-Grayscale formula
RMS Contrast                    Contrast of image, calculated by standard deviation of pixels
Mean Red Relative Intensity     Mean over all pixels of the red channel, scaled to their relative intensity in
                                comparison to the other channels [r / (r + g + b)].
Mean Green Relative Intensity   Mean over all pixels of the green channel, scaled to their relative intensity in
                                comparison to the other channels [g / (r + g + b)].
Mean Blue Relative Intensity    Mean over all pixels of the blue channel, scaled to their relative intensity in
                                comparison to the other channels [b / (r + g + b)].
==============================  ==========

Imports
-------
"""
from deepchecks.vision.checks import ImagePropertyDrift
from deepchecks.vision.datasets.detection import coco_torch as coco
from deepchecks.vision.utils import image_properties
train_dataset = coco.load_dataset(train=True, object_type='VisionData')
test_dataset = coco.load_dataset(train=False, object_type='VisionData')
check_result = ImagePropertyDrift().run(train_dataset, test_dataset)
check_result
check_result.value
check_result = ImagePropertyDrift(classes_to_display=['person', 'traffic light'], min_samples=5).run(train_dataset, test_dataset)
check_result
check_result = ImagePropertyDrift().add_condition_drift_score_less_than(0.001).run(train_dataset, test_dataset)
check_result.show(show_additional_outputs=False)
from typing import List
import numpy as np

def area(images: List[np.ndarray]) -> List[int]:
    if False:
        print('Hello World!')
    return [img.shape[0] * img.shape[1] for img in images]

def aspect_ratio(images: List[np.ndarray]) -> List[float]:
    if False:
        for i in range(10):
            print('nop')
    return [img.shape[0] / img.shape[1] for img in images]
properties = [{'name': 'Area', 'method': area, 'output_type': 'numerical'}, {'name': 'Aspect Ratio', 'method': aspect_ratio, 'output_type': 'numerical'}]
check_result = ImagePropertyDrift(image_properties=properties, max_num_categories_for_drift=20).run(train_dataset, test_dataset)
check_result