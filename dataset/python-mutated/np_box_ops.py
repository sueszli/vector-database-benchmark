"""Operations for [N, 4] numpy arrays representing bounding boxes.

Example box operations that are supported:
  * Areas: compute bounding box areas
  * IOU: pairwise intersection-over-union scores
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

def area(boxes):
    if False:
        while True:
            i = 10
    'Computes area of boxes.\n\n  Args:\n    boxes: Numpy array with shape [N, 4] holding N boxes\n\n  Returns:\n    a numpy array with shape [N*1] representing box areas\n  '
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def intersection(boxes1, boxes2):
    if False:
        return 10
    'Compute pairwise intersection areas between boxes.\n\n  Args:\n    boxes1: a numpy array with shape [N, 4] holding N boxes\n    boxes2: a numpy array with shape [M, 4] holding M boxes\n\n  Returns:\n    a numpy array with shape [N*M] representing pairwise intersection area\n  '
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)
    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(np.zeros(all_pairs_max_ymin.shape), all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(np.zeros(all_pairs_max_xmin.shape), all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths

def iou(boxes1, boxes2):
    if False:
        print('Hello World!')
    'Computes pairwise intersection-over-union between box collections.\n\n  Args:\n    boxes1: a numpy array with shape [N, 4] holding N boxes.\n    boxes2: a numpy array with shape [M, 4] holding N boxes.\n\n  Returns:\n    a numpy array with shape [N, M] representing pairwise iou scores.\n  '
    intersect = intersection(boxes1, boxes2)
    area1 = area(boxes1)
    area2 = area(boxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect
    return intersect / union

def ioa(boxes1, boxes2):
    if False:
        print('Hello World!')
    "Computes pairwise intersection-over-area between box collections.\n\n  Intersection-over-area (ioa) between two boxes box1 and box2 is defined as\n  their intersection area over box2's area. Note that ioa is not symmetric,\n  that is, IOA(box1, box2) != IOA(box2, box1).\n\n  Args:\n    boxes1: a numpy array with shape [N, 4] holding N boxes.\n    boxes2: a numpy array with shape [M, 4] holding N boxes.\n\n  Returns:\n    a numpy array with shape [N, M] representing pairwise ioa scores.\n  "
    intersect = intersection(boxes1, boxes2)
    areas = np.expand_dims(area(boxes2), axis=0)
    return intersect / areas