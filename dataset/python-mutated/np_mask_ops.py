"""Operations for [N, height, width] numpy arrays representing masks.

Example mask operations that are supported:
  * Areas: compute mask areas
  * IOU: pairwise intersection-over-union scores
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
EPSILON = 1e-07

def area(masks):
    if False:
        i = 10
        return i + 15
    'Computes area of masks.\n\n  Args:\n    masks: Numpy array with shape [N, height, width] holding N masks. Masks\n      values are of type np.uint8 and values are in {0,1}.\n\n  Returns:\n    a numpy array with shape [N*1] representing mask areas.\n\n  Raises:\n    ValueError: If masks.dtype is not np.uint8\n  '
    if masks.dtype != np.uint8:
        raise ValueError('Masks type should be np.uint8')
    return np.sum(masks, axis=(1, 2), dtype=np.float32)

def intersection(masks1, masks2):
    if False:
        print('Hello World!')
    'Compute pairwise intersection areas between masks.\n\n  Args:\n    masks1: a numpy array with shape [N, height, width] holding N masks. Masks\n      values are of type np.uint8 and values are in {0,1}.\n    masks2: a numpy array with shape [M, height, width] holding M masks. Masks\n      values are of type np.uint8 and values are in {0,1}.\n\n  Returns:\n    a numpy array with shape [N*M] representing pairwise intersection area.\n\n  Raises:\n    ValueError: If masks1 and masks2 are not of type np.uint8.\n  '
    if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
        raise ValueError('masks1 and masks2 should be of type np.uint8')
    n = masks1.shape[0]
    m = masks2.shape[0]
    answer = np.zeros([n, m], dtype=np.float32)
    for i in np.arange(n):
        for j in np.arange(m):
            answer[i, j] = np.sum(np.minimum(masks1[i], masks2[j]), dtype=np.float32)
    return answer

def iou(masks1, masks2):
    if False:
        i = 10
        return i + 15
    'Computes pairwise intersection-over-union between mask collections.\n\n  Args:\n    masks1: a numpy array with shape [N, height, width] holding N masks. Masks\n      values are of type np.uint8 and values are in {0,1}.\n    masks2: a numpy array with shape [M, height, width] holding N masks. Masks\n      values are of type np.uint8 and values are in {0,1}.\n\n  Returns:\n    a numpy array with shape [N, M] representing pairwise iou scores.\n\n  Raises:\n    ValueError: If masks1 and masks2 are not of type np.uint8.\n  '
    if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
        raise ValueError('masks1 and masks2 should be of type np.uint8')
    intersect = intersection(masks1, masks2)
    area1 = area(masks1)
    area2 = area(masks2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect
    return intersect / np.maximum(union, EPSILON)

def ioa(masks1, masks2):
    if False:
        return 10
    "Computes pairwise intersection-over-area between box collections.\n\n  Intersection-over-area (ioa) between two masks, mask1 and mask2 is defined as\n  their intersection area over mask2's area. Note that ioa is not symmetric,\n  that is, IOA(mask1, mask2) != IOA(mask2, mask1).\n\n  Args:\n    masks1: a numpy array with shape [N, height, width] holding N masks. Masks\n      values are of type np.uint8 and values are in {0,1}.\n    masks2: a numpy array with shape [M, height, width] holding N masks. Masks\n      values are of type np.uint8 and values are in {0,1}.\n\n  Returns:\n    a numpy array with shape [N, M] representing pairwise ioa scores.\n\n  Raises:\n    ValueError: If masks1 and masks2 are not of type np.uint8.\n  "
    if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
        raise ValueError('masks1 and masks2 should be of type np.uint8')
    intersect = intersection(masks1, masks2)
    areas = np.expand_dims(area(masks2), axis=0)
    return intersect / (areas + EPSILON)