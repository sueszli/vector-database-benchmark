"""Tests for object_detection.np_box_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from object_detection.utils import np_box_ops

class BoxOpsTests(tf.test.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        boxes1 = np.array([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]], dtype=float)
        boxes2 = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]], dtype=float)
        self.boxes1 = boxes1
        self.boxes2 = boxes2

    def testArea(self):
        if False:
            i = 10
            return i + 15
        areas = np_box_ops.area(self.boxes1)
        expected_areas = np.array([6.0, 5.0], dtype=float)
        self.assertAllClose(expected_areas, areas)

    def testIntersection(self):
        if False:
            return 10
        intersection = np_box_ops.intersection(self.boxes1, self.boxes2)
        expected_intersection = np.array([[2.0, 0.0, 6.0], [1.0, 0.0, 5.0]], dtype=float)
        self.assertAllClose(intersection, expected_intersection)

    def testIOU(self):
        if False:
            print('Hello World!')
        iou = np_box_ops.iou(self.boxes1, self.boxes2)
        expected_iou = np.array([[2.0 / 16.0, 0.0, 6.0 / 400.0], [1.0 / 16.0, 0.0, 5.0 / 400.0]], dtype=float)
        self.assertAllClose(iou, expected_iou)

    def testIOA(self):
        if False:
            i = 10
            return i + 15
        boxes1 = np.array([[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.5, 0.75]], dtype=np.float32)
        boxes2 = np.array([[0.5, 0.25, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        ioa21 = np_box_ops.ioa(boxes2, boxes1)
        expected_ioa21 = np.array([[0.5, 0.0], [1.0, 1.0]], dtype=np.float32)
        self.assertAllClose(ioa21, expected_ioa21)
if __name__ == '__main__':
    tf.test.main()