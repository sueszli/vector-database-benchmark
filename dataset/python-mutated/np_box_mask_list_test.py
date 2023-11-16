"""Tests for object_detection.utils.np_box_mask_list_test."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from object_detection.utils import np_box_mask_list

class BoxMaskListTest(tf.test.TestCase):

    def test_invalid_box_mask_data(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            np_box_mask_list.BoxMaskList(box_data=[0, 0, 1, 1], mask_data=np.zeros([1, 3, 3], dtype=np.uint8))
        with self.assertRaises(ValueError):
            np_box_mask_list.BoxMaskList(box_data=np.array([[0, 0, 1, 1]], dtype=int), mask_data=np.zeros([1, 3, 3], dtype=np.uint8))
        with self.assertRaises(ValueError):
            np_box_mask_list.BoxMaskList(box_data=np.array([0, 1, 1, 3, 4], dtype=float), mask_data=np.zeros([1, 3, 3], dtype=np.uint8))
        with self.assertRaises(ValueError):
            np_box_mask_list.BoxMaskList(box_data=np.array([[0, 1, 1, 3], [3, 1, 1, 5]], dtype=float), mask_data=np.zeros([2, 3, 3], dtype=np.uint8))
        with self.assertRaises(ValueError):
            np_box_mask_list.BoxMaskList(box_data=np.array([[0, 1, 1, 3], [1, 1, 1, 5]], dtype=float), mask_data=np.zeros([3, 5, 5], dtype=np.uint8))
        with self.assertRaises(ValueError):
            np_box_mask_list.BoxMaskList(box_data=np.array([[0, 1, 1, 3], [1, 1, 1, 5]], dtype=float), mask_data=np.zeros([2, 5], dtype=np.uint8))
        with self.assertRaises(ValueError):
            np_box_mask_list.BoxMaskList(box_data=np.array([[0, 1, 1, 3], [1, 1, 1, 5]], dtype=float), mask_data=np.zeros([2, 5, 5, 5], dtype=np.uint8))
        with self.assertRaises(ValueError):
            np_box_mask_list.BoxMaskList(box_data=np.array([[0, 1, 1, 3], [1, 1, 1, 5]], dtype=float), mask_data=np.zeros([2, 5, 5], dtype=np.int32))

    def test_has_field_with_existed_field(self):
        if False:
            for i in range(10):
                print('nop')
        boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]], dtype=float)
        box_mask_list = np_box_mask_list.BoxMaskList(box_data=boxes, mask_data=np.zeros([3, 5, 5], dtype=np.uint8))
        self.assertTrue(box_mask_list.has_field('boxes'))
        self.assertTrue(box_mask_list.has_field('masks'))

    def test_has_field_with_nonexisted_field(self):
        if False:
            for i in range(10):
                print('nop')
        boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]], dtype=float)
        box_mask_list = np_box_mask_list.BoxMaskList(box_data=boxes, mask_data=np.zeros([3, 3, 3], dtype=np.uint8))
        self.assertFalse(box_mask_list.has_field('scores'))

    def test_get_field_with_existed_field(self):
        if False:
            while True:
                i = 10
        boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]], dtype=float)
        masks = np.zeros([3, 3, 3], dtype=np.uint8)
        box_mask_list = np_box_mask_list.BoxMaskList(box_data=boxes, mask_data=masks)
        self.assertTrue(np.allclose(box_mask_list.get_field('boxes'), boxes))
        self.assertTrue(np.allclose(box_mask_list.get_field('masks'), masks))

    def test_get_field_with_nonexited_field(self):
        if False:
            i = 10
            return i + 15
        boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]], dtype=float)
        masks = np.zeros([3, 3, 3], dtype=np.uint8)
        box_mask_list = np_box_mask_list.BoxMaskList(box_data=boxes, mask_data=masks)
        with self.assertRaises(ValueError):
            box_mask_list.get_field('scores')

class AddExtraFieldTest(tf.test.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]], dtype=float)
        masks = np.zeros([3, 3, 3], dtype=np.uint8)
        self.box_mask_list = np_box_mask_list.BoxMaskList(box_data=boxes, mask_data=masks)

    def test_add_already_existed_field_bbox(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            self.box_mask_list.add_field('boxes', np.array([[0, 0, 0, 1, 0]], dtype=float))

    def test_add_already_existed_field_mask(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            self.box_mask_list.add_field('masks', np.zeros([3, 3, 3], dtype=np.uint8))

    def test_add_invalid_field_data(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            self.box_mask_list.add_field('scores', np.array([0.5, 0.7], dtype=float))
        with self.assertRaises(ValueError):
            self.box_mask_list.add_field('scores', np.array([0.5, 0.7, 0.9, 0.1], dtype=float))

    def test_add_single_dimensional_field_data(self):
        if False:
            i = 10
            return i + 15
        box_mask_list = self.box_mask_list
        scores = np.array([0.5, 0.7, 0.9], dtype=float)
        box_mask_list.add_field('scores', scores)
        self.assertTrue(np.allclose(scores, self.box_mask_list.get_field('scores')))

    def test_add_multi_dimensional_field_data(self):
        if False:
            for i in range(10):
                print('nop')
        box_mask_list = self.box_mask_list
        labels = np.array([[0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=int)
        box_mask_list.add_field('labels', labels)
        self.assertTrue(np.allclose(labels, self.box_mask_list.get_field('labels')))

    def test_get_extra_fields(self):
        if False:
            for i in range(10):
                print('nop')
        box_mask_list = self.box_mask_list
        self.assertItemsEqual(box_mask_list.get_extra_fields(), ['masks'])
        scores = np.array([0.5, 0.7, 0.9], dtype=float)
        box_mask_list.add_field('scores', scores)
        self.assertItemsEqual(box_mask_list.get_extra_fields(), ['masks', 'scores'])
        labels = np.array([[0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=int)
        box_mask_list.add_field('labels', labels)
        self.assertItemsEqual(box_mask_list.get_extra_fields(), ['masks', 'scores', 'labels'])

    def test_get_coordinates(self):
        if False:
            while True:
                i = 10
        (y_min, x_min, y_max, x_max) = self.box_mask_list.get_coordinates()
        expected_y_min = np.array([3.0, 14.0, 0.0], dtype=float)
        expected_x_min = np.array([4.0, 14.0, 0.0], dtype=float)
        expected_y_max = np.array([6.0, 15.0, 20.0], dtype=float)
        expected_x_max = np.array([8.0, 15.0, 20.0], dtype=float)
        self.assertTrue(np.allclose(y_min, expected_y_min))
        self.assertTrue(np.allclose(x_min, expected_x_min))
        self.assertTrue(np.allclose(y_max, expected_y_max))
        self.assertTrue(np.allclose(x_max, expected_x_max))

    def test_num_boxes(self):
        if False:
            for i in range(10):
                print('nop')
        boxes = np.array([[0.0, 0.0, 100.0, 100.0], [10.0, 30.0, 50.0, 70.0]], dtype=float)
        masks = np.zeros([2, 5, 5], dtype=np.uint8)
        box_mask_list = np_box_mask_list.BoxMaskList(box_data=boxes, mask_data=masks)
        expected_num_boxes = 2
        self.assertEquals(box_mask_list.num_boxes(), expected_num_boxes)
if __name__ == '__main__':
    tf.test.main()