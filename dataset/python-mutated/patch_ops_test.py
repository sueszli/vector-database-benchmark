"""Tests for object_detection.utils.patch_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from object_detection.utils import patch_ops

class GetPatchMaskTest(tf.test.TestCase, parameterized.TestCase):

    def testMaskShape(self):
        if False:
            while True:
                i = 10
        image_shape = [15, 10]
        mask = patch_ops.get_patch_mask(10, 5, patch_size=3, image_shape=image_shape)
        self.assertListEqual(mask.shape.as_list(), image_shape)

    def testHandleImageShapeWithChannels(self):
        if False:
            for i in range(10):
                print('nop')
        image_shape = [15, 10, 3]
        mask = patch_ops.get_patch_mask(10, 5, patch_size=3, image_shape=image_shape)
        self.assertListEqual(mask.shape.as_list(), image_shape[:2])

    def testMaskDType(self):
        if False:
            for i in range(10):
                print('nop')
        mask = patch_ops.get_patch_mask(2, 3, patch_size=2, image_shape=[6, 7])
        self.assertDTypeEqual(mask, bool)

    def testMaskAreaWithEvenPatchSize(self):
        if False:
            return 10
        image_shape = [6, 7]
        mask = patch_ops.get_patch_mask(2, 3, patch_size=2, image_shape=image_shape)
        expected_mask = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]).reshape(image_shape).astype(bool)
        self.assertAllEqual(mask, expected_mask)

    def testMaskAreaWithEvenPatchSize4(self):
        if False:
            return 10
        image_shape = [6, 7]
        mask = patch_ops.get_patch_mask(2, 3, patch_size=4, image_shape=image_shape)
        expected_mask = np.array([[0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]).reshape(image_shape).astype(bool)
        self.assertAllEqual(mask, expected_mask)

    def testMaskAreaWithOddPatchSize(self):
        if False:
            return 10
        image_shape = [6, 7]
        mask = patch_ops.get_patch_mask(2, 3, patch_size=3, image_shape=image_shape)
        expected_mask = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]).reshape(image_shape).astype(bool)
        self.assertAllEqual(mask, expected_mask)

    def testMaskAreaPartiallyOutsideImage(self):
        if False:
            i = 10
            return i + 15
        image_shape = [6, 7]
        mask = patch_ops.get_patch_mask(5, 6, patch_size=5, image_shape=image_shape)
        expected_mask = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1]]).reshape(image_shape).astype(bool)
        self.assertAllEqual(mask, expected_mask)

    @parameterized.parameters({'y': 0, 'x': -1}, {'y': -1, 'x': 0}, {'y': 0, 'x': 11}, {'y': 16, 'x': 0})
    def testStaticCoordinatesOutsideImageRaisesError(self, y, x):
        if False:
            return 10
        image_shape = [15, 10]
        with self.assertRaises(tf.errors.InvalidArgumentError):
            patch_ops.get_patch_mask(y, x, patch_size=3, image_shape=image_shape)

    def testDynamicCoordinatesOutsideImageRaisesError(self):
        if False:
            i = 10
            return i + 15
        image_shape = [15, 10]
        x = tf.random_uniform([], minval=-2, maxval=-1, dtype=tf.int32)
        y = tf.random_uniform([], minval=0, maxval=1, dtype=tf.int32)
        mask = patch_ops.get_patch_mask(y, x, patch_size=3, image_shape=image_shape)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.evaluate(mask)

    @parameterized.parameters({'patch_size': 0}, {'patch_size': -1})
    def testStaticNonPositivePatchSizeRaisesError(self, patch_size):
        if False:
            while True:
                i = 10
        image_shape = [6, 7]
        with self.assertRaises(tf.errors.InvalidArgumentError):
            patch_ops.get_patch_mask(0, 0, patch_size=patch_size, image_shape=image_shape)

    def testDynamicNonPositivePatchSizeRaisesError(self):
        if False:
            while True:
                i = 10
        image_shape = [6, 7]
        patch_size = -1 * tf.random_uniform([], minval=0, maxval=3, dtype=tf.int32)
        mask = patch_ops.get_patch_mask(0, 0, patch_size=patch_size, image_shape=image_shape)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.evaluate(mask)
if __name__ == '__main__':
    tf.test.main()