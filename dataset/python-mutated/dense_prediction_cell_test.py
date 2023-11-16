"""Tests for dense_prediction_cell."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from deeplab.core import dense_prediction_cell

class DensePredictionCellTest(tf.test.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.segmentation_layer = dense_prediction_cell.DensePredictionCell(config=[{dense_prediction_cell._INPUT: -1, dense_prediction_cell._OP: dense_prediction_cell._CONV, dense_prediction_cell._KERNEL: 1}, {dense_prediction_cell._INPUT: 0, dense_prediction_cell._OP: dense_prediction_cell._CONV, dense_prediction_cell._KERNEL: 3, dense_prediction_cell._RATE: [1, 3]}, {dense_prediction_cell._INPUT: 1, dense_prediction_cell._OP: dense_prediction_cell._PYRAMID_POOLING, dense_prediction_cell._GRID_SIZE: [1, 2]}], hparams={'conv_rate_multiplier': 2})

    def testPyramidPoolingArguments(self):
        if False:
            while True:
                i = 10
        (features_size, pooled_kernel) = self.segmentation_layer._get_pyramid_pooling_arguments(crop_size=[513, 513], output_stride=16, image_grid=[4, 4])
        self.assertListEqual(features_size, [33, 33])
        self.assertListEqual(pooled_kernel, [9, 9])

    def testPyramidPoolingArgumentsWithImageGrid1x1(self):
        if False:
            i = 10
            return i + 15
        (features_size, pooled_kernel) = self.segmentation_layer._get_pyramid_pooling_arguments(crop_size=[257, 257], output_stride=16, image_grid=[1, 1])
        self.assertListEqual(features_size, [17, 17])
        self.assertListEqual(pooled_kernel, [17, 17])

    def testParseOperationStringWithConv1x1(self):
        if False:
            print('Hello World!')
        operation = self.segmentation_layer._parse_operation(config={dense_prediction_cell._OP: dense_prediction_cell._CONV, dense_prediction_cell._KERNEL: [1, 1]}, crop_size=[513, 513], output_stride=16)
        self.assertEqual(operation[dense_prediction_cell._OP], dense_prediction_cell._CONV)
        self.assertListEqual(operation[dense_prediction_cell._KERNEL], [1, 1])

    def testParseOperationStringWithConv3x3(self):
        if False:
            i = 10
            return i + 15
        operation = self.segmentation_layer._parse_operation(config={dense_prediction_cell._OP: dense_prediction_cell._CONV, dense_prediction_cell._KERNEL: [3, 3], dense_prediction_cell._RATE: [9, 6]}, crop_size=[513, 513], output_stride=16)
        self.assertEqual(operation[dense_prediction_cell._OP], dense_prediction_cell._CONV)
        self.assertListEqual(operation[dense_prediction_cell._KERNEL], [3, 3])
        self.assertEqual(operation[dense_prediction_cell._RATE], [9, 6])

    def testParseOperationStringWithPyramidPooling2x2(self):
        if False:
            return 10
        operation = self.segmentation_layer._parse_operation(config={dense_prediction_cell._OP: dense_prediction_cell._PYRAMID_POOLING, dense_prediction_cell._GRID_SIZE: [2, 2]}, crop_size=[513, 513], output_stride=16)
        self.assertEqual(operation[dense_prediction_cell._OP], dense_prediction_cell._PYRAMID_POOLING)
        self.assertListEqual(operation[dense_prediction_cell._TARGET_SIZE], [33, 33])
        self.assertListEqual(operation[dense_prediction_cell._KERNEL], [17, 17])

    def testBuildCell(self):
        if False:
            while True:
                i = 10
        with self.test_session(graph=tf.Graph()) as sess:
            features = tf.random_normal([2, 33, 33, 5])
            concat_logits = self.segmentation_layer.build_cell(features, output_stride=8, crop_size=[257, 257])
            sess.run(tf.global_variables_initializer())
            concat_logits = sess.run(concat_logits)
            self.assertTrue(concat_logits.any())

    def testBuildCellWithImagePoolingCropSize(self):
        if False:
            return 10
        with self.test_session(graph=tf.Graph()) as sess:
            features = tf.random_normal([2, 33, 33, 5])
            concat_logits = self.segmentation_layer.build_cell(features, output_stride=8, crop_size=[257, 257], image_pooling_crop_size=[129, 129])
            sess.run(tf.global_variables_initializer())
            concat_logits = sess.run(concat_logits)
            self.assertTrue(concat_logits.any())
if __name__ == '__main__':
    tf.test.main()