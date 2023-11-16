"""Tests for test_utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
import numpy as np
from deeplab.evaluation import test_utils

class TestUtilsTest(absltest.TestCase):

    def test_read_test_image(self):
        if False:
            return 10
        image_array = test_utils.read_test_image('team_pred_class.png')
        self.assertSequenceEqual(image_array.shape, (231, 345, 4))

    def test_reads_segmentation_with_color_map(self):
        if False:
            for i in range(10):
                print('nop')
        rgb_to_semantic_label = {(0, 0, 0): 0, (0, 0, 255): 1, (255, 0, 0): 23}
        labels = test_utils.read_segmentation_with_rgb_color_map('team_pred_class.png', rgb_to_semantic_label)
        input_image = test_utils.read_test_image('team_pred_class.png')
        np.testing.assert_array_equal(labels == 0, np.logical_and(input_image[:, :, 0] == 0, input_image[:, :, 2] == 0))
        np.testing.assert_array_equal(labels == 1, input_image[:, :, 2] == 255)
        np.testing.assert_array_equal(labels == 23, input_image[:, :, 0] == 255)

    def test_reads_gt_segmentation(self):
        if False:
            while True:
                i = 10
        instance_label_to_semantic_label = {0: 0, 47: 1, 97: 1, 133: 1, 150: 1, 174: 1, 198: 23, 215: 1, 244: 1, 255: 1}
        (instances, classes) = test_utils.panoptic_segmentation_with_class_map('team_gt_instance.png', instance_label_to_semantic_label)
        expected_label_shape = (231, 345)
        self.assertSequenceEqual(instances.shape, expected_label_shape)
        self.assertSequenceEqual(classes.shape, expected_label_shape)
        np.testing.assert_array_equal(instances == 0, classes == 0)
        np.testing.assert_array_equal(instances == 198, classes == 23)
        np.testing.assert_array_equal(np.logical_and(instances != 0, instances != 198), classes == 1)
if __name__ == '__main__':
    absltest.main()