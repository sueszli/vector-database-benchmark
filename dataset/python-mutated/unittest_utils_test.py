"""Tests for unittest_utils."""
import StringIO
import numpy as np
from PIL import Image as PILImage
import tensorflow as tf
import unittest_utils

class UnittestUtilsTest(tf.test.TestCase):

    def test_creates_an_image_of_specified_shape(self):
        if False:
            i = 10
            return i + 15
        (image, _) = unittest_utils.create_random_image('PNG', (10, 20, 3))
        self.assertEqual(image.shape, (10, 20, 3))

    def test_encoded_image_corresponds_to_numpy_array(self):
        if False:
            while True:
                i = 10
        (image, encoded) = unittest_utils.create_random_image('PNG', (20, 10, 3))
        pil_image = PILImage.open(StringIO.StringIO(encoded))
        self.assertAllEqual(image, np.array(pil_image))

    def test_created_example_has_correct_values(self):
        if False:
            while True:
                i = 10
        example_serialized = unittest_utils.create_serialized_example({'labels': [1, 2, 3], 'data': ['FAKE']})
        example = tf.train.Example()
        example.ParseFromString(example_serialized)
        self.assertProtoEquals('\n      features {\n        feature {\n          key: "labels"\n           value { int64_list {\n             value: 1\n             value: 2\n             value: 3\n           }}\n         }\n         feature {\n           key: "data"\n           value { bytes_list {\n             value: "FAKE"\n           }}\n         }\n      }\n    ', example)
if __name__ == '__main__':
    tf.test.main()