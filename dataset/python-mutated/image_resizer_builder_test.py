"""Tests for object_detection.builders.image_resizer_builder."""
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import image_resizer_builder
from object_detection.protos import image_resizer_pb2

class ImageResizerBuilderTest(tf.test.TestCase):

    def _shape_of_resized_random_image_given_text_proto(self, input_shape, text_proto):
        if False:
            return 10
        image_resizer_config = image_resizer_pb2.ImageResizer()
        text_format.Merge(text_proto, image_resizer_config)
        image_resizer_fn = image_resizer_builder.build(image_resizer_config)
        images = tf.cast(tf.random_uniform(input_shape, minval=0, maxval=255, dtype=tf.int32), dtype=tf.float32)
        (resized_images, _) = image_resizer_fn(images)
        with self.test_session() as sess:
            return sess.run(resized_images).shape

    def test_build_keep_aspect_ratio_resizer_returns_expected_shape(self):
        if False:
            i = 10
            return i + 15
        image_resizer_text_proto = '\n      keep_aspect_ratio_resizer {\n        min_dimension: 10\n        max_dimension: 20\n      }\n    '
        input_shape = (50, 25, 3)
        expected_output_shape = (20, 10, 3)
        output_shape = self._shape_of_resized_random_image_given_text_proto(input_shape, image_resizer_text_proto)
        self.assertEqual(output_shape, expected_output_shape)

    def test_build_keep_aspect_ratio_resizer_grayscale(self):
        if False:
            print('Hello World!')
        image_resizer_text_proto = '\n      keep_aspect_ratio_resizer {\n        min_dimension: 10\n        max_dimension: 20\n        convert_to_grayscale: true\n      }\n    '
        input_shape = (50, 25, 3)
        expected_output_shape = (20, 10, 1)
        output_shape = self._shape_of_resized_random_image_given_text_proto(input_shape, image_resizer_text_proto)
        self.assertEqual(output_shape, expected_output_shape)

    def test_build_keep_aspect_ratio_resizer_with_padding(self):
        if False:
            for i in range(10):
                print('nop')
        image_resizer_text_proto = '\n      keep_aspect_ratio_resizer {\n        min_dimension: 10\n        max_dimension: 20\n        pad_to_max_dimension: true\n        per_channel_pad_value: 3\n        per_channel_pad_value: 4\n        per_channel_pad_value: 5\n      }\n    '
        input_shape = (50, 25, 3)
        expected_output_shape = (20, 20, 3)
        output_shape = self._shape_of_resized_random_image_given_text_proto(input_shape, image_resizer_text_proto)
        self.assertEqual(output_shape, expected_output_shape)

    def test_built_fixed_shape_resizer_returns_expected_shape(self):
        if False:
            print('Hello World!')
        image_resizer_text_proto = '\n      fixed_shape_resizer {\n        height: 10\n        width: 20\n      }\n    '
        input_shape = (50, 25, 3)
        expected_output_shape = (10, 20, 3)
        output_shape = self._shape_of_resized_random_image_given_text_proto(input_shape, image_resizer_text_proto)
        self.assertEqual(output_shape, expected_output_shape)

    def test_built_fixed_shape_resizer_grayscale(self):
        if False:
            print('Hello World!')
        image_resizer_text_proto = '\n      fixed_shape_resizer {\n        height: 10\n        width: 20\n        convert_to_grayscale: true\n      }\n    '
        input_shape = (50, 25, 3)
        expected_output_shape = (10, 20, 1)
        output_shape = self._shape_of_resized_random_image_given_text_proto(input_shape, image_resizer_text_proto)
        self.assertEqual(output_shape, expected_output_shape)

    def test_identity_resizer_returns_expected_shape(self):
        if False:
            while True:
                i = 10
        image_resizer_text_proto = '\n      identity_resizer {\n      }\n    '
        input_shape = (10, 20, 3)
        expected_output_shape = (10, 20, 3)
        output_shape = self._shape_of_resized_random_image_given_text_proto(input_shape, image_resizer_text_proto)
        self.assertEqual(output_shape, expected_output_shape)

    def test_raises_error_on_invalid_input(self):
        if False:
            i = 10
            return i + 15
        invalid_input = 'invalid_input'
        with self.assertRaises(ValueError):
            image_resizer_builder.build(invalid_input)

    def _resized_image_given_text_proto(self, image, text_proto):
        if False:
            return 10
        image_resizer_config = image_resizer_pb2.ImageResizer()
        text_format.Merge(text_proto, image_resizer_config)
        image_resizer_fn = image_resizer_builder.build(image_resizer_config)
        image_placeholder = tf.placeholder(tf.uint8, [1, None, None, 3])
        (resized_image, _) = image_resizer_fn(image_placeholder)
        with self.test_session() as sess:
            return sess.run(resized_image, feed_dict={image_placeholder: image})

    def test_fixed_shape_resizer_nearest_neighbor_method(self):
        if False:
            print('Hello World!')
        image_resizer_text_proto = '\n      fixed_shape_resizer {\n        height: 1\n        width: 1\n        resize_method: NEAREST_NEIGHBOR\n      }\n    '
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        image = np.expand_dims(image, axis=2)
        image = np.tile(image, (1, 1, 3))
        image = np.expand_dims(image, axis=0)
        resized_image = self._resized_image_given_text_proto(image, image_resizer_text_proto)
        vals = np.unique(resized_image).tolist()
        self.assertEqual(len(vals), 1)
        self.assertEqual(vals[0], 1)

    def test_build_conditional_shape_resizer_greater_returns_expected_shape(self):
        if False:
            return 10
        image_resizer_text_proto = '\n      conditional_shape_resizer {\n        condition: GREATER\n        size_threshold: 30\n      }\n    '
        input_shape = (60, 30, 3)
        expected_output_shape = (30, 15, 3)
        output_shape = self._shape_of_resized_random_image_given_text_proto(input_shape, image_resizer_text_proto)
        self.assertEqual(output_shape, expected_output_shape)

    def test_build_conditional_shape_resizer_same_shape_with_no_resize(self):
        if False:
            print('Hello World!')
        image_resizer_text_proto = '\n      conditional_shape_resizer {\n        condition: GREATER\n        size_threshold: 30\n      }\n    '
        input_shape = (15, 15, 3)
        expected_output_shape = (15, 15, 3)
        output_shape = self._shape_of_resized_random_image_given_text_proto(input_shape, image_resizer_text_proto)
        self.assertEqual(output_shape, expected_output_shape)

    def test_build_conditional_shape_resizer_smaller_returns_expected_shape(self):
        if False:
            while True:
                i = 10
        image_resizer_text_proto = '\n      conditional_shape_resizer {\n        condition: SMALLER\n        size_threshold: 30\n      }\n    '
        input_shape = (30, 15, 3)
        expected_output_shape = (60, 30, 3)
        output_shape = self._shape_of_resized_random_image_given_text_proto(input_shape, image_resizer_text_proto)
        self.assertEqual(output_shape, expected_output_shape)

    def test_build_conditional_shape_resizer_grayscale(self):
        if False:
            print('Hello World!')
        image_resizer_text_proto = '\n      conditional_shape_resizer {\n        condition: GREATER\n        size_threshold: 30\n        convert_to_grayscale: true\n      }\n    '
        input_shape = (60, 30, 3)
        expected_output_shape = (30, 15, 1)
        output_shape = self._shape_of_resized_random_image_given_text_proto(input_shape, image_resizer_text_proto)
        self.assertEqual(output_shape, expected_output_shape)

    def test_build_conditional_shape_resizer_error_on_invalid_condition(self):
        if False:
            for i in range(10):
                print('nop')
        invalid_image_resizer_text_proto = '\n      conditional_shape_resizer {\n        condition: INVALID\n        size_threshold: 30\n      }\n    '
        with self.assertRaises(ValueError):
            image_resizer_builder.build(invalid_image_resizer_text_proto)
if __name__ == '__main__':
    tf.test.main()