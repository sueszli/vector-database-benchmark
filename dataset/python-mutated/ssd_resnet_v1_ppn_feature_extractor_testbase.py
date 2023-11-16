"""Tests for ssd resnet v1 feature extractors."""
import abc
import numpy as np
import tensorflow as tf
from object_detection.models import ssd_feature_extractor_test

class SSDResnetPpnFeatureExtractorTestBase(ssd_feature_extractor_test.SsdFeatureExtractorTestBase):
    """Helper test class for SSD Resnet PPN feature extractors."""

    @abc.abstractmethod
    def _scope_name(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_extract_features_returns_correct_shapes_289(self):
        if False:
            i = 10
            return i + 15
        image_height = 289
        image_width = 289
        depth_multiplier = 1.0
        pad_to_multiple = 1
        expected_feature_map_shape = [(2, 19, 19, 1024), (2, 10, 10, 1024), (2, 5, 5, 1024), (2, 3, 3, 1024), (2, 2, 2, 1024), (2, 1, 1, 1024)]
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape)

    def test_extract_features_returns_correct_shapes_with_dynamic_inputs(self):
        if False:
            print('Hello World!')
        image_height = 289
        image_width = 289
        depth_multiplier = 1.0
        pad_to_multiple = 1
        expected_feature_map_shape = [(2, 19, 19, 1024), (2, 10, 10, 1024), (2, 5, 5, 1024), (2, 3, 3, 1024), (2, 2, 2, 1024), (2, 1, 1, 1024)]
        self.check_extract_features_returns_correct_shapes_with_dynamic_inputs(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape)

    def test_extract_features_raises_error_with_invalid_image_size(self):
        if False:
            i = 10
            return i + 15
        image_height = 32
        image_width = 32
        depth_multiplier = 1.0
        pad_to_multiple = 1
        self.check_extract_features_raises_error_with_invalid_image_size(image_height, image_width, depth_multiplier, pad_to_multiple)

    def test_preprocess_returns_correct_value_range(self):
        if False:
            print('Hello World!')
        image_height = 128
        image_width = 128
        depth_multiplier = 1
        pad_to_multiple = 1
        test_image = tf.constant(np.random.rand(4, image_height, image_width, 3))
        feature_extractor = self._create_feature_extractor(depth_multiplier, pad_to_multiple)
        preprocessed_image = feature_extractor.preprocess(test_image)
        with self.test_session() as sess:
            (test_image_out, preprocessed_image_out) = sess.run([test_image, preprocessed_image])
            self.assertAllClose(preprocessed_image_out, test_image_out - [[123.68, 116.779, 103.939]])

    def test_variables_only_created_in_scope(self):
        if False:
            print('Hello World!')
        depth_multiplier = 1
        pad_to_multiple = 1
        self.check_feature_extractor_variables_under_scope(depth_multiplier, pad_to_multiple, self._scope_name())