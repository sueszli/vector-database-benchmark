"""Tests for ssd resnet v1 FPN feature extractors."""
import abc
import itertools
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from object_detection.models import ssd_feature_extractor_test

@parameterized.parameters({'use_keras': False}, {'use_keras': True})
class SSDResnetFPNFeatureExtractorTestBase(ssd_feature_extractor_test.SsdFeatureExtractorTestBase):
    """Helper test class for SSD Resnet v1 FPN feature extractors."""

    @abc.abstractmethod
    def _resnet_scope_name(self, use_keras):
        if False:
            i = 10
            return i + 15
        pass

    @abc.abstractmethod
    def _fpn_scope_name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'fpn'

    @abc.abstractmethod
    def _create_feature_extractor(self, depth_multiplier, pad_to_multiple, use_explicit_padding=False, min_depth=32, use_keras=False):
        if False:
            return 10
        pass

    def test_extract_features_returns_correct_shapes_256(self, use_keras):
        if False:
            for i in range(10):
                print('nop')
        image_height = 256
        image_width = 256
        depth_multiplier = 1.0
        pad_to_multiple = 1
        expected_feature_map_shape = [(2, 32, 32, 256), (2, 16, 16, 256), (2, 8, 8, 256), (2, 4, 4, 256), (2, 2, 2, 256)]
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape)

    def test_extract_features_returns_correct_shapes_with_dynamic_inputs(self, use_keras):
        if False:
            for i in range(10):
                print('nop')
        image_height = 256
        image_width = 256
        depth_multiplier = 1.0
        pad_to_multiple = 1
        expected_feature_map_shape = [(2, 32, 32, 256), (2, 16, 16, 256), (2, 8, 8, 256), (2, 4, 4, 256), (2, 2, 2, 256)]
        self.check_extract_features_returns_correct_shapes_with_dynamic_inputs(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape, use_keras=use_keras)

    def test_extract_features_returns_correct_shapes_with_depth_multiplier(self, use_keras):
        if False:
            print('Hello World!')
        image_height = 256
        image_width = 256
        depth_multiplier = 0.5
        expected_num_channels = int(256 * depth_multiplier)
        pad_to_multiple = 1
        expected_feature_map_shape = [(2, 32, 32, expected_num_channels), (2, 16, 16, expected_num_channels), (2, 8, 8, expected_num_channels), (2, 4, 4, expected_num_channels), (2, 2, 2, expected_num_channels)]
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape, use_keras=use_keras)

    def test_extract_features_returns_correct_shapes_with_min_depth(self, use_keras):
        if False:
            print('Hello World!')
        image_height = 256
        image_width = 256
        depth_multiplier = 1.0
        pad_to_multiple = 1
        min_depth = 320
        expected_feature_map_shape = [(2, 32, 32, min_depth), (2, 16, 16, min_depth), (2, 8, 8, min_depth), (2, 4, 4, min_depth), (2, 2, 2, min_depth)]

        def graph_fn(image_tensor):
            if False:
                while True:
                    i = 10
            feature_extractor = self._create_feature_extractor(depth_multiplier, pad_to_multiple, min_depth=min_depth, use_keras=use_keras)
            if use_keras:
                return feature_extractor(image_tensor)
            return feature_extractor.extract_features(image_tensor)
        image_tensor = np.random.rand(2, image_height, image_width, 3).astype(np.float32)
        feature_maps = self.execute(graph_fn, [image_tensor])
        for (feature_map, expected_shape) in itertools.izip(feature_maps, expected_feature_map_shape):
            self.assertAllEqual(feature_map.shape, expected_shape)

    def test_extract_features_returns_correct_shapes_with_pad_to_multiple(self, use_keras):
        if False:
            return 10
        image_height = 254
        image_width = 254
        depth_multiplier = 1.0
        pad_to_multiple = 32
        expected_feature_map_shape = [(2, 32, 32, 256), (2, 16, 16, 256), (2, 8, 8, 256), (2, 4, 4, 256), (2, 2, 2, 256)]
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape, use_keras=use_keras)

    def test_extract_features_raises_error_with_invalid_image_size(self, use_keras):
        if False:
            return 10
        image_height = 32
        image_width = 32
        depth_multiplier = 1.0
        pad_to_multiple = 1
        self.check_extract_features_raises_error_with_invalid_image_size(image_height, image_width, depth_multiplier, pad_to_multiple, use_keras=use_keras)

    def test_preprocess_returns_correct_value_range(self, use_keras):
        if False:
            while True:
                i = 10
        image_height = 128
        image_width = 128
        depth_multiplier = 1
        pad_to_multiple = 1
        test_image = tf.constant(np.random.rand(4, image_height, image_width, 3))
        feature_extractor = self._create_feature_extractor(depth_multiplier, pad_to_multiple, use_keras=use_keras)
        preprocessed_image = feature_extractor.preprocess(test_image)
        with self.test_session() as sess:
            (test_image_out, preprocessed_image_out) = sess.run([test_image, preprocessed_image])
            self.assertAllClose(preprocessed_image_out, test_image_out - [[123.68, 116.779, 103.939]])

    def test_variables_only_created_in_scope(self, use_keras):
        if False:
            print('Hello World!')
        depth_multiplier = 1
        pad_to_multiple = 1
        scope_name = self._resnet_scope_name(use_keras)
        self.check_feature_extractor_variables_under_scope(depth_multiplier, pad_to_multiple, scope_name, use_keras=use_keras)

    def test_variable_count(self, use_keras):
        if False:
            while True:
                i = 10
        depth_multiplier = 1
        pad_to_multiple = 1
        variables = self.get_feature_extractor_variables(depth_multiplier, pad_to_multiple, use_keras=use_keras)
        expected_variables_len = 279
        scope_name = self._resnet_scope_name(use_keras)
        if scope_name in ('ResNet101V1_FPN', 'resnet_v1_101'):
            expected_variables_len = 534
        elif scope_name in ('ResNet152V1_FPN', 'resnet_v1_152'):
            expected_variables_len = 789
        self.assertEqual(len(variables), expected_variables_len)