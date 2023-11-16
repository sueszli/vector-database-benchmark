"""Tests for ssd_mobilenet_v1_ppn_feature_extractor."""
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from object_detection.models import ssd_feature_extractor_test
from object_detection.models import ssd_mobilenet_v1_ppn_feature_extractor
slim = contrib_slim

class SsdMobilenetV1PpnFeatureExtractorTest(ssd_feature_extractor_test.SsdFeatureExtractorTestBase):

    def _create_feature_extractor(self, depth_multiplier, pad_to_multiple, is_training=True, use_explicit_padding=False):
        if False:
            i = 10
            return i + 15
        "Constructs a new feature extractor.\n\n    Args:\n      depth_multiplier: float depth multiplier for feature extractor\n      pad_to_multiple: the nearest multiple to zero pad the input height and\n        width dimensions to.\n      is_training: whether the network is in training mode.\n      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad\n        inputs so that the output dimensions are the same as if 'SAME' padding\n        were used.\n    Returns:\n      an ssd_meta_arch.SSDFeatureExtractor object.\n    "
        min_depth = 32
        return ssd_mobilenet_v1_ppn_feature_extractor.SSDMobileNetV1PpnFeatureExtractor(is_training, depth_multiplier, min_depth, pad_to_multiple, self.conv_hyperparams_fn, use_explicit_padding=use_explicit_padding)

    def test_extract_features_returns_correct_shapes_320(self):
        if False:
            i = 10
            return i + 15
        image_height = 320
        image_width = 320
        depth_multiplier = 1.0
        pad_to_multiple = 1
        expected_feature_map_shape = [(2, 20, 20, 512), (2, 10, 10, 512), (2, 5, 5, 512), (2, 3, 3, 512), (2, 2, 2, 512), (2, 1, 1, 512)]
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape, use_explicit_padding=False)
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape, use_explicit_padding=True)

    def test_extract_features_returns_correct_shapes_300(self):
        if False:
            for i in range(10):
                print('nop')
        image_height = 300
        image_width = 300
        depth_multiplier = 1.0
        pad_to_multiple = 1
        expected_feature_map_shape = [(2, 19, 19, 512), (2, 10, 10, 512), (2, 5, 5, 512), (2, 3, 3, 512), (2, 2, 2, 512), (2, 1, 1, 512)]
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape, use_explicit_padding=False)
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape, use_explicit_padding=True)

    def test_extract_features_returns_correct_shapes_640(self):
        if False:
            i = 10
            return i + 15
        image_height = 640
        image_width = 640
        depth_multiplier = 1.0
        pad_to_multiple = 1
        expected_feature_map_shape = [(2, 40, 40, 512), (2, 20, 20, 512), (2, 10, 10, 512), (2, 5, 5, 512), (2, 3, 3, 512), (2, 2, 2, 512)]
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape, use_explicit_padding=False)
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape, use_explicit_padding=True)

    def test_extract_features_with_dynamic_image_shape(self):
        if False:
            print('Hello World!')
        image_height = 320
        image_width = 320
        depth_multiplier = 1.0
        pad_to_multiple = 1
        expected_feature_map_shape = [(2, 20, 20, 512), (2, 10, 10, 512), (2, 5, 5, 512), (2, 3, 3, 512), (2, 2, 2, 512), (2, 1, 1, 512)]
        self.check_extract_features_returns_correct_shapes_with_dynamic_inputs(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape, use_explicit_padding=False)
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape, use_explicit_padding=True)

    def test_extract_features_returns_correct_shapes_with_pad_to_multiple(self):
        if False:
            for i in range(10):
                print('nop')
        image_height = 299
        image_width = 299
        depth_multiplier = 1.0
        pad_to_multiple = 32
        expected_feature_map_shape = [(2, 20, 20, 512), (2, 10, 10, 512), (2, 5, 5, 512), (2, 3, 3, 512), (2, 2, 2, 512)]
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape, use_explicit_padding=False)
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape, use_explicit_padding=True)

    def test_extract_features_returns_correct_shapes_enforcing_min_depth(self):
        if False:
            return 10
        image_height = 256
        image_width = 256
        depth_multiplier = 0.5 ** 12
        pad_to_multiple = 1
        expected_feature_map_shape = [(2, 16, 16, 32), (2, 8, 8, 32), (2, 4, 4, 32), (2, 2, 2, 32), (2, 1, 1, 32)]
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape, use_explicit_padding=False)
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape, use_explicit_padding=True)

    def test_extract_features_raises_error_with_invalid_image_size(self):
        if False:
            print('Hello World!')
        image_height = 32
        image_width = 32
        depth_multiplier = 1.0
        pad_to_multiple = 1
        self.check_extract_features_raises_error_with_invalid_image_size(image_height, image_width, depth_multiplier, pad_to_multiple)

    def test_preprocess_returns_correct_value_range(self):
        if False:
            while True:
                i = 10
        image_height = 128
        image_width = 128
        depth_multiplier = 1
        pad_to_multiple = 1
        test_image = np.random.rand(2, image_height, image_width, 3)
        feature_extractor = self._create_feature_extractor(depth_multiplier, pad_to_multiple)
        preprocessed_image = feature_extractor.preprocess(test_image)
        self.assertTrue(np.all(np.less_equal(np.abs(preprocessed_image), 1.0)))

    def test_variables_only_created_in_scope(self):
        if False:
            i = 10
            return i + 15
        depth_multiplier = 1
        pad_to_multiple = 1
        scope_name = 'MobilenetV1'
        self.check_feature_extractor_variables_under_scope(depth_multiplier, pad_to_multiple, scope_name)

    def test_has_fused_batchnorm(self):
        if False:
            return 10
        image_height = 320
        image_width = 320
        depth_multiplier = 1
        pad_to_multiple = 1
        image_placeholder = tf.placeholder(tf.float32, [1, image_height, image_width, 3])
        feature_extractor = self._create_feature_extractor(depth_multiplier, pad_to_multiple)
        preprocessed_image = feature_extractor.preprocess(image_placeholder)
        _ = feature_extractor.extract_features(preprocessed_image)
        self.assertTrue(any(('FusedBatchNorm' in op.type for op in tf.get_default_graph().get_operations())))
if __name__ == '__main__':
    tf.test.main()