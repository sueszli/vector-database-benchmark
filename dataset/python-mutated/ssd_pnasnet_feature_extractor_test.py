"""Tests for ssd_pnas_feature_extractor."""
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from object_detection.models import ssd_feature_extractor_test
from object_detection.models import ssd_pnasnet_feature_extractor
slim = contrib_slim

class SsdPnasNetFeatureExtractorTest(ssd_feature_extractor_test.SsdFeatureExtractorTestBase):

    def _create_feature_extractor(self, depth_multiplier, pad_to_multiple, use_explicit_padding=False, num_layers=6, is_training=True):
        if False:
            return 10
        "Constructs a new feature extractor.\n\n    Args:\n      depth_multiplier: float depth multiplier for feature extractor\n      pad_to_multiple: the nearest multiple to zero pad the input height and\n        width dimensions to.\n      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad\n        inputs so that the output dimensions are the same as if 'SAME' padding\n        were used.\n      num_layers: number of SSD layers.\n      is_training: whether the network is in training mode.\n    Returns:\n      an ssd_meta_arch.SSDFeatureExtractor object.\n    "
        min_depth = 32
        return ssd_pnasnet_feature_extractor.SSDPNASNetFeatureExtractor(is_training, depth_multiplier, min_depth, pad_to_multiple, self.conv_hyperparams_fn, use_explicit_padding=use_explicit_padding, num_layers=num_layers)

    def test_extract_features_returns_correct_shapes_128(self):
        if False:
            return 10
        image_height = 128
        image_width = 128
        depth_multiplier = 1.0
        pad_to_multiple = 1
        expected_feature_map_shape = [(2, 8, 8, 2160), (2, 4, 4, 4320), (2, 2, 2, 512), (2, 1, 1, 256), (2, 1, 1, 256), (2, 1, 1, 128)]
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape)

    def test_extract_features_returns_correct_shapes_299(self):
        if False:
            i = 10
            return i + 15
        image_height = 299
        image_width = 299
        depth_multiplier = 1.0
        pad_to_multiple = 1
        expected_feature_map_shape = [(2, 19, 19, 2160), (2, 10, 10, 4320), (2, 5, 5, 512), (2, 3, 3, 256), (2, 2, 2, 256), (2, 1, 1, 128)]
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape)

    def test_preprocess_returns_correct_value_range(self):
        if False:
            i = 10
            return i + 15
        image_height = 128
        image_width = 128
        depth_multiplier = 1
        pad_to_multiple = 1
        test_image = np.random.rand(2, image_height, image_width, 3)
        feature_extractor = self._create_feature_extractor(depth_multiplier, pad_to_multiple)
        preprocessed_image = feature_extractor.preprocess(test_image)
        self.assertTrue(np.all(np.less_equal(np.abs(preprocessed_image), 1.0)))

    def test_extract_features_with_fewer_layers(self):
        if False:
            while True:
                i = 10
        image_height = 128
        image_width = 128
        depth_multiplier = 1.0
        pad_to_multiple = 1
        expected_feature_map_shape = [(2, 8, 8, 2160), (2, 4, 4, 4320), (2, 2, 2, 512), (2, 1, 1, 256)]
        self.check_extract_features_returns_correct_shape(2, image_height, image_width, depth_multiplier, pad_to_multiple, expected_feature_map_shape, num_layers=4)
if __name__ == '__main__':
    tf.test.main()