"""Tests for ssd resnet v1 feature extractors."""
import tensorflow as tf
from object_detection.models import ssd_resnet_v1_ppn_feature_extractor
from object_detection.models import ssd_resnet_v1_ppn_feature_extractor_testbase

class SSDResnet50V1PpnFeatureExtractorTest(ssd_resnet_v1_ppn_feature_extractor_testbase.SSDResnetPpnFeatureExtractorTestBase):
    """SSDResnet50v1 feature extractor test."""

    def _create_feature_extractor(self, depth_multiplier, pad_to_multiple, use_explicit_padding=False):
        if False:
            return 10
        min_depth = 32
        is_training = True
        return ssd_resnet_v1_ppn_feature_extractor.SSDResnet50V1PpnFeatureExtractor(is_training, depth_multiplier, min_depth, pad_to_multiple, self.conv_hyperparams_fn, use_explicit_padding=use_explicit_padding)

    def _scope_name(self):
        if False:
            while True:
                i = 10
        return 'resnet_v1_50'

class SSDResnet101V1PpnFeatureExtractorTest(ssd_resnet_v1_ppn_feature_extractor_testbase.SSDResnetPpnFeatureExtractorTestBase):
    """SSDResnet101v1 feature extractor test."""

    def _create_feature_extractor(self, depth_multiplier, pad_to_multiple, use_explicit_padding=False):
        if False:
            while True:
                i = 10
        min_depth = 32
        is_training = True
        return ssd_resnet_v1_ppn_feature_extractor.SSDResnet101V1PpnFeatureExtractor(is_training, depth_multiplier, min_depth, pad_to_multiple, self.conv_hyperparams_fn, use_explicit_padding=use_explicit_padding)

    def _scope_name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'resnet_v1_101'

class SSDResnet152V1PpnFeatureExtractorTest(ssd_resnet_v1_ppn_feature_extractor_testbase.SSDResnetPpnFeatureExtractorTestBase):
    """SSDResnet152v1 feature extractor test."""

    def _create_feature_extractor(self, depth_multiplier, pad_to_multiple, use_explicit_padding=False):
        if False:
            while True:
                i = 10
        min_depth = 32
        is_training = True
        return ssd_resnet_v1_ppn_feature_extractor.SSDResnet152V1PpnFeatureExtractor(is_training, depth_multiplier, min_depth, pad_to_multiple, self.conv_hyperparams_fn, use_explicit_padding=use_explicit_padding)

    def _scope_name(self):
        if False:
            return 10
        return 'resnet_v1_152'
if __name__ == '__main__':
    tf.test.main()