"""Tests for ssd resnet v1 FPN feature extractors."""
import tensorflow as tf
from object_detection.models import ssd_resnet_v1_fpn_feature_extractor
from object_detection.models import ssd_resnet_v1_fpn_feature_extractor_testbase
from object_detection.models import ssd_resnet_v1_fpn_keras_feature_extractor

class SSDResnet50V1FeatureExtractorTest(ssd_resnet_v1_fpn_feature_extractor_testbase.SSDResnetFPNFeatureExtractorTestBase):
    """SSDResnet50v1Fpn feature extractor test."""

    def _create_feature_extractor(self, depth_multiplier, pad_to_multiple, use_explicit_padding=False, min_depth=32, use_keras=False):
        if False:
            i = 10
            return i + 15
        is_training = True
        if use_keras:
            return ssd_resnet_v1_fpn_keras_feature_extractor.SSDResNet50V1FpnKerasFeatureExtractor(is_training=is_training, depth_multiplier=depth_multiplier, min_depth=min_depth, pad_to_multiple=pad_to_multiple, conv_hyperparams=self._build_conv_hyperparams(add_batch_norm=False), freeze_batchnorm=False, inplace_batchnorm_update=False, name='ResNet50V1_FPN')
        else:
            return ssd_resnet_v1_fpn_feature_extractor.SSDResnet50V1FpnFeatureExtractor(is_training, depth_multiplier, min_depth, pad_to_multiple, self.conv_hyperparams_fn, use_explicit_padding=use_explicit_padding)

    def _resnet_scope_name(self, use_keras=False):
        if False:
            while True:
                i = 10
        if use_keras:
            return 'ResNet50V1_FPN'
        return 'resnet_v1_50'

class SSDResnet101V1FeatureExtractorTest(ssd_resnet_v1_fpn_feature_extractor_testbase.SSDResnetFPNFeatureExtractorTestBase):
    """SSDResnet101v1Fpn feature extractor test."""

    def _create_feature_extractor(self, depth_multiplier, pad_to_multiple, use_explicit_padding=False, min_depth=32, use_keras=False):
        if False:
            print('Hello World!')
        is_training = True
        if use_keras:
            return ssd_resnet_v1_fpn_keras_feature_extractor.SSDResNet101V1FpnKerasFeatureExtractor(is_training=is_training, depth_multiplier=depth_multiplier, min_depth=min_depth, pad_to_multiple=pad_to_multiple, conv_hyperparams=self._build_conv_hyperparams(add_batch_norm=False), freeze_batchnorm=False, inplace_batchnorm_update=False, name='ResNet101V1_FPN')
        else:
            return ssd_resnet_v1_fpn_feature_extractor.SSDResnet101V1FpnFeatureExtractor(is_training, depth_multiplier, min_depth, pad_to_multiple, self.conv_hyperparams_fn, use_explicit_padding=use_explicit_padding)

    def _resnet_scope_name(self, use_keras):
        if False:
            return 10
        if use_keras:
            return 'ResNet101V1_FPN'
        return 'resnet_v1_101'

class SSDResnet152V1FeatureExtractorTest(ssd_resnet_v1_fpn_feature_extractor_testbase.SSDResnetFPNFeatureExtractorTestBase):
    """SSDResnet152v1Fpn feature extractor test."""

    def _create_feature_extractor(self, depth_multiplier, pad_to_multiple, use_explicit_padding=False, min_depth=32, use_keras=False):
        if False:
            print('Hello World!')
        is_training = True
        if use_keras:
            return ssd_resnet_v1_fpn_keras_feature_extractor.SSDResNet152V1FpnKerasFeatureExtractor(is_training=is_training, depth_multiplier=depth_multiplier, min_depth=min_depth, pad_to_multiple=pad_to_multiple, conv_hyperparams=self._build_conv_hyperparams(add_batch_norm=False), freeze_batchnorm=False, inplace_batchnorm_update=False, name='ResNet152V1_FPN')
        else:
            return ssd_resnet_v1_fpn_feature_extractor.SSDResnet152V1FpnFeatureExtractor(is_training, depth_multiplier, min_depth, pad_to_multiple, self.conv_hyperparams_fn, use_explicit_padding=use_explicit_padding)

    def _resnet_scope_name(self, use_keras):
        if False:
            while True:
                i = 10
        if use_keras:
            return 'ResNet152V1_FPN'
        return 'resnet_v1_152'
if __name__ == '__main__':
    tf.test.main()