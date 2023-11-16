"""Tests for ssd_mobilenet_edgetpu_feature_extractor."""
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from object_detection.models import ssd_mobilenet_edgetpu_feature_extractor
from object_detection.models import ssd_mobilenet_edgetpu_feature_extractor_testbase
slim = contrib_slim

class SsdMobilenetEdgeTPUFeatureExtractorTest(ssd_mobilenet_edgetpu_feature_extractor_testbase._SsdMobilenetEdgeTPUFeatureExtractorTestBase):

    def _get_input_sizes(self):
        if False:
            while True:
                i = 10
        'Return first two input feature map sizes.'
        return [384, 192]

    def _create_feature_extractor(self, depth_multiplier, pad_to_multiple, use_explicit_padding=False, use_keras=False):
        if False:
            return 10
        "Constructs a new MobileNetEdgeTPU feature extractor.\n\n    Args:\n      depth_multiplier: float depth multiplier for feature extractor\n      pad_to_multiple: the nearest multiple to zero pad the input height and\n        width dimensions to.\n      use_explicit_padding: use 'VALID' padding for convolutions, but prepad\n        inputs so that the output dimensions are the same as if 'SAME' padding\n        were used.\n      use_keras: if True builds a keras-based feature extractor, if False builds\n        a slim-based one.\n\n    Returns:\n      an ssd_meta_arch.SSDFeatureExtractor object.\n    "
        min_depth = 32
        return ssd_mobilenet_edgetpu_feature_extractor.SSDMobileNetEdgeTPUFeatureExtractor(False, depth_multiplier, min_depth, pad_to_multiple, self.conv_hyperparams_fn, use_explicit_padding=use_explicit_padding)
if __name__ == '__main__':
    tf.test.main()