"""Helper functions for SSD models meta architecture tests."""
import functools
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.contrib import slim as contrib_slim
from object_detection.builders import post_processing_builder
from object_detection.core import anchor_generator
from object_detection.core import balanced_positive_negative_sampler as sampler
from object_detection.core import box_list
from object_detection.core import losses
from object_detection.core import post_processing
from object_detection.core import region_similarity_calculator as sim_calc
from object_detection.core import target_assigner
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.protos import calibration_pb2
from object_detection.protos import model_pb2
from object_detection.utils import ops
from object_detection.utils import test_case
from object_detection.utils import test_utils
slim = contrib_slim
keras = tf.keras.layers

class FakeSSDFeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
    """Fake ssd feature extracture for ssd meta arch tests."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(FakeSSDFeatureExtractor, self).__init__(is_training=True, depth_multiplier=0, min_depth=0, pad_to_multiple=1, conv_hyperparams_fn=None)

    def preprocess(self, resized_inputs):
        if False:
            for i in range(10):
                print('nop')
        return tf.identity(resized_inputs)

    def extract_features(self, preprocessed_inputs):
        if False:
            i = 10
            return i + 15
        with tf.variable_scope('mock_model'):
            features = slim.conv2d(inputs=preprocessed_inputs, num_outputs=32, kernel_size=1, scope='layer1')
            return [features]

class FakeSSDKerasFeatureExtractor(ssd_meta_arch.SSDKerasFeatureExtractor):
    """Fake keras based ssd feature extracture for ssd meta arch tests."""

    def __init__(self):
        if False:
            print('Hello World!')
        with tf.name_scope('mock_model'):
            super(FakeSSDKerasFeatureExtractor, self).__init__(is_training=True, depth_multiplier=0, min_depth=0, pad_to_multiple=1, conv_hyperparams=None, freeze_batchnorm=False, inplace_batchnorm_update=False)
            self._conv = keras.Conv2D(filters=32, kernel_size=1, name='layer1')

    def preprocess(self, resized_inputs):
        if False:
            print('Hello World!')
        return tf.identity(resized_inputs)

    def _extract_features(self, preprocessed_inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        with tf.name_scope('mock_model'):
            return [self._conv(preprocessed_inputs)]

class MockAnchorGenerator2x2(anchor_generator.AnchorGenerator):
    """A simple 2x2 anchor grid on the unit square used for test only."""

    def name_scope(self):
        if False:
            while True:
                i = 10
        return 'MockAnchorGenerator'

    def num_anchors_per_location(self):
        if False:
            return 10
        return [1]

    def _generate(self, feature_map_shape_list, im_height, im_width):
        if False:
            i = 10
            return i + 15
        return [box_list.BoxList(tf.constant([[0, 0, 0.5, 0.5], [0, 0.5, 0.5, 1], [0.5, 0, 1, 0.5], [1.0, 1.0, 1.5, 1.5]], tf.float32))]

    def num_anchors(self):
        if False:
            for i in range(10):
                print('nop')
        return 4

class SSDMetaArchTestBase(test_case.TestCase):
    """Base class to test SSD based meta architectures."""

    def _create_model(self, model_fn=ssd_meta_arch.SSDMetaArch, apply_hard_mining=True, normalize_loc_loss_by_codesize=False, add_background_class=True, random_example_sampling=False, expected_loss_weights=model_pb2.DetectionModel().ssd.loss.NONE, min_num_negative_samples=1, desired_negative_sampling_ratio=3, use_keras=False, predict_mask=False, use_static_shapes=False, nms_max_size_per_class=5, calibration_mapping_value=None, return_raw_detections_during_predict=False):
        if False:
            print('Hello World!')
        is_training = False
        num_classes = 1
        mock_anchor_generator = MockAnchorGenerator2x2()
        if use_keras:
            mock_box_predictor = test_utils.MockKerasBoxPredictor(is_training, num_classes, add_background_class=add_background_class)
        else:
            mock_box_predictor = test_utils.MockBoxPredictor(is_training, num_classes, add_background_class=add_background_class)
        mock_box_coder = test_utils.MockBoxCoder()
        if use_keras:
            fake_feature_extractor = FakeSSDKerasFeatureExtractor()
        else:
            fake_feature_extractor = FakeSSDFeatureExtractor()
        mock_matcher = test_utils.MockMatcher()
        region_similarity_calculator = sim_calc.IouSimilarity()
        encode_background_as_zeros = False

        def image_resizer_fn(image):
            if False:
                return 10
            return [tf.identity(image), tf.shape(image)]
        classification_loss = losses.WeightedSigmoidClassificationLoss()
        localization_loss = losses.WeightedSmoothL1LocalizationLoss()
        non_max_suppression_fn = functools.partial(post_processing.batch_multiclass_non_max_suppression, score_thresh=-20.0, iou_thresh=1.0, max_size_per_class=nms_max_size_per_class, max_total_size=nms_max_size_per_class, use_static_shapes=use_static_shapes)
        score_conversion_fn = tf.identity
        calibration_config = calibration_pb2.CalibrationConfig()
        if calibration_mapping_value:
            calibration_text_proto = '\n      function_approximation {\n        x_y_pairs {\n            x_y_pair {\n              x: 0.0\n              y: %f\n            }\n            x_y_pair {\n              x: 1.0\n              y: %f\n            }}}' % (calibration_mapping_value, calibration_mapping_value)
            text_format.Merge(calibration_text_proto, calibration_config)
            score_conversion_fn = post_processing_builder._build_calibrated_score_converter(tf.identity, calibration_config)
        classification_loss_weight = 1.0
        localization_loss_weight = 1.0
        negative_class_weight = 1.0
        normalize_loss_by_num_matches = False
        hard_example_miner = None
        if apply_hard_mining:
            hard_example_miner = losses.HardExampleMiner(num_hard_examples=None, iou_threshold=1.0)
        random_example_sampler = None
        if random_example_sampling:
            random_example_sampler = sampler.BalancedPositiveNegativeSampler(positive_fraction=0.5)
        target_assigner_instance = target_assigner.TargetAssigner(region_similarity_calculator, mock_matcher, mock_box_coder, negative_class_weight=negative_class_weight)
        model_config = model_pb2.DetectionModel()
        if expected_loss_weights == model_config.ssd.loss.NONE:
            expected_loss_weights_fn = None
        else:
            raise ValueError('Not a valid value for expected_loss_weights.')
        code_size = 4
        kwargs = {}
        if predict_mask:
            kwargs.update({'mask_prediction_fn': test_utils.MockMaskHead(num_classes=1).predict})
        model = model_fn(is_training=is_training, anchor_generator=mock_anchor_generator, box_predictor=mock_box_predictor, box_coder=mock_box_coder, feature_extractor=fake_feature_extractor, encode_background_as_zeros=encode_background_as_zeros, image_resizer_fn=image_resizer_fn, non_max_suppression_fn=non_max_suppression_fn, score_conversion_fn=score_conversion_fn, classification_loss=classification_loss, localization_loss=localization_loss, classification_loss_weight=classification_loss_weight, localization_loss_weight=localization_loss_weight, normalize_loss_by_num_matches=normalize_loss_by_num_matches, hard_example_miner=hard_example_miner, target_assigner_instance=target_assigner_instance, add_summaries=False, normalize_loc_loss_by_codesize=normalize_loc_loss_by_codesize, freeze_batchnorm=False, inplace_batchnorm_update=False, add_background_class=add_background_class, random_example_sampler=random_example_sampler, expected_loss_weights_fn=expected_loss_weights_fn, return_raw_detections_during_predict=return_raw_detections_during_predict, **kwargs)
        return (model, num_classes, mock_anchor_generator.num_anchors(), code_size)

    def _get_value_for_matching_key(self, dictionary, suffix):
        if False:
            i = 10
            return i + 15
        for key in dictionary.keys():
            if key.endswith(suffix):
                return dictionary[key]
        raise ValueError('key not found {}'.format(suffix))
if __name__ == '__main__':
    tf.test.main()