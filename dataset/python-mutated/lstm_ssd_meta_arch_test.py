"""Tests for meta_architectures.lstm_ssd_meta_arch."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import numpy as np
import tensorflow as tf
from lstm_object_detection.lstm import lstm_cells
from lstm_object_detection.meta_architectures import lstm_ssd_meta_arch
from object_detection.core import anchor_generator
from object_detection.core import box_list
from object_detection.core import losses
from object_detection.core import post_processing
from object_detection.core import region_similarity_calculator as sim_calc
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.models import feature_map_generators
from object_detection.utils import test_case
from object_detection.utils import test_utils
slim = tf.contrib.slim
MAX_TOTAL_NUM_BOXES = 5
NUM_CLASSES = 1

class FakeLSTMFeatureExtractor(lstm_ssd_meta_arch.LSTMSSDFeatureExtractor):

    def __init__(self):
        if False:
            print('Hello World!')
        super(FakeLSTMFeatureExtractor, self).__init__(is_training=True, depth_multiplier=1.0, min_depth=0, pad_to_multiple=1, conv_hyperparams_fn=self.scope_fn)
        self._lstm_state_depth = 256

    def scope_fn(self):
        if False:
            print('Hello World!')
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu6) as sc:
            return sc

    def create_lstm_cell(self):
        if False:
            while True:
                i = 10
        pass

    def extract_features(self, preprocessed_inputs, state_saver=None, state_name='lstm_state', unroll_length=5, scope=None):
        if False:
            while True:
                i = 10
        with tf.variable_scope('mock_model'):
            net = slim.conv2d(inputs=preprocessed_inputs, num_outputs=32, kernel_size=1, scope='layer1')
            image_features = {'last_layer': net}
        self._states_out = {}
        feature_map_layout = {'from_layer': ['last_layer'], 'layer_depth': [-1], 'use_explicit_padding': self._use_explicit_padding, 'use_depthwise': self._use_depthwise}
        feature_maps = feature_map_generators.multi_resolution_feature_maps(feature_map_layout=feature_map_layout, depth_multiplier=self._depth_multiplier, min_depth=self._min_depth, insert_1x1_conv=True, image_features=image_features)
        return feature_maps.values()

class FakeLSTMInterleavedFeatureExtractor(lstm_ssd_meta_arch.LSTMSSDInterleavedFeatureExtractor):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(FakeLSTMInterleavedFeatureExtractor, self).__init__(is_training=True, depth_multiplier=1.0, min_depth=0, pad_to_multiple=1, conv_hyperparams_fn=self.scope_fn)
        self._lstm_state_depth = 256

    def scope_fn(self):
        if False:
            return 10
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu6) as sc:
            return sc

    def create_lstm_cell(self):
        if False:
            while True:
                i = 10
        pass

    def extract_base_features_large(self, preprocessed_inputs):
        if False:
            for i in range(10):
                print('nop')
        with tf.variable_scope('base_large'):
            net = slim.conv2d(inputs=preprocessed_inputs, num_outputs=32, kernel_size=1, scope='layer1')
        return net

    def extract_base_features_small(self, preprocessed_inputs):
        if False:
            for i in range(10):
                print('nop')
        with tf.variable_scope('base_small'):
            net = slim.conv2d(inputs=preprocessed_inputs, num_outputs=32, kernel_size=1, scope='layer1')
        return net

    def extract_features(self, preprocessed_inputs, state_saver=None, state_name='lstm_state', unroll_length=5, scope=None):
        if False:
            print('Hello World!')
        with tf.variable_scope('mock_model'):
            net_large = self.extract_base_features_large(preprocessed_inputs)
            net_small = self.extract_base_features_small(preprocessed_inputs)
            net = slim.conv2d(inputs=tf.concat([net_large, net_small], axis=3), num_outputs=32, kernel_size=1, scope='layer1')
            image_features = {'last_layer': net}
        self._states_out = {}
        feature_map_layout = {'from_layer': ['last_layer'], 'layer_depth': [-1], 'use_explicit_padding': self._use_explicit_padding, 'use_depthwise': self._use_depthwise}
        feature_maps = feature_map_generators.multi_resolution_feature_maps(feature_map_layout=feature_map_layout, depth_multiplier=self._depth_multiplier, min_depth=self._min_depth, insert_1x1_conv=True, image_features=image_features)
        return feature_maps.values()

class MockAnchorGenerator2x2(anchor_generator.AnchorGenerator):
    """Sets up a simple 2x2 anchor grid on the unit square."""

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
            return 10
        return [box_list.BoxList(tf.constant([[0, 0, 0.5, 0.5], [0, 0.5, 0.5, 1], [0.5, 0, 1, 0.5], [1.0, 1.0, 1.5, 1.5]], tf.float32))]

    def num_anchors(self):
        if False:
            for i in range(10):
                print('nop')
        return 4

class LSTMSSDMetaArchTest(test_case.TestCase):

    def _create_model(self, interleaved=False, apply_hard_mining=True, normalize_loc_loss_by_codesize=False, add_background_class=True, random_example_sampling=False, use_expected_classification_loss_under_sampling=False, min_num_negative_samples=1, desired_negative_sampling_ratio=3, unroll_length=1):
        if False:
            while True:
                i = 10
        num_classes = NUM_CLASSES
        is_training = False
        mock_anchor_generator = MockAnchorGenerator2x2()
        mock_box_predictor = test_utils.MockBoxPredictor(is_training, num_classes)
        mock_box_coder = test_utils.MockBoxCoder()
        if interleaved:
            fake_feature_extractor = FakeLSTMInterleavedFeatureExtractor()
        else:
            fake_feature_extractor = FakeLSTMFeatureExtractor()
        mock_matcher = test_utils.MockMatcher()
        region_similarity_calculator = sim_calc.IouSimilarity()
        encode_background_as_zeros = False

        def image_resizer_fn(image):
            if False:
                while True:
                    i = 10
            return [tf.identity(image), tf.shape(image)]
        classification_loss = losses.WeightedSigmoidClassificationLoss()
        localization_loss = losses.WeightedSmoothL1LocalizationLoss()
        non_max_suppression_fn = functools.partial(post_processing.batch_multiclass_non_max_suppression, score_thresh=-20.0, iou_thresh=1.0, max_size_per_class=5, max_total_size=MAX_TOTAL_NUM_BOXES)
        classification_loss_weight = 1.0
        localization_loss_weight = 1.0
        negative_class_weight = 1.0
        normalize_loss_by_num_matches = False
        hard_example_miner = None
        if apply_hard_mining:
            hard_example_miner = losses.HardExampleMiner(num_hard_examples=None, iou_threshold=1.0)
        target_assigner_instance = target_assigner.TargetAssigner(region_similarity_calculator, mock_matcher, mock_box_coder, negative_class_weight=negative_class_weight)
        code_size = 4
        model = lstm_ssd_meta_arch.LSTMSSDMetaArch(is_training=is_training, anchor_generator=mock_anchor_generator, box_predictor=mock_box_predictor, box_coder=mock_box_coder, feature_extractor=fake_feature_extractor, encode_background_as_zeros=encode_background_as_zeros, image_resizer_fn=image_resizer_fn, non_max_suppression_fn=non_max_suppression_fn, score_conversion_fn=tf.identity, classification_loss=classification_loss, localization_loss=localization_loss, classification_loss_weight=classification_loss_weight, localization_loss_weight=localization_loss_weight, normalize_loss_by_num_matches=normalize_loss_by_num_matches, hard_example_miner=hard_example_miner, unroll_length=unroll_length, target_assigner_instance=target_assigner_instance, add_summaries=False)
        return (model, num_classes, mock_anchor_generator.num_anchors(), code_size)

    def _get_value_for_matching_key(self, dictionary, suffix):
        if False:
            print('Hello World!')
        for key in dictionary.keys():
            if key.endswith(suffix):
                return dictionary[key]
        raise ValueError('key not found {}'.format(suffix))

    def test_predict_returns_correct_items_and_sizes(self):
        if False:
            while True:
                i = 10
        batch_size = 3
        height = width = 2
        num_unroll = 1
        graph = tf.Graph()
        with graph.as_default():
            (model, num_classes, num_anchors, code_size) = self._create_model()
            preprocessed_images = tf.random_uniform([batch_size * num_unroll, height, width, 3], minval=-1.0, maxval=1.0)
            true_image_shapes = tf.tile([[height, width, 3]], [batch_size, 1])
            prediction_dict = model.predict(preprocessed_images, true_image_shapes)
            self.assertIn('preprocessed_inputs', prediction_dict)
            self.assertIn('box_encodings', prediction_dict)
            self.assertIn('class_predictions_with_background', prediction_dict)
            self.assertIn('feature_maps', prediction_dict)
            self.assertIn('anchors', prediction_dict)
            self.assertAllEqual([batch_size * num_unroll, height, width, 3], prediction_dict['preprocessed_inputs'].shape.as_list())
            self.assertAllEqual([batch_size * num_unroll, num_anchors, code_size], prediction_dict['box_encodings'].shape.as_list())
            self.assertAllEqual([batch_size * num_unroll, num_anchors, num_classes + 1], prediction_dict['class_predictions_with_background'].shape.as_list())
            self.assertAllEqual([num_anchors, code_size], prediction_dict['anchors'].shape.as_list())

    def test_interleaved_predict_returns_correct_items_and_sizes(self):
        if False:
            print('Hello World!')
        batch_size = 3
        height = width = 2
        num_unroll = 1
        graph = tf.Graph()
        with graph.as_default():
            (model, num_classes, num_anchors, code_size) = self._create_model(interleaved=True)
            preprocessed_images = tf.random_uniform([batch_size * num_unroll, height, width, 3], minval=-1.0, maxval=1.0)
            true_image_shapes = tf.tile([[height, width, 3]], [batch_size, 1])
            prediction_dict = model.predict(preprocessed_images, true_image_shapes)
            self.assertIn('preprocessed_inputs', prediction_dict)
            self.assertIn('box_encodings', prediction_dict)
            self.assertIn('class_predictions_with_background', prediction_dict)
            self.assertIn('feature_maps', prediction_dict)
            self.assertIn('anchors', prediction_dict)
            self.assertAllEqual([batch_size * num_unroll, height, width, 3], prediction_dict['preprocessed_inputs'].shape.as_list())
            self.assertAllEqual([batch_size * num_unroll, num_anchors, code_size], prediction_dict['box_encodings'].shape.as_list())
            self.assertAllEqual([batch_size * num_unroll, num_anchors, num_classes + 1], prediction_dict['class_predictions_with_background'].shape.as_list())
            self.assertAllEqual([num_anchors, code_size], prediction_dict['anchors'].shape.as_list())
if __name__ == '__main__':
    tf.test.main()