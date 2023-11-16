"""Tests for object_detection.meta_architectures.faster_rcnn_meta_arch."""
import functools
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.contrib import slim as contrib_slim
from object_detection.anchor_generators import grid_anchor_generator
from object_detection.builders import box_predictor_builder
from object_detection.builders import hyperparams_builder
from object_detection.builders import post_processing_builder
from object_detection.core import balanced_positive_negative_sampler as sampler
from object_detection.core import losses
from object_detection.core import post_processing
from object_detection.core import target_assigner
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.protos import box_predictor_pb2
from object_detection.protos import hyperparams_pb2
from object_detection.protos import post_processing_pb2
from object_detection.utils import ops
from object_detection.utils import test_case
from object_detection.utils import test_utils
slim = contrib_slim
BOX_CODE_SIZE = 4

class FakeFasterRCNNFeatureExtractor(faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
    """Fake feature extractor to use in tests."""

    def __init__(self):
        if False:
            print('Hello World!')
        super(FakeFasterRCNNFeatureExtractor, self).__init__(is_training=False, first_stage_features_stride=32, reuse_weights=None, weight_decay=0.0)

    def preprocess(self, resized_inputs):
        if False:
            while True:
                i = 10
        return tf.identity(resized_inputs)

    def _extract_proposal_features(self, preprocessed_inputs, scope):
        if False:
            print('Hello World!')
        with tf.variable_scope('mock_model'):
            proposal_features = 0 * slim.conv2d(preprocessed_inputs, num_outputs=3, kernel_size=1, scope='layer1')
            return (proposal_features, {})

    def _extract_box_classifier_features(self, proposal_feature_maps, scope):
        if False:
            i = 10
            return i + 15
        with tf.variable_scope('mock_model'):
            return 0 * slim.conv2d(proposal_feature_maps, num_outputs=3, kernel_size=1, scope='layer2')

class FakeFasterRCNNKerasFeatureExtractor(faster_rcnn_meta_arch.FasterRCNNKerasFeatureExtractor):
    """Fake feature extractor to use in tests."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(FakeFasterRCNNKerasFeatureExtractor, self).__init__(is_training=False, first_stage_features_stride=32, weight_decay=0.0)

    def preprocess(self, resized_inputs):
        if False:
            while True:
                i = 10
        return tf.identity(resized_inputs)

    def get_proposal_feature_extractor_model(self, name):
        if False:
            print('Hello World!')

        class ProposalFeatureExtractor(tf.keras.Model):
            """Dummy proposal feature extraction."""

            def __init__(self, name):
                if False:
                    while True:
                        i = 10
                super(ProposalFeatureExtractor, self).__init__(name=name)
                self.conv = None

            def build(self, input_shape):
                if False:
                    for i in range(10):
                        print('nop')
                self.conv = tf.keras.layers.Conv2D(3, kernel_size=1, padding='SAME', name='layer1')

            def call(self, inputs):
                if False:
                    print('Hello World!')
                return self.conv(inputs)
        return ProposalFeatureExtractor(name=name)

    def get_box_classifier_feature_extractor_model(self, name):
        if False:
            i = 10
            return i + 15
        return tf.keras.Sequential([tf.keras.layers.Conv2D(3, kernel_size=1, padding='SAME', name=name + '_layer2')])

class FasterRCNNMetaArchTestBase(test_case.TestCase, parameterized.TestCase):
    """Base class to test Faster R-CNN and R-FCN meta architectures."""

    def _build_arg_scope_with_hyperparams(self, hyperparams_text_proto, is_training):
        if False:
            while True:
                i = 10
        hyperparams = hyperparams_pb2.Hyperparams()
        text_format.Merge(hyperparams_text_proto, hyperparams)
        return hyperparams_builder.build(hyperparams, is_training=is_training)

    def _build_keras_layer_hyperparams(self, hyperparams_text_proto):
        if False:
            return 10
        hyperparams = hyperparams_pb2.Hyperparams()
        text_format.Merge(hyperparams_text_proto, hyperparams)
        return hyperparams_builder.KerasLayerHyperparams(hyperparams)

    def _get_second_stage_box_predictor_text_proto(self, share_box_across_classes=False):
        if False:
            i = 10
            return i + 15
        share_box_field = 'true' if share_box_across_classes else 'false'
        box_predictor_text_proto = '\n      mask_rcnn_box_predictor {{\n        fc_hyperparams {{\n          op: FC\n          activation: NONE\n          regularizer {{\n            l2_regularizer {{\n              weight: 0.0005\n            }}\n          }}\n          initializer {{\n            variance_scaling_initializer {{\n              factor: 1.0\n              uniform: true\n              mode: FAN_AVG\n            }}\n          }}\n        }}\n        share_box_across_classes: {share_box_across_classes}\n      }}\n    '.format(share_box_across_classes=share_box_field)
        return box_predictor_text_proto

    def _add_mask_to_second_stage_box_predictor_text_proto(self, masks_are_class_agnostic=False):
        if False:
            while True:
                i = 10
        agnostic = 'true' if masks_are_class_agnostic else 'false'
        box_predictor_text_proto = '\n      mask_rcnn_box_predictor {\n        predict_instance_masks: true\n        masks_are_class_agnostic: ' + agnostic + '\n        mask_height: 14\n        mask_width: 14\n        conv_hyperparams {\n          op: CONV\n          regularizer {\n            l2_regularizer {\n              weight: 0.0\n            }\n          }\n          initializer {\n            truncated_normal_initializer {\n              stddev: 0.01\n            }\n          }\n        }\n      }\n    '
        return box_predictor_text_proto

    def _get_second_stage_box_predictor(self, num_classes, is_training, predict_masks, masks_are_class_agnostic, share_box_across_classes=False, use_keras=False):
        if False:
            i = 10
            return i + 15
        box_predictor_proto = box_predictor_pb2.BoxPredictor()
        text_format.Merge(self._get_second_stage_box_predictor_text_proto(share_box_across_classes), box_predictor_proto)
        if predict_masks:
            text_format.Merge(self._add_mask_to_second_stage_box_predictor_text_proto(masks_are_class_agnostic), box_predictor_proto)
        if use_keras:
            return box_predictor_builder.build_keras(hyperparams_builder.KerasLayerHyperparams, inplace_batchnorm_update=False, freeze_batchnorm=False, box_predictor_config=box_predictor_proto, num_classes=num_classes, num_predictions_per_location_list=None, is_training=is_training)
        else:
            return box_predictor_builder.build(hyperparams_builder.build, box_predictor_proto, num_classes=num_classes, is_training=is_training)

    def _get_model(self, box_predictor, keras_model=False, **common_kwargs):
        if False:
            for i in range(10):
                print('nop')
        return faster_rcnn_meta_arch.FasterRCNNMetaArch(initial_crop_size=3, maxpool_kernel_size=1, maxpool_stride=1, second_stage_mask_rcnn_box_predictor=box_predictor, **common_kwargs)

    def _build_model(self, is_training, number_of_stages, second_stage_batch_size, use_keras=False, first_stage_max_proposals=8, num_classes=2, hard_mining=False, softmax_second_stage_classification_loss=True, predict_masks=False, pad_to_max_dimension=None, masks_are_class_agnostic=False, use_matmul_crop_and_resize=False, clip_anchors_to_image=False, use_matmul_gather_in_matcher=False, use_static_shapes=False, calibration_mapping_value=None, share_box_across_classes=False, return_raw_detections_during_predict=False):
        if False:
            while True:
                i = 10

        def image_resizer_fn(image, masks=None):
            if False:
                i = 10
                return i + 15
            'Fake image resizer function.'
            resized_inputs = []
            resized_image = tf.identity(image)
            if pad_to_max_dimension is not None:
                resized_image = tf.image.pad_to_bounding_box(image, 0, 0, pad_to_max_dimension, pad_to_max_dimension)
            resized_inputs.append(resized_image)
            if masks is not None:
                resized_masks = tf.identity(masks)
                if pad_to_max_dimension is not None:
                    resized_masks = tf.image.pad_to_bounding_box(tf.transpose(masks, [1, 2, 0]), 0, 0, pad_to_max_dimension, pad_to_max_dimension)
                    resized_masks = tf.transpose(resized_masks, [2, 0, 1])
                resized_inputs.append(resized_masks)
            resized_inputs.append(tf.shape(image))
            return resized_inputs
        first_stage_anchor_scales = (0.001, 0.005, 0.1)
        first_stage_anchor_aspect_ratios = (0.5, 1.0, 2.0)
        first_stage_anchor_strides = (1, 1)
        first_stage_anchor_generator = grid_anchor_generator.GridAnchorGenerator(first_stage_anchor_scales, first_stage_anchor_aspect_ratios, anchor_stride=first_stage_anchor_strides)
        first_stage_target_assigner = target_assigner.create_target_assigner('FasterRCNN', 'proposal', use_matmul_gather=use_matmul_gather_in_matcher)
        if use_keras:
            fake_feature_extractor = FakeFasterRCNNKerasFeatureExtractor()
        else:
            fake_feature_extractor = FakeFasterRCNNFeatureExtractor()
        first_stage_box_predictor_hyperparams_text_proto = '\n      op: CONV\n      activation: RELU\n      regularizer {\n        l2_regularizer {\n          weight: 0.00004\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n          stddev: 0.03\n        }\n      }\n    '
        if use_keras:
            first_stage_box_predictor_arg_scope_fn = self._build_keras_layer_hyperparams(first_stage_box_predictor_hyperparams_text_proto)
        else:
            first_stage_box_predictor_arg_scope_fn = self._build_arg_scope_with_hyperparams(first_stage_box_predictor_hyperparams_text_proto, is_training)
        first_stage_box_predictor_kernel_size = 3
        first_stage_atrous_rate = 1
        first_stage_box_predictor_depth = 512
        first_stage_minibatch_size = 3
        first_stage_sampler = sampler.BalancedPositiveNegativeSampler(positive_fraction=0.5, is_static=use_static_shapes)
        first_stage_nms_score_threshold = -1.0
        first_stage_nms_iou_threshold = 1.0
        first_stage_max_proposals = first_stage_max_proposals
        first_stage_non_max_suppression_fn = functools.partial(post_processing.batch_multiclass_non_max_suppression, score_thresh=first_stage_nms_score_threshold, iou_thresh=first_stage_nms_iou_threshold, max_size_per_class=first_stage_max_proposals, max_total_size=first_stage_max_proposals, use_static_shapes=use_static_shapes)
        first_stage_localization_loss_weight = 1.0
        first_stage_objectness_loss_weight = 1.0
        post_processing_config = post_processing_pb2.PostProcessing()
        post_processing_text_proto = '\n      score_converter: IDENTITY\n      batch_non_max_suppression {\n        score_threshold: -20.0\n        iou_threshold: 1.0\n        max_detections_per_class: 5\n        max_total_detections: 5\n        use_static_shapes: ' + '{}'.format(use_static_shapes) + '\n      }\n    '
        if calibration_mapping_value:
            calibration_text_proto = '\n      calibration_config {\n        function_approximation {\n          x_y_pairs {\n            x_y_pair {\n              x: 0.0\n              y: %f\n            }\n            x_y_pair {\n              x: 1.0\n              y: %f\n              }}}}' % (calibration_mapping_value, calibration_mapping_value)
            post_processing_text_proto = post_processing_text_proto + ' ' + calibration_text_proto
        text_format.Merge(post_processing_text_proto, post_processing_config)
        (second_stage_non_max_suppression_fn, second_stage_score_conversion_fn) = post_processing_builder.build(post_processing_config)
        second_stage_target_assigner = target_assigner.create_target_assigner('FasterRCNN', 'detection', use_matmul_gather=use_matmul_gather_in_matcher)
        second_stage_sampler = sampler.BalancedPositiveNegativeSampler(positive_fraction=1.0, is_static=use_static_shapes)
        second_stage_localization_loss_weight = 1.0
        second_stage_classification_loss_weight = 1.0
        if softmax_second_stage_classification_loss:
            second_stage_classification_loss = losses.WeightedSoftmaxClassificationLoss()
        else:
            second_stage_classification_loss = losses.WeightedSigmoidClassificationLoss()
        hard_example_miner = None
        if hard_mining:
            hard_example_miner = losses.HardExampleMiner(num_hard_examples=1, iou_threshold=0.99, loss_type='both', cls_loss_weight=second_stage_classification_loss_weight, loc_loss_weight=second_stage_localization_loss_weight, max_negatives_per_positive=None)
        crop_and_resize_fn = ops.matmul_crop_and_resize if use_matmul_crop_and_resize else ops.native_crop_and_resize
        common_kwargs = {'is_training': is_training, 'num_classes': num_classes, 'image_resizer_fn': image_resizer_fn, 'feature_extractor': fake_feature_extractor, 'number_of_stages': number_of_stages, 'first_stage_anchor_generator': first_stage_anchor_generator, 'first_stage_target_assigner': first_stage_target_assigner, 'first_stage_atrous_rate': first_stage_atrous_rate, 'first_stage_box_predictor_arg_scope_fn': first_stage_box_predictor_arg_scope_fn, 'first_stage_box_predictor_kernel_size': first_stage_box_predictor_kernel_size, 'first_stage_box_predictor_depth': first_stage_box_predictor_depth, 'first_stage_minibatch_size': first_stage_minibatch_size, 'first_stage_sampler': first_stage_sampler, 'first_stage_non_max_suppression_fn': first_stage_non_max_suppression_fn, 'first_stage_max_proposals': first_stage_max_proposals, 'first_stage_localization_loss_weight': first_stage_localization_loss_weight, 'first_stage_objectness_loss_weight': first_stage_objectness_loss_weight, 'second_stage_target_assigner': second_stage_target_assigner, 'second_stage_batch_size': second_stage_batch_size, 'second_stage_sampler': second_stage_sampler, 'second_stage_non_max_suppression_fn': second_stage_non_max_suppression_fn, 'second_stage_score_conversion_fn': second_stage_score_conversion_fn, 'second_stage_localization_loss_weight': second_stage_localization_loss_weight, 'second_stage_classification_loss_weight': second_stage_classification_loss_weight, 'second_stage_classification_loss': second_stage_classification_loss, 'hard_example_miner': hard_example_miner, 'crop_and_resize_fn': crop_and_resize_fn, 'clip_anchors_to_image': clip_anchors_to_image, 'use_static_shapes': use_static_shapes, 'resize_masks': True, 'return_raw_detections_during_predict': return_raw_detections_during_predict}
        return self._get_model(self._get_second_stage_box_predictor(num_classes=num_classes, is_training=is_training, use_keras=use_keras, predict_masks=predict_masks, masks_are_class_agnostic=masks_are_class_agnostic, share_box_across_classes=share_box_across_classes), **common_kwargs)

    @parameterized.parameters({'use_static_shapes': False, 'use_keras': True}, {'use_static_shapes': False, 'use_keras': False}, {'use_static_shapes': True, 'use_keras': True}, {'use_static_shapes': True, 'use_keras': False})
    def test_predict_gives_correct_shapes_in_inference_mode_first_stage_only(self, use_static_shapes=False, use_keras=False):
        if False:
            print('Hello World!')
        batch_size = 2
        height = 10
        width = 12
        input_image_shape = (batch_size, height, width, 3)

        def graph_fn(images):
            if False:
                print('Hello World!')
            'Function to construct tf graph for the test.'
            model = self._build_model(is_training=False, use_keras=use_keras, number_of_stages=1, second_stage_batch_size=2, clip_anchors_to_image=use_static_shapes, use_static_shapes=use_static_shapes)
            (preprocessed_inputs, true_image_shapes) = model.preprocess(images)
            prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
            return (prediction_dict['rpn_box_predictor_features'], prediction_dict['rpn_features_to_crop'], prediction_dict['image_shape'], prediction_dict['rpn_box_encodings'], prediction_dict['rpn_objectness_predictions_with_background'], prediction_dict['anchors'])
        images = np.zeros(input_image_shape, dtype=np.float32)
        expected_num_anchors = height * width * 3 * 3
        expected_output_shapes = {'rpn_box_predictor_features': (batch_size, height, width, 512), 'rpn_features_to_crop': (batch_size, height, width, 3), 'rpn_box_encodings': (batch_size, expected_num_anchors, 4), 'rpn_objectness_predictions_with_background': (batch_size, expected_num_anchors, 2), 'anchors': (expected_num_anchors, 4)}
        if use_static_shapes:
            results = self.execute(graph_fn, [images])
        else:
            results = self.execute_cpu(graph_fn, [images])
        self.assertAllEqual(results[0].shape, expected_output_shapes['rpn_box_predictor_features'])
        self.assertAllEqual(results[1].shape, expected_output_shapes['rpn_features_to_crop'])
        self.assertAllEqual(results[2], input_image_shape)
        self.assertAllEqual(results[3].shape, expected_output_shapes['rpn_box_encodings'])
        self.assertAllEqual(results[4].shape, expected_output_shapes['rpn_objectness_predictions_with_background'])
        self.assertAllEqual(results[5].shape, expected_output_shapes['anchors'])
        anchors = results[5]
        self.assertTrue(np.all(np.greater_equal(anchors, 0)))
        self.assertTrue(np.all(np.less_equal(anchors[:, 0], height)))
        self.assertTrue(np.all(np.less_equal(anchors[:, 1], width)))
        self.assertTrue(np.all(np.less_equal(anchors[:, 2], height)))
        self.assertTrue(np.all(np.less_equal(anchors[:, 3], width)))

    @parameterized.parameters({'use_keras': True}, {'use_keras': False})
    def test_regularization_losses(self, use_keras=False):
        if False:
            print('Hello World!')
        test_graph = tf.Graph()
        with test_graph.as_default():
            model = self._build_model(is_training=True, use_keras=use_keras, number_of_stages=1, second_stage_batch_size=2)
            batch_size = 2
            height = 10
            width = 12
            input_image_shape = (batch_size, height, width, 3)
            (_, true_image_shapes) = model.preprocess(tf.zeros(input_image_shape))
            preprocessed_inputs = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, None, 3))
            model.predict(preprocessed_inputs, true_image_shapes)
            reg_losses = tf.math.add_n(model.regularization_losses())
            init_op = tf.global_variables_initializer()
            with self.test_session(graph=test_graph) as sess:
                sess.run(init_op)
                self.assertGreaterEqual(sess.run(reg_losses), 0)

    @parameterized.parameters({'use_keras': True}, {'use_keras': False})
    def test_predict_gives_valid_anchors_in_training_mode_first_stage_only(self, use_keras=False):
        if False:
            while True:
                i = 10
        test_graph = tf.Graph()
        with test_graph.as_default():
            model = self._build_model(is_training=True, use_keras=use_keras, number_of_stages=1, second_stage_batch_size=2)
            batch_size = 2
            height = 10
            width = 12
            input_image_shape = (batch_size, height, width, 3)
            (_, true_image_shapes) = model.preprocess(tf.zeros(input_image_shape))
            preprocessed_inputs = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, None, 3))
            prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
            expected_output_keys = set(['rpn_box_predictor_features', 'rpn_features_to_crop', 'image_shape', 'rpn_box_encodings', 'rpn_objectness_predictions_with_background', 'anchors', 'feature_maps'])
            num_anchors_strict_upper_bound = height * width * 3 * 3
            init_op = tf.global_variables_initializer()
            with self.test_session(graph=test_graph) as sess:
                sess.run(init_op)
                prediction_out = sess.run(prediction_dict, feed_dict={preprocessed_inputs: np.zeros(input_image_shape)})
                self.assertEqual(set(prediction_out.keys()), expected_output_keys)
                self.assertAllEqual(prediction_out['image_shape'], input_image_shape)
                anchors = prediction_out['anchors']
                self.assertTrue(len(anchors.shape) == 2 and anchors.shape[1] == 4)
                num_anchors_out = anchors.shape[0]
                self.assertLess(num_anchors_out, num_anchors_strict_upper_bound)
                self.assertTrue(np.all(np.greater_equal(anchors, 0)))
                self.assertTrue(np.all(np.less_equal(anchors[:, 0], height)))
                self.assertTrue(np.all(np.less_equal(anchors[:, 1], width)))
                self.assertTrue(np.all(np.less_equal(anchors[:, 2], height)))
                self.assertTrue(np.all(np.less_equal(anchors[:, 3], width)))
                self.assertAllEqual(prediction_out['rpn_box_encodings'].shape, (batch_size, num_anchors_out, 4))
                self.assertAllEqual(prediction_out['rpn_objectness_predictions_with_background'].shape, (batch_size, num_anchors_out, 2))

    @parameterized.parameters({'use_static_shapes': False, 'use_keras': True}, {'use_static_shapes': False, 'use_keras': False}, {'use_static_shapes': True, 'use_keras': True}, {'use_static_shapes': True, 'use_keras': False})
    def test_predict_correct_shapes_in_inference_mode_two_stages(self, use_static_shapes=False, use_keras=False):
        if False:
            while True:
                i = 10

        def compare_results(results, expected_output_shapes):
            if False:
                for i in range(10):
                    print('nop')
            'Checks if the shape of the predictions are as expected.'
            self.assertAllEqual(results[0].shape, expected_output_shapes['rpn_box_predictor_features'])
            self.assertAllEqual(results[1].shape, expected_output_shapes['rpn_features_to_crop'])
            self.assertAllEqual(results[2].shape, expected_output_shapes['image_shape'])
            self.assertAllEqual(results[3].shape, expected_output_shapes['rpn_box_encodings'])
            self.assertAllEqual(results[4].shape, expected_output_shapes['rpn_objectness_predictions_with_background'])
            self.assertAllEqual(results[5].shape, expected_output_shapes['anchors'])
            self.assertAllEqual(results[6].shape, expected_output_shapes['refined_box_encodings'])
            self.assertAllEqual(results[7].shape, expected_output_shapes['class_predictions_with_background'])
            self.assertAllEqual(results[8].shape, expected_output_shapes['num_proposals'])
            self.assertAllEqual(results[9].shape, expected_output_shapes['proposal_boxes'])
            self.assertAllEqual(results[10].shape, expected_output_shapes['proposal_boxes_normalized'])
            self.assertAllEqual(results[11].shape, expected_output_shapes['box_classifier_features'])
            self.assertAllEqual(results[12].shape, expected_output_shapes['final_anchors'])
        batch_size = 2
        image_size = 10
        max_num_proposals = 8
        initial_crop_size = 3
        maxpool_stride = 1
        input_shapes = [(batch_size, image_size, image_size, 3), (None, image_size, image_size, 3), (batch_size, None, None, 3), (None, None, None, 3)]

        def graph_fn_tpu(images):
            if False:
                for i in range(10):
                    print('nop')
            'Function to construct tf graph for the test.'
            model = self._build_model(is_training=False, use_keras=use_keras, number_of_stages=2, second_stage_batch_size=2, predict_masks=False, use_matmul_crop_and_resize=use_static_shapes, clip_anchors_to_image=use_static_shapes, use_static_shapes=use_static_shapes)
            (preprocessed_inputs, true_image_shapes) = model.preprocess(images)
            prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
            return (prediction_dict['rpn_box_predictor_features'], prediction_dict['rpn_features_to_crop'], prediction_dict['image_shape'], prediction_dict['rpn_box_encodings'], prediction_dict['rpn_objectness_predictions_with_background'], prediction_dict['anchors'], prediction_dict['refined_box_encodings'], prediction_dict['class_predictions_with_background'], prediction_dict['num_proposals'], prediction_dict['proposal_boxes'], prediction_dict['proposal_boxes_normalized'], prediction_dict['box_classifier_features'], prediction_dict['final_anchors'])
        expected_num_anchors = image_size * image_size * 3 * 3
        expected_shapes = {'rpn_box_predictor_features': (2, image_size, image_size, 512), 'rpn_features_to_crop': (2, image_size, image_size, 3), 'image_shape': (4,), 'rpn_box_encodings': (2, expected_num_anchors, 4), 'rpn_objectness_predictions_with_background': (2, expected_num_anchors, 2), 'anchors': (expected_num_anchors, 4), 'refined_box_encodings': (2 * max_num_proposals, 2, 4), 'class_predictions_with_background': (2 * max_num_proposals, 2 + 1), 'num_proposals': (2,), 'proposal_boxes': (2, max_num_proposals, 4), 'proposal_boxes_normalized': (2, max_num_proposals, 4), 'box_classifier_features': self._get_box_classifier_features_shape(image_size, batch_size, max_num_proposals, initial_crop_size, maxpool_stride, 3), 'feature_maps': [(2, image_size, image_size, 512)], 'final_anchors': (2, max_num_proposals, 4)}
        if use_static_shapes:
            input_shape = (batch_size, image_size, image_size, 3)
            images = np.zeros(input_shape, dtype=np.float32)
            results = self.execute(graph_fn_tpu, [images])
            compare_results(results, expected_shapes)
        else:
            for input_shape in input_shapes:
                test_graph = tf.Graph()
                with test_graph.as_default():
                    model = self._build_model(is_training=False, use_keras=use_keras, number_of_stages=2, second_stage_batch_size=2, predict_masks=False)
                    preprocessed_inputs = tf.placeholder(tf.float32, shape=input_shape)
                    (_, true_image_shapes) = model.preprocess(preprocessed_inputs)
                    result_tensor_dict = model.predict(preprocessed_inputs, true_image_shapes)
                    init_op = tf.global_variables_initializer()
                with self.test_session(graph=test_graph) as sess:
                    sess.run(init_op)
                    tensor_dict_out = sess.run(result_tensor_dict, feed_dict={preprocessed_inputs: np.zeros((batch_size, image_size, image_size, 3))})
                self.assertEqual(set(tensor_dict_out.keys()), set(expected_shapes.keys()))
                for key in expected_shapes:
                    if isinstance(tensor_dict_out[key], list):
                        continue
                    self.assertAllEqual(tensor_dict_out[key].shape, expected_shapes[key])

    @parameterized.parameters({'use_static_shapes': False, 'use_keras': True}, {'use_static_shapes': False, 'use_keras': False}, {'use_static_shapes': True, 'use_keras': True}, {'use_static_shapes': True, 'use_keras': False})
    def test_predict_gives_correct_shapes_in_train_mode_both_stages(self, use_static_shapes=False, use_keras=False):
        if False:
            return 10
        batch_size = 2
        image_size = 10
        max_num_proposals = 7
        initial_crop_size = 3
        maxpool_stride = 1

        def graph_fn(images, gt_boxes, gt_classes, gt_weights):
            if False:
                while True:
                    i = 10
            'Function to construct tf graph for the test.'
            model = self._build_model(is_training=True, use_keras=use_keras, number_of_stages=2, second_stage_batch_size=7, predict_masks=False, use_matmul_crop_and_resize=use_static_shapes, clip_anchors_to_image=use_static_shapes, use_static_shapes=use_static_shapes)
            (preprocessed_inputs, true_image_shapes) = model.preprocess(images)
            model.provide_groundtruth(groundtruth_boxes_list=tf.unstack(gt_boxes), groundtruth_classes_list=tf.unstack(gt_classes), groundtruth_weights_list=tf.unstack(gt_weights))
            result_tensor_dict = model.predict(preprocessed_inputs, true_image_shapes)
            updates = model.updates()
            return (result_tensor_dict['refined_box_encodings'], result_tensor_dict['class_predictions_with_background'], result_tensor_dict['proposal_boxes'], result_tensor_dict['proposal_boxes_normalized'], result_tensor_dict['anchors'], result_tensor_dict['rpn_box_encodings'], result_tensor_dict['rpn_objectness_predictions_with_background'], result_tensor_dict['rpn_features_to_crop'], result_tensor_dict['rpn_box_predictor_features'], updates, result_tensor_dict['final_anchors'])
        image_shape = (batch_size, image_size, image_size, 3)
        images = np.zeros(image_shape, dtype=np.float32)
        gt_boxes = np.stack([np.array([[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]], dtype=np.float32), np.array([[0, 0.5, 0.5, 1], [0.5, 0, 1, 0.5]], dtype=np.float32)])
        gt_classes = np.stack([np.array([[1, 0], [0, 1]], dtype=np.float32), np.array([[1, 0], [1, 0]], dtype=np.float32)])
        gt_weights = np.stack([np.array([1, 1], dtype=np.float32), np.array([1, 1], dtype=np.float32)])
        if use_static_shapes:
            results = self.execute(graph_fn, [images, gt_boxes, gt_classes, gt_weights])
        else:
            results = self.execute_cpu(graph_fn, [images, gt_boxes, gt_classes, gt_weights])
        expected_shapes = {'rpn_box_predictor_features': (2, image_size, image_size, 512), 'rpn_features_to_crop': (2, image_size, image_size, 3), 'refined_box_encodings': (2 * max_num_proposals, 2, 4), 'class_predictions_with_background': (2 * max_num_proposals, 2 + 1), 'proposal_boxes': (2, max_num_proposals, 4), 'rpn_box_encodings': (2, image_size * image_size * 9, 4), 'proposal_boxes_normalized': (2, max_num_proposals, 4), 'box_classifier_features': self._get_box_classifier_features_shape(image_size, batch_size, max_num_proposals, initial_crop_size, maxpool_stride, 3), 'rpn_objectness_predictions_with_background': (2, image_size * image_size * 9, 2), 'final_anchors': (2, max_num_proposals, 4)}
        self.assertAllEqual(results[0].shape, expected_shapes['refined_box_encodings'])
        self.assertAllEqual(results[1].shape, expected_shapes['class_predictions_with_background'])
        self.assertAllEqual(results[2].shape, expected_shapes['proposal_boxes'])
        self.assertAllEqual(results[3].shape, expected_shapes['proposal_boxes_normalized'])
        anchors_shape = results[4].shape
        self.assertAllEqual(results[5].shape, [batch_size, anchors_shape[0], 4])
        self.assertAllEqual(results[6].shape, [batch_size, anchors_shape[0], 2])
        self.assertAllEqual(results[7].shape, expected_shapes['rpn_features_to_crop'])
        self.assertAllEqual(results[8].shape, expected_shapes['rpn_box_predictor_features'])
        self.assertAllEqual(results[10].shape, expected_shapes['final_anchors'])

    @parameterized.parameters({'use_static_shapes': False, 'pad_to_max_dimension': None, 'use_keras': True}, {'use_static_shapes': True, 'pad_to_max_dimension': None, 'use_keras': True}, {'use_static_shapes': False, 'pad_to_max_dimension': 56, 'use_keras': True}, {'use_static_shapes': True, 'pad_to_max_dimension': 56, 'use_keras': True}, {'use_static_shapes': False, 'pad_to_max_dimension': None, 'use_keras': False}, {'use_static_shapes': True, 'pad_to_max_dimension': None, 'use_keras': False}, {'use_static_shapes': False, 'pad_to_max_dimension': 56, 'use_keras': False}, {'use_static_shapes': True, 'pad_to_max_dimension': 56, 'use_keras': False})
    def test_postprocess_first_stage_only_inference_mode(self, use_static_shapes=False, pad_to_max_dimension=None, use_keras=False):
        if False:
            return 10
        batch_size = 2
        first_stage_max_proposals = 4 if use_static_shapes else 8

        def graph_fn(images, rpn_box_encodings, rpn_objectness_predictions_with_background, rpn_features_to_crop, anchors):
            if False:
                print('Hello World!')
            'Function to construct tf graph for the test.'
            model = self._build_model(is_training=False, use_keras=use_keras, number_of_stages=1, second_stage_batch_size=6, use_matmul_crop_and_resize=use_static_shapes, clip_anchors_to_image=use_static_shapes, use_static_shapes=use_static_shapes, use_matmul_gather_in_matcher=use_static_shapes, first_stage_max_proposals=first_stage_max_proposals, pad_to_max_dimension=pad_to_max_dimension)
            (_, true_image_shapes) = model.preprocess(images)
            proposals = model.postprocess({'rpn_box_encodings': rpn_box_encodings, 'rpn_objectness_predictions_with_background': rpn_objectness_predictions_with_background, 'rpn_features_to_crop': rpn_features_to_crop, 'anchors': anchors}, true_image_shapes)
            return (proposals['num_detections'], proposals['detection_boxes'], proposals['detection_scores'], proposals['raw_detection_boxes'], proposals['raw_detection_scores'])
        anchors = np.array([[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16], [16, 16, 32, 32]], dtype=np.float32)
        rpn_box_encodings = np.zeros((batch_size, anchors.shape[0], BOX_CODE_SIZE), dtype=np.float32)
        rpn_objectness_predictions_with_background = np.array([[[-10, 13], [10, -10], [10, -11], [-10, 12]], [[10, -10], [-10, 13], [-10, 12], [10, -11]]], dtype=np.float32)
        rpn_features_to_crop = np.ones((batch_size, 8, 8, 10), dtype=np.float32)
        image_shape = (batch_size, 32, 32, 3)
        images = np.zeros(image_shape, dtype=np.float32)
        if use_static_shapes:
            results = self.execute(graph_fn, [images, rpn_box_encodings, rpn_objectness_predictions_with_background, rpn_features_to_crop, anchors])
        else:
            results = self.execute_cpu(graph_fn, [images, rpn_box_encodings, rpn_objectness_predictions_with_background, rpn_features_to_crop, anchors])
        expected_proposal_boxes = [[[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1], [0, 0.5, 0.5, 1], [0.5, 0, 1.0, 0.5]] + 4 * [4 * [0]], [[0, 0.5, 0.5, 1], [0.5, 0, 1.0, 0.5], [0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]] + 4 * [4 * [0]]]
        expected_proposal_scores = [[1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]]
        expected_num_proposals = [4, 4]
        expected_raw_proposal_boxes = [[[0.0, 0.0, 0.5, 0.5], [0.0, 0.5, 0.5, 1.0], [0.5, 0.0, 1.0, 0.5], [0.5, 0.5, 1.0, 1.0]], [[0.0, 0.0, 0.5, 0.5], [0.0, 0.5, 0.5, 1.0], [0.5, 0.0, 1.0, 0.5], [0.5, 0.5, 1.0, 1.0]]]
        expected_raw_scores = [[[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]]
        self.assertAllClose(results[0], expected_num_proposals)
        for (indx, num_proposals) in enumerate(expected_num_proposals):
            self.assertAllClose(results[1][indx][0:num_proposals], expected_proposal_boxes[indx][0:num_proposals])
            self.assertAllClose(results[2][indx][0:num_proposals], expected_proposal_scores[indx][0:num_proposals])
        self.assertAllClose(results[3], expected_raw_proposal_boxes)
        self.assertAllClose(results[4], expected_raw_scores)

    def _test_postprocess_first_stage_only_train_mode(self, use_keras=False, pad_to_max_dimension=None):
        if False:
            return 10
        model = self._build_model(is_training=True, use_keras=use_keras, number_of_stages=1, second_stage_batch_size=2, pad_to_max_dimension=pad_to_max_dimension)
        batch_size = 2
        anchors = tf.constant([[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16], [16, 16, 32, 32]], dtype=tf.float32)
        rpn_box_encodings = tf.zeros([batch_size, anchors.get_shape().as_list()[0], BOX_CODE_SIZE], dtype=tf.float32)
        rpn_objectness_predictions_with_background = tf.constant([[[-10, 13], [-10, 12], [-10, 11], [-10, 10]], [[-10, 13], [-10, 12], [-10, 11], [-10, 10]]], dtype=tf.float32)
        rpn_features_to_crop = tf.ones((batch_size, 8, 8, 10), dtype=tf.float32)
        image_shape = tf.constant([batch_size, 32, 32, 3], dtype=tf.int32)
        groundtruth_boxes_list = [tf.constant([[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]], dtype=tf.float32), tf.constant([[0, 0.5, 0.5, 1], [0.5, 0, 1, 0.5]], dtype=tf.float32)]
        groundtruth_classes_list = [tf.constant([[1, 0], [0, 1]], dtype=tf.float32), tf.constant([[1, 0], [1, 0]], dtype=tf.float32)]
        groundtruth_weights_list = [tf.constant([1, 1], dtype=tf.float32), tf.constant([1, 1], dtype=tf.float32)]
        (_, true_image_shapes) = model.preprocess(tf.zeros(image_shape))
        model.provide_groundtruth(groundtruth_boxes_list, groundtruth_classes_list, groundtruth_weights_list=groundtruth_weights_list)
        proposals = model.postprocess({'rpn_box_encodings': rpn_box_encodings, 'rpn_objectness_predictions_with_background': rpn_objectness_predictions_with_background, 'rpn_features_to_crop': rpn_features_to_crop, 'anchors': anchors}, true_image_shapes)
        expected_proposal_boxes = [[[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]], [[0, 0.5, 0.5, 1], [0.5, 0, 1, 0.5]]]
        expected_proposal_scores = [[1, 1], [1, 1]]
        expected_proposal_multiclass_scores = [[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]]
        expected_raw_proposal_boxes = [[[0.0, 0.0, 0.5, 0.5], [0.0, 0.5, 0.5, 1.0], [0.5, 0.0, 1.0, 0.5], [0.5, 0.5, 1.0, 1.0]], [[0.0, 0.0, 0.5, 0.5], [0.0, 0.5, 0.5, 1.0], [0.5, 0.0, 1.0, 0.5], [0.5, 0.5, 1.0, 1.0]]]
        expected_raw_scores = [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]
        expected_output_keys = set(['detection_boxes', 'detection_scores', 'detection_multiclass_scores', 'num_detections', 'raw_detection_boxes', 'raw_detection_scores'])
        self.assertEqual(set(proposals.keys()), expected_output_keys)
        with self.test_session() as sess:
            proposals_out = sess.run(proposals)
            for image_idx in range(batch_size):
                num_detections = int(proposals_out['num_detections'][image_idx])
                boxes = proposals_out['detection_boxes'][image_idx][:num_detections, :].tolist()
                scores = proposals_out['detection_scores'][image_idx][:num_detections].tolist()
                multiclass_scores = proposals_out['detection_multiclass_scores'][image_idx][:num_detections, :].tolist()
                expected_boxes = expected_proposal_boxes[image_idx]
                expected_scores = expected_proposal_scores[image_idx]
                expected_multiclass_scores = expected_proposal_multiclass_scores[image_idx]
                self.assertTrue(test_utils.first_rows_close_as_set(boxes, expected_boxes))
                self.assertTrue(test_utils.first_rows_close_as_set(scores, expected_scores))
                self.assertTrue(test_utils.first_rows_close_as_set(multiclass_scores, expected_multiclass_scores))
        self.assertAllClose(proposals_out['raw_detection_boxes'], expected_raw_proposal_boxes)
        self.assertAllClose(proposals_out['raw_detection_scores'], expected_raw_scores)

    @parameterized.named_parameters({'testcase_name': 'keras', 'use_keras': True}, {'testcase_name': 'slim', 'use_keras': False})
    def test_postprocess_first_stage_only_train_mode(self, use_keras=False):
        if False:
            return 10
        self._test_postprocess_first_stage_only_train_mode(use_keras=use_keras)

    @parameterized.parameters({'use_keras': True}, {'use_keras': False})
    def test_postprocess_first_stage_only_train_mode_padded_image(self, use_keras=False):
        if False:
            print('Hello World!')
        self._test_postprocess_first_stage_only_train_mode(pad_to_max_dimension=56, use_keras=use_keras)

    @parameterized.parameters({'use_static_shapes': False, 'pad_to_max_dimension': None, 'use_keras': True}, {'use_static_shapes': True, 'pad_to_max_dimension': None, 'use_keras': True}, {'use_static_shapes': False, 'pad_to_max_dimension': 56, 'use_keras': True}, {'use_static_shapes': True, 'pad_to_max_dimension': 56, 'use_keras': True}, {'use_static_shapes': False, 'pad_to_max_dimension': None, 'use_keras': False}, {'use_static_shapes': True, 'pad_to_max_dimension': None, 'use_keras': False}, {'use_static_shapes': False, 'pad_to_max_dimension': 56, 'use_keras': False}, {'use_static_shapes': True, 'pad_to_max_dimension': 56, 'use_keras': False})
    def test_postprocess_second_stage_only_inference_mode(self, use_static_shapes=False, pad_to_max_dimension=None, use_keras=False):
        if False:
            for i in range(10):
                print('nop')
        batch_size = 2
        num_classes = 2
        image_shape = np.array((2, 36, 48, 3), dtype=np.int32)
        first_stage_max_proposals = 8
        total_num_padded_proposals = batch_size * first_stage_max_proposals

        def graph_fn(images, refined_box_encodings, class_predictions_with_background, num_proposals, proposal_boxes):
            if False:
                i = 10
                return i + 15
            'Function to construct tf graph for the test.'
            model = self._build_model(is_training=False, use_keras=use_keras, number_of_stages=2, second_stage_batch_size=6, use_matmul_crop_and_resize=use_static_shapes, clip_anchors_to_image=use_static_shapes, use_static_shapes=use_static_shapes, use_matmul_gather_in_matcher=use_static_shapes, pad_to_max_dimension=pad_to_max_dimension)
            (_, true_image_shapes) = model.preprocess(images)
            detections = model.postprocess({'refined_box_encodings': refined_box_encodings, 'class_predictions_with_background': class_predictions_with_background, 'num_proposals': num_proposals, 'proposal_boxes': proposal_boxes}, true_image_shapes)
            return (detections['num_detections'], detections['detection_boxes'], detections['detection_scores'], detections['detection_classes'], detections['raw_detection_boxes'], detections['raw_detection_scores'], detections['detection_multiclass_scores'], detections['detection_anchor_indices'])
        proposal_boxes = np.array([[[1, 1, 2, 3], [0, 0, 1, 1], [0.5, 0.5, 0.6, 0.6], 4 * [0], 4 * [0], 4 * [0], 4 * [0], 4 * [0]], [[2, 3, 6, 8], [1, 2, 5, 3], 4 * [0], 4 * [0], 4 * [0], 4 * [0], 4 * [0], 4 * [0]]], dtype=np.float32)
        num_proposals = np.array([3, 2], dtype=np.int32)
        refined_box_encodings = np.zeros([total_num_padded_proposals, num_classes, 4], dtype=np.float32)
        class_predictions_with_background = np.ones([total_num_padded_proposals, num_classes + 1], dtype=np.float32)
        images = np.zeros(image_shape, dtype=np.float32)
        if use_static_shapes:
            results = self.execute(graph_fn, [images, refined_box_encodings, class_predictions_with_background, num_proposals, proposal_boxes])
        else:
            results = self.execute_cpu(graph_fn, [images, refined_box_encodings, class_predictions_with_background, num_proposals, proposal_boxes])
        expected_num_detections = [5, 4]
        expected_detection_classes = [[0, 0, 0, 1, 1], [0, 0, 1, 1, 0]]
        expected_detection_scores = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
        expected_multiclass_scores = [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0]]]
        expected_anchor_indices = [[0, 1, 2, 0, 1], [0, 1, 0, 1]]
        h = float(image_shape[1])
        w = float(image_shape[2])
        expected_raw_detection_boxes = np.array([[[1 / h, 1 / w, 2 / h, 3 / w], [0, 0, 1 / h, 1 / w], [0.5 / h, 0.5 / w, 0.6 / h, 0.6 / w], 4 * [0], 4 * [0], 4 * [0], 4 * [0], 4 * [0]], [[2 / h, 3 / w, 6 / h, 8 / w], [1 / h, 2 / w, 5 / h, 3 / w], 4 * [0], 4 * [0], 4 * [0], 4 * [0], 4 * [0], 4 * [0]]], dtype=np.float32)
        self.assertAllClose(results[0], expected_num_detections)
        for (indx, num_proposals) in enumerate(expected_num_detections):
            self.assertAllClose(results[2][indx][0:num_proposals], expected_detection_scores[indx][0:num_proposals])
            self.assertAllClose(results[3][indx][0:num_proposals], expected_detection_classes[indx][0:num_proposals])
            self.assertAllClose(results[6][indx][0:num_proposals], expected_multiclass_scores[indx][0:num_proposals])
            self.assertAllClose(results[7][indx][0:num_proposals], expected_anchor_indices[indx][0:num_proposals])
        self.assertAllClose(results[4], expected_raw_detection_boxes)
        self.assertAllClose(results[5], class_predictions_with_background.reshape([-1, 8, 3]))
        if not use_static_shapes:
            self.assertAllEqual(results[1].shape, [2, 5, 4])

    @parameterized.parameters({'use_keras': True}, {'use_keras': False})
    def test_preprocess_preserves_input_shapes(self, use_keras=False):
        if False:
            i = 10
            return i + 15
        image_shapes = [(3, None, None, 3), (None, 10, 10, 3), (None, None, None, 3)]
        for image_shape in image_shapes:
            model = self._build_model(is_training=False, use_keras=use_keras, number_of_stages=2, second_stage_batch_size=6)
            image_placeholder = tf.placeholder(tf.float32, shape=image_shape)
            (preprocessed_inputs, _) = model.preprocess(image_placeholder)
            self.assertAllEqual(preprocessed_inputs.shape.as_list(), image_shape)

    @parameterized.parameters({'use_keras': True}, {'use_keras': False})
    def test_loss_first_stage_only_mode(self, use_keras=False):
        if False:
            return 10
        model = self._build_model(is_training=True, use_keras=use_keras, number_of_stages=1, second_stage_batch_size=6)
        batch_size = 2
        anchors = tf.constant([[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16], [16, 16, 32, 32]], dtype=tf.float32)
        rpn_box_encodings = tf.zeros([batch_size, anchors.get_shape().as_list()[0], BOX_CODE_SIZE], dtype=tf.float32)
        rpn_objectness_predictions_with_background = tf.constant([[[-10, 13], [10, -10], [10, -11], [-10, 12]], [[10, -10], [-10, 13], [-10, 12], [10, -11]]], dtype=tf.float32)
        image_shape = tf.constant([batch_size, 32, 32, 3], dtype=tf.int32)
        groundtruth_boxes_list = [tf.constant([[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]], dtype=tf.float32), tf.constant([[0, 0.5, 0.5, 1], [0.5, 0, 1, 0.5]], dtype=tf.float32)]
        groundtruth_classes_list = [tf.constant([[1, 0], [0, 1]], dtype=tf.float32), tf.constant([[1, 0], [1, 0]], dtype=tf.float32)]
        prediction_dict = {'rpn_box_encodings': rpn_box_encodings, 'rpn_objectness_predictions_with_background': rpn_objectness_predictions_with_background, 'image_shape': image_shape, 'anchors': anchors}
        (_, true_image_shapes) = model.preprocess(tf.zeros(image_shape))
        model.provide_groundtruth(groundtruth_boxes_list, groundtruth_classes_list)
        loss_dict = model.loss(prediction_dict, true_image_shapes)
        with self.test_session() as sess:
            loss_dict_out = sess.run(loss_dict)
            self.assertAllClose(loss_dict_out['Loss/RPNLoss/localization_loss'], 0)
            self.assertAllClose(loss_dict_out['Loss/RPNLoss/objectness_loss'], 0)
            self.assertNotIn('Loss/BoxClassifierLoss/localization_loss', loss_dict_out)
            self.assertNotIn('Loss/BoxClassifierLoss/classification_loss', loss_dict_out)

    @parameterized.parameters({'use_keras': True}, {'use_keras': False})
    def test_loss_full(self, use_keras=False):
        if False:
            while True:
                i = 10
        model = self._build_model(is_training=True, use_keras=use_keras, number_of_stages=2, second_stage_batch_size=6)
        batch_size = 3
        anchors = tf.constant([[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16], [16, 16, 32, 32]], dtype=tf.float32)
        rpn_box_encodings = tf.zeros([batch_size, anchors.get_shape().as_list()[0], BOX_CODE_SIZE], dtype=tf.float32)
        rpn_objectness_predictions_with_background = tf.constant([[[-10, 13], [10, -10], [10, -11], [-10, 12]], [[10, -10], [-10, 13], [-10, 12], [10, -11]], [[10, -10], [-10, 13], [-10, 12], [10, -11]]], dtype=tf.float32)
        image_shape = tf.constant([batch_size, 32, 32, 3], dtype=tf.int32)
        num_proposals = tf.constant([6, 6, 6], dtype=tf.int32)
        proposal_boxes = tf.constant(3 * [[[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16], [16, 16, 32, 32], [0, 0, 16, 16], [0, 16, 16, 32]]], dtype=tf.float32)
        refined_box_encodings = tf.zeros((batch_size * model.max_num_proposals, model.num_classes, BOX_CODE_SIZE), dtype=tf.float32)
        class_predictions_with_background = tf.constant([[-10, 10, -10], [10, -10, -10], [10, -10, -10], [-10, -10, 10], [-10, 10, -10], [10, -10, -10], [10, -10, -10], [-10, 10, -10], [-10, 10, -10], [10, -10, -10], [10, -10, -10], [-10, 10, -10], [10, -10, -10], [-10, 10, -10], [-10, 10, -10], [10, -10, -10], [10, -10, -10], [-10, 10, -10]], dtype=tf.float32)
        mask_predictions_logits = 20 * tf.ones((batch_size * model.max_num_proposals, model.num_classes, 14, 14), dtype=tf.float32)
        groundtruth_boxes_list = [tf.constant([[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]], dtype=tf.float32), tf.constant([[0, 0.5, 0.5, 1], [0.5, 0, 1, 0.5]], dtype=tf.float32), tf.constant([[0, 0.5, 0.5, 1], [0.5, 0, 1, 1]], dtype=tf.float32)]
        groundtruth_classes_list = [tf.constant([[1, 0], [0, 1]], dtype=tf.float32), tf.constant([[1, 0], [1, 0]], dtype=tf.float32), tf.constant([[1, 0], [0, 1]], dtype=tf.float32)]
        groundtruth_masks_list = [tf.convert_to_tensor(np.ones((2, 32, 32)), dtype=tf.float32), tf.convert_to_tensor(np.ones((2, 32, 32)), dtype=tf.float32), tf.convert_to_tensor(np.ones((2, 32, 32)), dtype=tf.float32)]
        groundtruth_weights_list = [tf.constant([1, 1], dtype=tf.float32), tf.constant([1, 1], dtype=tf.float32), tf.constant([1, 0], dtype=tf.float32)]
        prediction_dict = {'rpn_box_encodings': rpn_box_encodings, 'rpn_objectness_predictions_with_background': rpn_objectness_predictions_with_background, 'image_shape': image_shape, 'anchors': anchors, 'refined_box_encodings': refined_box_encodings, 'class_predictions_with_background': class_predictions_with_background, 'proposal_boxes': proposal_boxes, 'num_proposals': num_proposals, 'mask_predictions': mask_predictions_logits}
        (_, true_image_shapes) = model.preprocess(tf.zeros(image_shape))
        model.provide_groundtruth(groundtruth_boxes_list, groundtruth_classes_list, groundtruth_masks_list, groundtruth_weights_list=groundtruth_weights_list)
        loss_dict = model.loss(prediction_dict, true_image_shapes)
        with self.test_session() as sess:
            loss_dict_out = sess.run(loss_dict)
            self.assertAllClose(loss_dict_out['Loss/RPNLoss/localization_loss'], 0)
            self.assertAllClose(loss_dict_out['Loss/RPNLoss/objectness_loss'], 0)
            self.assertAllClose(loss_dict_out['Loss/BoxClassifierLoss/localization_loss'], 0)
            self.assertAllClose(loss_dict_out['Loss/BoxClassifierLoss/classification_loss'], 0)
            self.assertAllClose(loss_dict_out['Loss/BoxClassifierLoss/mask_loss'], 0)

    @parameterized.parameters({'use_keras': True}, {'use_keras': False})
    def test_loss_full_zero_padded_proposals(self, use_keras=False):
        if False:
            return 10
        model = self._build_model(is_training=True, use_keras=use_keras, number_of_stages=2, second_stage_batch_size=6)
        batch_size = 1
        anchors = tf.constant([[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16], [16, 16, 32, 32]], dtype=tf.float32)
        rpn_box_encodings = tf.zeros([batch_size, anchors.get_shape().as_list()[0], BOX_CODE_SIZE], dtype=tf.float32)
        rpn_objectness_predictions_with_background = tf.constant([[[-10, 13], [10, -10], [10, -11], [10, -12]]], dtype=tf.float32)
        image_shape = tf.constant([batch_size, 32, 32, 3], dtype=tf.int32)
        num_proposals = tf.constant([3], dtype=tf.int32)
        proposal_boxes = tf.constant([[[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=tf.float32)
        refined_box_encodings = tf.zeros((batch_size * model.max_num_proposals, model.num_classes, BOX_CODE_SIZE), dtype=tf.float32)
        class_predictions_with_background = tf.constant([[-10, 10, -10], [10, -10, -10], [10, -10, -10], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=tf.float32)
        mask_predictions_logits = 20 * tf.ones((batch_size * model.max_num_proposals, model.num_classes, 14, 14), dtype=tf.float32)
        groundtruth_boxes_list = [tf.constant([[0, 0, 0.5, 0.5]], dtype=tf.float32)]
        groundtruth_classes_list = [tf.constant([[1, 0]], dtype=tf.float32)]
        groundtruth_masks_list = [tf.convert_to_tensor(np.ones((1, 32, 32)), dtype=tf.float32)]
        prediction_dict = {'rpn_box_encodings': rpn_box_encodings, 'rpn_objectness_predictions_with_background': rpn_objectness_predictions_with_background, 'image_shape': image_shape, 'anchors': anchors, 'refined_box_encodings': refined_box_encodings, 'class_predictions_with_background': class_predictions_with_background, 'proposal_boxes': proposal_boxes, 'num_proposals': num_proposals, 'mask_predictions': mask_predictions_logits}
        (_, true_image_shapes) = model.preprocess(tf.zeros(image_shape))
        model.provide_groundtruth(groundtruth_boxes_list, groundtruth_classes_list, groundtruth_masks_list)
        loss_dict = model.loss(prediction_dict, true_image_shapes)
        with self.test_session() as sess:
            loss_dict_out = sess.run(loss_dict)
            self.assertAllClose(loss_dict_out['Loss/RPNLoss/localization_loss'], 0)
            self.assertAllClose(loss_dict_out['Loss/RPNLoss/objectness_loss'], 0)
            self.assertAllClose(loss_dict_out['Loss/BoxClassifierLoss/localization_loss'], 0)
            self.assertAllClose(loss_dict_out['Loss/BoxClassifierLoss/classification_loss'], 0)
            self.assertAllClose(loss_dict_out['Loss/BoxClassifierLoss/mask_loss'], 0)

    @parameterized.parameters({'use_keras': True}, {'use_keras': False})
    def test_loss_full_multiple_label_groundtruth(self, use_keras=False):
        if False:
            i = 10
            return i + 15
        model = self._build_model(is_training=True, use_keras=use_keras, number_of_stages=2, second_stage_batch_size=6, softmax_second_stage_classification_loss=False)
        batch_size = 1
        anchors = tf.constant([[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16], [16, 16, 32, 32]], dtype=tf.float32)
        rpn_box_encodings = tf.zeros([batch_size, anchors.get_shape().as_list()[0], BOX_CODE_SIZE], dtype=tf.float32)
        rpn_objectness_predictions_with_background = tf.constant([[[-10, 13], [10, -10], [10, -11], [10, -12]]], dtype=tf.float32)
        image_shape = tf.constant([batch_size, 32, 32, 3], dtype=tf.int32)
        num_proposals = tf.constant([3], dtype=tf.int32)
        proposal_boxes = tf.constant([[[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=tf.float32)
        refined_box_encodings = tf.constant([[[0, 0, 0, 0], [1, 1, -1, -1]], [[1, 1, -1, -1], [1, 1, 1, 1]], [[1, 1, -1, -1], [1, 1, 1, 1]], [[1, 1, -1, -1], [1, 1, 1, 1]], [[1, 1, -1, -1], [1, 1, 1, 1]], [[1, 1, -1, -1], [1, 1, 1, 1]]], dtype=tf.float32)
        class_predictions_with_background = tf.constant([[-100, 100, 100], [100, -100, -100], [100, -100, -100], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=tf.float32)
        mask_predictions_logits = 20 * tf.ones((batch_size * model.max_num_proposals, model.num_classes, 14, 14), dtype=tf.float32)
        groundtruth_boxes_list = [tf.constant([[0, 0, 0.5, 0.5]], dtype=tf.float32)]
        groundtruth_classes_list = [tf.constant([[1, 1]], dtype=tf.float32)]
        groundtruth_masks_list = [tf.convert_to_tensor(np.ones((1, 32, 32)), dtype=tf.float32)]
        prediction_dict = {'rpn_box_encodings': rpn_box_encodings, 'rpn_objectness_predictions_with_background': rpn_objectness_predictions_with_background, 'image_shape': image_shape, 'anchors': anchors, 'refined_box_encodings': refined_box_encodings, 'class_predictions_with_background': class_predictions_with_background, 'proposal_boxes': proposal_boxes, 'num_proposals': num_proposals, 'mask_predictions': mask_predictions_logits}
        (_, true_image_shapes) = model.preprocess(tf.zeros(image_shape))
        model.provide_groundtruth(groundtruth_boxes_list, groundtruth_classes_list, groundtruth_masks_list)
        loss_dict = model.loss(prediction_dict, true_image_shapes)
        with self.test_session() as sess:
            loss_dict_out = sess.run(loss_dict)
            self.assertAllClose(loss_dict_out['Loss/RPNLoss/localization_loss'], 0)
            self.assertAllClose(loss_dict_out['Loss/RPNLoss/objectness_loss'], 0)
            self.assertAllClose(loss_dict_out['Loss/BoxClassifierLoss/localization_loss'], 0)
            self.assertAllClose(loss_dict_out['Loss/BoxClassifierLoss/classification_loss'], 0)
            self.assertAllClose(loss_dict_out['Loss/BoxClassifierLoss/mask_loss'], 0)

    @parameterized.parameters({'use_static_shapes': False, 'shared_boxes': False, 'use_keras': True}, {'use_static_shapes': False, 'shared_boxes': True, 'use_keras': True}, {'use_static_shapes': True, 'shared_boxes': False, 'use_keras': True}, {'use_static_shapes': True, 'shared_boxes': True, 'use_keras': True}, {'use_static_shapes': False, 'shared_boxes': False, 'use_keras': False}, {'use_static_shapes': False, 'shared_boxes': True, 'use_keras': False}, {'use_static_shapes': True, 'shared_boxes': False, 'use_keras': False}, {'use_static_shapes': True, 'shared_boxes': True, 'use_keras': False})
    def test_loss_full_zero_padded_proposals_nonzero_loss_with_two_images(self, use_static_shapes=False, shared_boxes=False, use_keras=False):
        if False:
            for i in range(10):
                print('nop')
        batch_size = 2
        first_stage_max_proposals = 8
        second_stage_batch_size = 6
        num_classes = 2

        def graph_fn(anchors, rpn_box_encodings, rpn_objectness_predictions_with_background, images, num_proposals, proposal_boxes, refined_box_encodings, class_predictions_with_background, groundtruth_boxes, groundtruth_classes):
            if False:
                print('Hello World!')
            'Function to construct tf graph for the test.'
            model = self._build_model(is_training=True, use_keras=use_keras, number_of_stages=2, second_stage_batch_size=second_stage_batch_size, first_stage_max_proposals=first_stage_max_proposals, num_classes=num_classes, use_matmul_crop_and_resize=use_static_shapes, clip_anchors_to_image=use_static_shapes, use_static_shapes=use_static_shapes)
            prediction_dict = {'rpn_box_encodings': rpn_box_encodings, 'rpn_objectness_predictions_with_background': rpn_objectness_predictions_with_background, 'image_shape': tf.shape(images), 'anchors': anchors, 'refined_box_encodings': refined_box_encodings, 'class_predictions_with_background': class_predictions_with_background, 'proposal_boxes': proposal_boxes, 'num_proposals': num_proposals}
            (_, true_image_shapes) = model.preprocess(images)
            model.provide_groundtruth(tf.unstack(groundtruth_boxes), tf.unstack(groundtruth_classes))
            loss_dict = model.loss(prediction_dict, true_image_shapes)
            return (loss_dict['Loss/RPNLoss/localization_loss'], loss_dict['Loss/RPNLoss/objectness_loss'], loss_dict['Loss/BoxClassifierLoss/localization_loss'], loss_dict['Loss/BoxClassifierLoss/classification_loss'])
        anchors = np.array([[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16], [16, 16, 32, 32]], dtype=np.float32)
        rpn_box_encodings = np.zeros([batch_size, anchors.shape[1], BOX_CODE_SIZE], dtype=np.float32)
        rpn_objectness_predictions_with_background = np.array([[[-10, 13], [10, -10], [10, -11], [10, -12]], [[-10, 13], [10, -10], [10, -11], [10, -12]]], dtype=np.float32)
        images = np.zeros([batch_size, 32, 32, 3], dtype=np.float32)
        num_proposals = np.array([3, 2], dtype=np.int32)
        proposal_boxes = np.array([[[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 16, 16], [0, 16, 16, 32], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=np.float32)
        refined_box_encodings = np.zeros((batch_size * second_stage_batch_size, 1 if shared_boxes else num_classes, BOX_CODE_SIZE), dtype=np.float32)
        class_predictions_with_background = np.array([[-10, 10, -10], [10, -10, -10], [10, -10, -10], [0, 0, 0], [0, 0, 0], [0, 0, 0], [-10, -10, 10], [10, -10, -10], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
        groundtruth_boxes = np.stack([np.array([[0.05, 0.05, 0.45, 0.45]], dtype=np.float32), np.array([[0.0, 0.0, 0.5, 0.5]], dtype=np.float32)])
        groundtruth_classes = np.stack([np.array([[1, 0]], dtype=np.float32), np.array([[0, 1]], dtype=np.float32)])
        execute_fn = self.execute_cpu
        if use_static_shapes:
            execute_fn = self.execute
        results = execute_fn(graph_fn, [anchors, rpn_box_encodings, rpn_objectness_predictions_with_background, images, num_proposals, proposal_boxes, refined_box_encodings, class_predictions_with_background, groundtruth_boxes, groundtruth_classes])
        exp_loc_loss = (-5 * np.log(0.8) - 0.5) / 3.0
        self.assertAllClose(results[0], exp_loc_loss, rtol=0.0001, atol=0.0001)
        self.assertAllClose(results[1], 0.0)
        self.assertAllClose(results[2], exp_loc_loss, rtol=0.0001, atol=0.0001)
        self.assertAllClose(results[3], 0.0)

    @parameterized.parameters({'use_keras': True}, {'use_keras': False})
    def test_loss_with_hard_mining(self, use_keras=False):
        if False:
            i = 10
            return i + 15
        model = self._build_model(is_training=True, use_keras=use_keras, number_of_stages=2, second_stage_batch_size=None, first_stage_max_proposals=6, hard_mining=True)
        batch_size = 1
        anchors = tf.constant([[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16], [16, 16, 32, 32]], dtype=tf.float32)
        rpn_box_encodings = tf.zeros([batch_size, anchors.get_shape().as_list()[0], BOX_CODE_SIZE], dtype=tf.float32)
        rpn_objectness_predictions_with_background = tf.constant([[[-10, 13], [-10, 12], [10, -11], [10, -12]]], dtype=tf.float32)
        image_shape = tf.constant([batch_size, 32, 32, 3], dtype=tf.int32)
        num_proposals = tf.constant([3], dtype=tf.int32)
        proposal_boxes = tf.constant([[[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=tf.float32)
        refined_box_encodings = tf.zeros((batch_size * model.max_num_proposals, model.num_classes, BOX_CODE_SIZE), dtype=tf.float32)
        class_predictions_with_background = tf.constant([[-10, 10, -10], [-10, -10, 10], [10, -10, -10], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=tf.float32)
        groundtruth_boxes_list = [tf.constant([[0.05, 0.05, 0.45, 0.45], [0.02, 0.52, 0.48, 0.98]], dtype=tf.float32)]
        groundtruth_classes_list = [tf.constant([[1, 0], [0, 1]], dtype=tf.float32)]
        exp_loc_loss = 2 * (-5 * np.log(0.8) - 0.5) / 3.0
        prediction_dict = {'rpn_box_encodings': rpn_box_encodings, 'rpn_objectness_predictions_with_background': rpn_objectness_predictions_with_background, 'image_shape': image_shape, 'anchors': anchors, 'refined_box_encodings': refined_box_encodings, 'class_predictions_with_background': class_predictions_with_background, 'proposal_boxes': proposal_boxes, 'num_proposals': num_proposals}
        (_, true_image_shapes) = model.preprocess(tf.zeros(image_shape))
        model.provide_groundtruth(groundtruth_boxes_list, groundtruth_classes_list)
        loss_dict = model.loss(prediction_dict, true_image_shapes)
        with self.test_session() as sess:
            loss_dict_out = sess.run(loss_dict)
            self.assertAllClose(loss_dict_out['Loss/BoxClassifierLoss/localization_loss'], exp_loc_loss)
            self.assertAllClose(loss_dict_out['Loss/BoxClassifierLoss/classification_loss'], 0)

    @parameterized.parameters({'use_keras': True}, {'use_keras': False})
    def test_loss_with_hard_mining_and_losses_mask(self, use_keras=False):
        if False:
            return 10
        model = self._build_model(is_training=True, use_keras=use_keras, number_of_stages=2, second_stage_batch_size=None, first_stage_max_proposals=6, hard_mining=True)
        batch_size = 2
        number_of_proposals = 3
        anchors = tf.constant([[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16], [16, 16, 32, 32]], dtype=tf.float32)
        rpn_box_encodings = tf.zeros([batch_size, anchors.get_shape().as_list()[0], BOX_CODE_SIZE], dtype=tf.float32)
        rpn_objectness_predictions_with_background = tf.constant([[[-10, 13], [-10, 12], [10, -11], [10, -12]], [[-10, 13], [-10, 12], [10, -11], [10, -12]]], dtype=tf.float32)
        image_shape = tf.constant([batch_size, 32, 32, 3], dtype=tf.int32)
        num_proposals = tf.constant([number_of_proposals, number_of_proposals], dtype=tf.int32)
        proposal_boxes = tf.constant([[[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=tf.float32)
        refined_box_encodings = tf.zeros((batch_size * model.max_num_proposals, model.num_classes, BOX_CODE_SIZE), dtype=tf.float32)
        class_predictions_with_background = tf.constant([[-10, 10, -10], [-10, -10, 10], [10, -10, -10], [0, 0, 0], [0, 0, 0], [0, 0, 0], [-10, 10, -10], [-10, -10, 10], [10, -10, -10], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=tf.float32)
        groundtruth_boxes_list = [tf.constant([[0.05, 0.05, 0.45, 0.45], [0.02, 0.52, 0.48, 0.98]], dtype=tf.float32), tf.constant([[0.05, 0.05, 0.45, 0.45], [0.02, 0.52, 0.48, 0.98]], dtype=tf.float32)]
        groundtruth_classes_list = [tf.constant([[1, 0], [0, 1]], dtype=tf.float32), tf.constant([[1, 0], [0, 1]], dtype=tf.float32)]
        is_annotated_list = [tf.constant(True, dtype=tf.bool), tf.constant(False, dtype=tf.bool)]
        exp_loc_loss = 2 * (-5 * np.log(0.8) - 0.5) / (number_of_proposals * batch_size)
        prediction_dict = {'rpn_box_encodings': rpn_box_encodings, 'rpn_objectness_predictions_with_background': rpn_objectness_predictions_with_background, 'image_shape': image_shape, 'anchors': anchors, 'refined_box_encodings': refined_box_encodings, 'class_predictions_with_background': class_predictions_with_background, 'proposal_boxes': proposal_boxes, 'num_proposals': num_proposals}
        (_, true_image_shapes) = model.preprocess(tf.zeros(image_shape))
        model.provide_groundtruth(groundtruth_boxes_list, groundtruth_classes_list, is_annotated_list=is_annotated_list)
        loss_dict = model.loss(prediction_dict, true_image_shapes)
        with self.test_session() as sess:
            loss_dict_out = sess.run(loss_dict)
            self.assertAllClose(loss_dict_out['Loss/BoxClassifierLoss/localization_loss'], exp_loc_loss)
            self.assertAllClose(loss_dict_out['Loss/BoxClassifierLoss/classification_loss'], 0)

    @parameterized.parameters({'use_keras': True}, {'use_keras': False})
    def test_restore_map_for_classification_ckpt(self, use_keras=False):
        if False:
            for i in range(10):
                print('nop')
        test_graph_classification = tf.Graph()
        with test_graph_classification.as_default():
            image = tf.placeholder(dtype=tf.float32, shape=[1, 20, 20, 3])
            with tf.variable_scope('mock_model'):
                net = slim.conv2d(image, num_outputs=3, kernel_size=1, scope='layer1')
                slim.conv2d(net, num_outputs=3, kernel_size=1, scope='layer2')
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
            save_path = self.get_temp_dir()
            with self.test_session(graph=test_graph_classification) as sess:
                sess.run(init_op)
                saved_model_path = saver.save(sess, save_path)
        test_graph_detection = tf.Graph()
        with test_graph_detection.as_default():
            model = self._build_model(is_training=False, use_keras=use_keras, number_of_stages=2, second_stage_batch_size=6)
            inputs_shape = (2, 20, 20, 3)
            inputs = tf.cast(tf.random_uniform(inputs_shape, minval=0, maxval=255, dtype=tf.int32), dtype=tf.float32)
            (preprocessed_inputs, true_image_shapes) = model.preprocess(inputs)
            prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
            model.postprocess(prediction_dict, true_image_shapes)
            var_map = model.restore_map(fine_tune_checkpoint_type='classification')
            self.assertIsInstance(var_map, dict)
            saver = tf.train.Saver(var_map)
            with self.test_session(graph=test_graph_classification) as sess:
                saver.restore(sess, saved_model_path)
                for var in sess.run(tf.report_uninitialized_variables()):
                    self.assertNotIn(model.first_stage_feature_extractor_scope, var)
                    self.assertNotIn(model.second_stage_feature_extractor_scope, var)

    @parameterized.parameters({'use_keras': True}, {'use_keras': False})
    def test_restore_map_for_detection_ckpt(self, use_keras=False):
        if False:
            i = 10
            return i + 15
        test_graph_detection1 = tf.Graph()
        with test_graph_detection1.as_default():
            model = self._build_model(is_training=False, use_keras=use_keras, number_of_stages=2, second_stage_batch_size=6)
            inputs_shape = (2, 20, 20, 3)
            inputs = tf.cast(tf.random_uniform(inputs_shape, minval=0, maxval=255, dtype=tf.int32), dtype=tf.float32)
            (preprocessed_inputs, true_image_shapes) = model.preprocess(inputs)
            prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
            model.postprocess(prediction_dict, true_image_shapes)
            another_variable = tf.Variable([17.0], name='another_variable')
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
            save_path = self.get_temp_dir()
            with self.test_session(graph=test_graph_detection1) as sess:
                sess.run(init_op)
                saved_model_path = saver.save(sess, save_path)
        test_graph_detection2 = tf.Graph()
        with test_graph_detection2.as_default():
            model2 = self._build_model(is_training=False, use_keras=use_keras, number_of_stages=2, second_stage_batch_size=6, num_classes=42)
            inputs_shape2 = (2, 20, 20, 3)
            inputs2 = tf.cast(tf.random_uniform(inputs_shape2, minval=0, maxval=255, dtype=tf.int32), dtype=tf.float32)
            (preprocessed_inputs2, true_image_shapes) = model2.preprocess(inputs2)
            prediction_dict2 = model2.predict(preprocessed_inputs2, true_image_shapes)
            model2.postprocess(prediction_dict2, true_image_shapes)
            another_variable = tf.Variable([17.0], name='another_variable')
            var_map = model2.restore_map(fine_tune_checkpoint_type='detection')
            self.assertIsInstance(var_map, dict)
            saver = tf.train.Saver(var_map)
            with self.test_session(graph=test_graph_detection2) as sess:
                saver.restore(sess, saved_model_path)
                uninitialized_vars_list = sess.run(tf.report_uninitialized_variables())
                self.assertIn('another_variable', uninitialized_vars_list)
                for var in uninitialized_vars_list:
                    self.assertNotIn(model2.first_stage_feature_extractor_scope, var)
                    self.assertNotIn(model2.second_stage_feature_extractor_scope, var)

    @parameterized.parameters({'use_keras': True}, {'use_keras': False})
    def test_load_all_det_checkpoint_vars(self, use_keras=False):
        if False:
            i = 10
            return i + 15
        test_graph_detection = tf.Graph()
        with test_graph_detection.as_default():
            model = self._build_model(is_training=False, use_keras=use_keras, number_of_stages=2, second_stage_batch_size=6, num_classes=42)
            inputs_shape = (2, 20, 20, 3)
            inputs = tf.cast(tf.random_uniform(inputs_shape, minval=0, maxval=255, dtype=tf.int32), dtype=tf.float32)
            (preprocessed_inputs, true_image_shapes) = model.preprocess(inputs)
            prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
            model.postprocess(prediction_dict, true_image_shapes)
            another_variable = tf.Variable([17.0], name='another_variable')
            var_map = model.restore_map(fine_tune_checkpoint_type='detection', load_all_detection_checkpoint_vars=True)
            self.assertIsInstance(var_map, dict)
            self.assertIn('another_variable', var_map)
if __name__ == '__main__':
    tf.test.main()