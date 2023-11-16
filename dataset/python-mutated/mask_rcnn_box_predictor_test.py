"""Tests for object_detection.predictors.mask_rcnn_box_predictor."""
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import box_predictor_builder
from object_detection.builders import hyperparams_builder
from object_detection.predictors import mask_rcnn_box_predictor as box_predictor
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case

class MaskRCNNBoxPredictorTest(test_case.TestCase):

    def _build_arg_scope_with_hyperparams(self, op_type=hyperparams_pb2.Hyperparams.FC):
        if False:
            for i in range(10):
                print('nop')
        hyperparams = hyperparams_pb2.Hyperparams()
        hyperparams_text_proto = '\n      activation: NONE\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        text_format.Merge(hyperparams_text_proto, hyperparams)
        hyperparams.op = op_type
        return hyperparams_builder.build(hyperparams, is_training=True)

    def test_get_boxes_with_five_classes(self):
        if False:
            i = 10
            return i + 15

        def graph_fn(image_features):
            if False:
                i = 10
                return i + 15
            mask_box_predictor = box_predictor_builder.build_mask_rcnn_box_predictor(is_training=False, num_classes=5, fc_hyperparams_fn=self._build_arg_scope_with_hyperparams(), use_dropout=False, dropout_keep_prob=0.5, box_code_size=4)
            box_predictions = mask_box_predictor.predict([image_features], num_predictions_per_location=[1], scope='BoxPredictor', prediction_stage=2)
            return (box_predictions[box_predictor.BOX_ENCODINGS], box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND])
        image_features = np.random.rand(2, 7, 7, 3).astype(np.float32)
        (box_encodings, class_predictions_with_background) = self.execute(graph_fn, [image_features])
        self.assertAllEqual(box_encodings.shape, [2, 1, 5, 4])
        self.assertAllEqual(class_predictions_with_background.shape, [2, 1, 6])

    def test_get_boxes_with_five_classes_share_box_across_classes(self):
        if False:
            i = 10
            return i + 15

        def graph_fn(image_features):
            if False:
                print('Hello World!')
            mask_box_predictor = box_predictor_builder.build_mask_rcnn_box_predictor(is_training=False, num_classes=5, fc_hyperparams_fn=self._build_arg_scope_with_hyperparams(), use_dropout=False, dropout_keep_prob=0.5, box_code_size=4, share_box_across_classes=True)
            box_predictions = mask_box_predictor.predict([image_features], num_predictions_per_location=[1], scope='BoxPredictor', prediction_stage=2)
            return (box_predictions[box_predictor.BOX_ENCODINGS], box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND])
        image_features = np.random.rand(2, 7, 7, 3).astype(np.float32)
        (box_encodings, class_predictions_with_background) = self.execute(graph_fn, [image_features])
        self.assertAllEqual(box_encodings.shape, [2, 1, 1, 4])
        self.assertAllEqual(class_predictions_with_background.shape, [2, 1, 6])

    def test_value_error_on_predict_instance_masks_with_no_conv_hyperparms(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            box_predictor_builder.build_mask_rcnn_box_predictor(is_training=False, num_classes=5, fc_hyperparams_fn=self._build_arg_scope_with_hyperparams(), use_dropout=False, dropout_keep_prob=0.5, box_code_size=4, predict_instance_masks=True)

    def test_get_instance_masks(self):
        if False:
            i = 10
            return i + 15

        def graph_fn(image_features):
            if False:
                return 10
            mask_box_predictor = box_predictor_builder.build_mask_rcnn_box_predictor(is_training=False, num_classes=5, fc_hyperparams_fn=self._build_arg_scope_with_hyperparams(), use_dropout=False, dropout_keep_prob=0.5, box_code_size=4, conv_hyperparams_fn=self._build_arg_scope_with_hyperparams(op_type=hyperparams_pb2.Hyperparams.CONV), predict_instance_masks=True)
            box_predictions = mask_box_predictor.predict([image_features], num_predictions_per_location=[1], scope='BoxPredictor', prediction_stage=3)
            return (box_predictions[box_predictor.MASK_PREDICTIONS],)
        image_features = np.random.rand(2, 7, 7, 3).astype(np.float32)
        mask_predictions = self.execute(graph_fn, [image_features])
        self.assertAllEqual(mask_predictions.shape, [2, 1, 5, 14, 14])

    def test_do_not_return_instance_masks_without_request(self):
        if False:
            for i in range(10):
                print('nop')
        image_features = tf.random_uniform([2, 7, 7, 3], dtype=tf.float32)
        mask_box_predictor = box_predictor_builder.build_mask_rcnn_box_predictor(is_training=False, num_classes=5, fc_hyperparams_fn=self._build_arg_scope_with_hyperparams(), use_dropout=False, dropout_keep_prob=0.5, box_code_size=4)
        box_predictions = mask_box_predictor.predict([image_features], num_predictions_per_location=[1], scope='BoxPredictor', prediction_stage=2)
        self.assertEqual(len(box_predictions), 2)
        self.assertTrue(box_predictor.BOX_ENCODINGS in box_predictions)
        self.assertTrue(box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND in box_predictions)
if __name__ == '__main__':
    tf.test.main()