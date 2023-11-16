"""Tests for object_detection.predictors.heads.box_head."""
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.predictors.heads import box_head
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case

class MaskRCNNBoxHeadTest(test_case.TestCase):

    def _build_arg_scope_with_hyperparams(self, op_type=hyperparams_pb2.Hyperparams.FC):
        if False:
            print('Hello World!')
        hyperparams = hyperparams_pb2.Hyperparams()
        hyperparams_text_proto = '\n      activation: NONE\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        text_format.Merge(hyperparams_text_proto, hyperparams)
        hyperparams.op = op_type
        return hyperparams_builder.build(hyperparams, is_training=True)

    def test_prediction_size(self):
        if False:
            print('Hello World!')
        box_prediction_head = box_head.MaskRCNNBoxHead(is_training=False, num_classes=20, fc_hyperparams_fn=self._build_arg_scope_with_hyperparams(), use_dropout=True, dropout_keep_prob=0.5, box_code_size=4, share_box_across_classes=False)
        roi_pooled_features = tf.random_uniform([64, 7, 7, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
        prediction = box_prediction_head.predict(features=roi_pooled_features, num_predictions_per_location=1)
        self.assertAllEqual([64, 1, 20, 4], prediction.get_shape().as_list())

class ConvolutionalBoxPredictorTest(test_case.TestCase):

    def _build_arg_scope_with_hyperparams(self, op_type=hyperparams_pb2.Hyperparams.CONV):
        if False:
            while True:
                i = 10
        hyperparams = hyperparams_pb2.Hyperparams()
        hyperparams_text_proto = '\n      activation: NONE\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        text_format.Merge(hyperparams_text_proto, hyperparams)
        hyperparams.op = op_type
        return hyperparams_builder.build(hyperparams, is_training=True)

    def test_prediction_size(self):
        if False:
            i = 10
            return i + 15
        box_prediction_head = box_head.ConvolutionalBoxHead(is_training=True, box_code_size=4, kernel_size=3)
        image_feature = tf.random_uniform([64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
        box_encodings = box_prediction_head.predict(features=image_feature, num_predictions_per_location=1)
        self.assertAllEqual([64, 323, 1, 4], box_encodings.get_shape().as_list())

class WeightSharedConvolutionalBoxPredictorTest(test_case.TestCase):

    def _build_arg_scope_with_hyperparams(self, op_type=hyperparams_pb2.Hyperparams.CONV):
        if False:
            return 10
        hyperparams = hyperparams_pb2.Hyperparams()
        hyperparams_text_proto = '\n      activation: NONE\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        text_format.Merge(hyperparams_text_proto, hyperparams)
        hyperparams.op = op_type
        return hyperparams_builder.build(hyperparams, is_training=True)

    def test_prediction_size(self):
        if False:
            while True:
                i = 10
        box_prediction_head = box_head.WeightSharedConvolutionalBoxHead(box_code_size=4)
        image_feature = tf.random_uniform([64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
        box_encodings = box_prediction_head.predict(features=image_feature, num_predictions_per_location=1)
        self.assertAllEqual([64, 323, 4], box_encodings.get_shape().as_list())
if __name__ == '__main__':
    tf.test.main()