"""Tests for object_detection.predictors.heads.class_head."""
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.predictors.heads import class_head
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case

class MaskRCNNClassHeadTest(test_case.TestCase):

    def _build_arg_scope_with_hyperparams(self, op_type=hyperparams_pb2.Hyperparams.FC):
        if False:
            for i in range(10):
                print('nop')
        hyperparams = hyperparams_pb2.Hyperparams()
        hyperparams_text_proto = '\n      activation: NONE\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        text_format.Merge(hyperparams_text_proto, hyperparams)
        hyperparams.op = op_type
        return hyperparams_builder.build(hyperparams, is_training=True)

    def test_prediction_size(self):
        if False:
            print('Hello World!')
        class_prediction_head = class_head.MaskRCNNClassHead(is_training=False, num_class_slots=20, fc_hyperparams_fn=self._build_arg_scope_with_hyperparams(), use_dropout=True, dropout_keep_prob=0.5)
        roi_pooled_features = tf.random_uniform([64, 7, 7, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
        prediction = class_prediction_head.predict(features=roi_pooled_features, num_predictions_per_location=1)
        self.assertAllEqual([64, 1, 20], prediction.get_shape().as_list())

    def test_scope_name(self):
        if False:
            for i in range(10):
                print('nop')
        expected_var_names = set(['ClassPredictor/weights', 'ClassPredictor/biases'])
        g = tf.Graph()
        with g.as_default():
            class_prediction_head = class_head.MaskRCNNClassHead(is_training=True, num_class_slots=20, fc_hyperparams_fn=self._build_arg_scope_with_hyperparams(), use_dropout=True, dropout_keep_prob=0.5)
            image_feature = tf.random_uniform([64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
            class_prediction_head.predict(features=image_feature, num_predictions_per_location=1)
            actual_variable_set = set([var.op.name for var in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
            self.assertSetEqual(expected_var_names, actual_variable_set)

class ConvolutionalClassPredictorTest(test_case.TestCase):

    def _build_arg_scope_with_hyperparams(self, op_type=hyperparams_pb2.Hyperparams.CONV):
        if False:
            print('Hello World!')
        hyperparams = hyperparams_pb2.Hyperparams()
        hyperparams_text_proto = '\n      activation: NONE\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        text_format.Merge(hyperparams_text_proto, hyperparams)
        hyperparams.op = op_type
        return hyperparams_builder.build(hyperparams, is_training=True)

    def test_prediction_size(self):
        if False:
            for i in range(10):
                print('nop')
        class_prediction_head = class_head.ConvolutionalClassHead(is_training=True, num_class_slots=20, use_dropout=True, dropout_keep_prob=0.5, kernel_size=3)
        image_feature = tf.random_uniform([64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
        class_predictions = class_prediction_head.predict(features=image_feature, num_predictions_per_location=1)
        self.assertAllEqual([64, 323, 20], class_predictions.get_shape().as_list())

    def test_scope_name(self):
        if False:
            print('Hello World!')
        expected_var_names = set(['ClassPredictor/weights', 'ClassPredictor/biases'])
        g = tf.Graph()
        with g.as_default():
            class_prediction_head = class_head.ConvolutionalClassHead(is_training=True, num_class_slots=20, use_dropout=True, dropout_keep_prob=0.5, kernel_size=3)
            image_feature = tf.random_uniform([64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
            class_prediction_head.predict(features=image_feature, num_predictions_per_location=1)
            actual_variable_set = set([var.op.name for var in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
            self.assertSetEqual(expected_var_names, actual_variable_set)

class WeightSharedConvolutionalClassPredictorTest(test_case.TestCase):

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
            while True:
                i = 10
        class_prediction_head = class_head.WeightSharedConvolutionalClassHead(num_class_slots=20)
        image_feature = tf.random_uniform([64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
        class_predictions = class_prediction_head.predict(features=image_feature, num_predictions_per_location=1)
        self.assertAllEqual([64, 323, 20], class_predictions.get_shape().as_list())

    def test_scope_name(self):
        if False:
            i = 10
            return i + 15
        expected_var_names = set(['ClassPredictor/weights', 'ClassPredictor/biases'])
        g = tf.Graph()
        with g.as_default():
            class_prediction_head = class_head.WeightSharedConvolutionalClassHead(num_class_slots=20)
            image_feature = tf.random_uniform([64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
            class_prediction_head.predict(features=image_feature, num_predictions_per_location=1)
            actual_variable_set = set([var.op.name for var in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
            self.assertSetEqual(expected_var_names, actual_variable_set)
if __name__ == '__main__':
    tf.test.main()