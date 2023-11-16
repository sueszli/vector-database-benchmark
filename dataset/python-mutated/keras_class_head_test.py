"""Tests for object_detection.predictors.heads.class_head."""
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.predictors.heads import keras_class_head
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case

class ConvolutionalKerasClassPredictorTest(test_case.TestCase):

    def _build_conv_hyperparams(self):
        if False:
            print('Hello World!')
        conv_hyperparams = hyperparams_pb2.Hyperparams()
        conv_hyperparams_text_proto = '\n    activation: NONE\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
        return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

    def test_prediction_size_depthwise_false(self):
        if False:
            for i in range(10):
                print('nop')
        conv_hyperparams = self._build_conv_hyperparams()
        class_prediction_head = keras_class_head.ConvolutionalClassHead(is_training=True, num_class_slots=20, use_dropout=True, dropout_keep_prob=0.5, kernel_size=3, conv_hyperparams=conv_hyperparams, freeze_batchnorm=False, num_predictions_per_location=1, use_depthwise=False)
        image_feature = tf.random_uniform([64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
        class_predictions = class_prediction_head(image_feature)
        self.assertAllEqual([64, 323, 20], class_predictions.get_shape().as_list())

    def test_prediction_size_depthwise_true(self):
        if False:
            return 10
        conv_hyperparams = self._build_conv_hyperparams()
        class_prediction_head = keras_class_head.ConvolutionalClassHead(is_training=True, num_class_slots=20, use_dropout=True, dropout_keep_prob=0.5, kernel_size=3, conv_hyperparams=conv_hyperparams, freeze_batchnorm=False, num_predictions_per_location=1, use_depthwise=True)
        image_feature = tf.random_uniform([64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
        class_predictions = class_prediction_head(image_feature)
        self.assertAllEqual([64, 323, 20], class_predictions.get_shape().as_list())

class MaskRCNNClassHeadTest(test_case.TestCase):

    def _build_fc_hyperparams(self, op_type=hyperparams_pb2.Hyperparams.FC):
        if False:
            for i in range(10):
                print('nop')
        hyperparams = hyperparams_pb2.Hyperparams()
        hyperparams_text_proto = '\n      activation: NONE\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        text_format.Merge(hyperparams_text_proto, hyperparams)
        hyperparams.op = op_type
        return hyperparams_builder.KerasLayerHyperparams(hyperparams)

    def test_prediction_size(self):
        if False:
            print('Hello World!')
        class_prediction_head = keras_class_head.MaskRCNNClassHead(is_training=False, num_class_slots=20, fc_hyperparams=self._build_fc_hyperparams(), freeze_batchnorm=False, use_dropout=True, dropout_keep_prob=0.5)
        roi_pooled_features = tf.random_uniform([64, 7, 7, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
        prediction = class_prediction_head(roi_pooled_features)
        self.assertAllEqual([64, 1, 20], prediction.get_shape().as_list())

class WeightSharedConvolutionalKerasClassPredictorTest(test_case.TestCase):

    def _build_conv_hyperparams(self):
        if False:
            print('Hello World!')
        conv_hyperparams = hyperparams_pb2.Hyperparams()
        conv_hyperparams_text_proto = '\n    activation: NONE\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
        return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

    def test_prediction_size_depthwise_false(self):
        if False:
            while True:
                i = 10
        conv_hyperparams = self._build_conv_hyperparams()
        class_prediction_head = keras_class_head.WeightSharedConvolutionalClassHead(num_class_slots=20, conv_hyperparams=conv_hyperparams, num_predictions_per_location=1, use_depthwise=False)
        image_feature = tf.random_uniform([64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
        class_predictions = class_prediction_head(image_feature)
        self.assertAllEqual([64, 323, 20], class_predictions.get_shape().as_list())

    def test_prediction_size_depthwise_true(self):
        if False:
            i = 10
            return i + 15
        conv_hyperparams = self._build_conv_hyperparams()
        class_prediction_head = keras_class_head.WeightSharedConvolutionalClassHead(num_class_slots=20, conv_hyperparams=conv_hyperparams, num_predictions_per_location=1, use_depthwise=True)
        image_feature = tf.random_uniform([64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
        class_predictions = class_prediction_head(image_feature)
        self.assertAllEqual([64, 323, 20], class_predictions.get_shape().as_list())

    def test_variable_count_depth_wise_true(self):
        if False:
            print('Hello World!')
        g = tf.Graph()
        with g.as_default():
            conv_hyperparams = self._build_conv_hyperparams()
            class_prediction_head = keras_class_head.WeightSharedConvolutionalClassHead(num_class_slots=20, conv_hyperparams=conv_hyperparams, num_predictions_per_location=1, use_depthwise=True)
            image_feature = tf.random_uniform([64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
            _ = class_prediction_head(image_feature)
            variables = g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.assertEqual(len(variables), 3)

    def test_variable_count_depth_wise_False(self):
        if False:
            print('Hello World!')
        g = tf.Graph()
        with g.as_default():
            conv_hyperparams = self._build_conv_hyperparams()
            class_prediction_head = keras_class_head.WeightSharedConvolutionalClassHead(num_class_slots=20, conv_hyperparams=conv_hyperparams, num_predictions_per_location=1, use_depthwise=False)
            image_feature = tf.random_uniform([64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
            _ = class_prediction_head(image_feature)
            variables = g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.assertEqual(len(variables), 2)
if __name__ == '__main__':
    tf.test.main()