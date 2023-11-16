"""Tests object_detection.core.hyperparams_builder."""
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.core import freezable_batch_norm
from object_detection.protos import hyperparams_pb2
slim = tf.contrib.slim

def _get_scope_key(op):
    if False:
        return 10
    return getattr(op, '_key_op', str(op))

class HyperparamsBuilderTest(tf.test.TestCase):

    def test_default_arg_scope_has_conv2d_op(self):
        if False:
            for i in range(10):
                print('nop')
        conv_hyperparams_text_proto = '\n      regularizer {\n        l1_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        self.assertTrue(_get_scope_key(slim.conv2d) in scope)

    def test_default_arg_scope_has_separable_conv2d_op(self):
        if False:
            return 10
        conv_hyperparams_text_proto = '\n      regularizer {\n        l1_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        self.assertTrue(_get_scope_key(slim.separable_conv2d) in scope)

    def test_default_arg_scope_has_conv2d_transpose_op(self):
        if False:
            return 10
        conv_hyperparams_text_proto = '\n      regularizer {\n        l1_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        self.assertTrue(_get_scope_key(slim.conv2d_transpose) in scope)

    def test_explicit_fc_op_arg_scope_has_fully_connected_op(self):
        if False:
            return 10
        conv_hyperparams_text_proto = '\n      op: FC\n      regularizer {\n        l1_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        self.assertTrue(_get_scope_key(slim.fully_connected) in scope)

    def test_separable_conv2d_and_conv2d_and_transpose_have_same_parameters(self):
        if False:
            print('Hello World!')
        conv_hyperparams_text_proto = '\n      regularizer {\n        l1_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        (kwargs_1, kwargs_2, kwargs_3) = scope.values()
        self.assertDictEqual(kwargs_1, kwargs_2)
        self.assertDictEqual(kwargs_1, kwargs_3)

    def test_return_l1_regularized_weights(self):
        if False:
            while True:
                i = 10
        conv_hyperparams_text_proto = '\n      regularizer {\n        l1_regularizer {\n          weight: 0.5\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        conv_scope_arguments = scope.values()[0]
        regularizer = conv_scope_arguments['weights_regularizer']
        weights = np.array([1.0, -1, 4.0, 2.0])
        with self.test_session() as sess:
            result = sess.run(regularizer(tf.constant(weights)))
        self.assertAllClose(np.abs(weights).sum() * 0.5, result)

    def test_return_l1_regularized_weights_keras(self):
        if False:
            for i in range(10):
                print('nop')
        conv_hyperparams_text_proto = '\n      regularizer {\n        l1_regularizer {\n          weight: 0.5\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        keras_config = hyperparams_builder.KerasLayerHyperparams(conv_hyperparams_proto)
        regularizer = keras_config.params()['kernel_regularizer']
        weights = np.array([1.0, -1, 4.0, 2.0])
        with self.test_session() as sess:
            result = sess.run(regularizer(tf.constant(weights)))
        self.assertAllClose(np.abs(weights).sum() * 0.5, result)

    def test_return_l2_regularizer_weights(self):
        if False:
            i = 10
            return i + 15
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n          weight: 0.42\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
        regularizer = conv_scope_arguments['weights_regularizer']
        weights = np.array([1.0, -1, 4.0, 2.0])
        with self.test_session() as sess:
            result = sess.run(regularizer(tf.constant(weights)))
        self.assertAllClose(np.power(weights, 2).sum() / 2.0 * 0.42, result)

    def test_return_l2_regularizer_weights_keras(self):
        if False:
            return 10
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n          weight: 0.42\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        keras_config = hyperparams_builder.KerasLayerHyperparams(conv_hyperparams_proto)
        regularizer = keras_config.params()['kernel_regularizer']
        weights = np.array([1.0, -1, 4.0, 2.0])
        with self.test_session() as sess:
            result = sess.run(regularizer(tf.constant(weights)))
        self.assertAllClose(np.power(weights, 2).sum() / 2.0 * 0.42, result)

    def test_return_non_default_batch_norm_params_with_train_during_train(self):
        if False:
            i = 10
            return i + 15
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n      batch_norm {\n        decay: 0.7\n        center: false\n        scale: true\n        epsilon: 0.03\n        train: true\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
        self.assertEqual(conv_scope_arguments['normalizer_fn'], slim.batch_norm)
        batch_norm_params = scope[_get_scope_key(slim.batch_norm)]
        self.assertAlmostEqual(batch_norm_params['decay'], 0.7)
        self.assertAlmostEqual(batch_norm_params['epsilon'], 0.03)
        self.assertFalse(batch_norm_params['center'])
        self.assertTrue(batch_norm_params['scale'])
        self.assertTrue(batch_norm_params['is_training'])

    def test_return_non_default_batch_norm_params_keras(self):
        if False:
            print('Hello World!')
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n      batch_norm {\n        decay: 0.7\n        center: false\n        scale: true\n        epsilon: 0.03\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        keras_config = hyperparams_builder.KerasLayerHyperparams(conv_hyperparams_proto)
        self.assertTrue(keras_config.use_batch_norm())
        batch_norm_params = keras_config.batch_norm_params()
        self.assertAlmostEqual(batch_norm_params['momentum'], 0.7)
        self.assertAlmostEqual(batch_norm_params['epsilon'], 0.03)
        self.assertFalse(batch_norm_params['center'])
        self.assertTrue(batch_norm_params['scale'])
        batch_norm_layer = keras_config.build_batch_norm()
        self.assertTrue(isinstance(batch_norm_layer, freezable_batch_norm.FreezableBatchNorm))

    def test_return_non_default_batch_norm_params_keras_override(self):
        if False:
            return 10
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n      batch_norm {\n        decay: 0.7\n        center: false\n        scale: true\n        epsilon: 0.03\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        keras_config = hyperparams_builder.KerasLayerHyperparams(conv_hyperparams_proto)
        self.assertTrue(keras_config.use_batch_norm())
        batch_norm_params = keras_config.batch_norm_params(momentum=0.4)
        self.assertAlmostEqual(batch_norm_params['momentum'], 0.4)
        self.assertAlmostEqual(batch_norm_params['epsilon'], 0.03)
        self.assertFalse(batch_norm_params['center'])
        self.assertTrue(batch_norm_params['scale'])

    def test_return_batch_norm_params_with_notrain_during_eval(self):
        if False:
            while True:
                i = 10
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n      batch_norm {\n        decay: 0.7\n        center: false\n        scale: true\n        epsilon: 0.03\n        train: true\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=False)
        scope = scope_fn()
        conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
        self.assertEqual(conv_scope_arguments['normalizer_fn'], slim.batch_norm)
        batch_norm_params = scope[_get_scope_key(slim.batch_norm)]
        self.assertAlmostEqual(batch_norm_params['decay'], 0.7)
        self.assertAlmostEqual(batch_norm_params['epsilon'], 0.03)
        self.assertFalse(batch_norm_params['center'])
        self.assertTrue(batch_norm_params['scale'])
        self.assertFalse(batch_norm_params['is_training'])

    def test_return_batch_norm_params_with_notrain_when_train_is_false(self):
        if False:
            while True:
                i = 10
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n      batch_norm {\n        decay: 0.7\n        center: false\n        scale: true\n        epsilon: 0.03\n        train: false\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
        self.assertEqual(conv_scope_arguments['normalizer_fn'], slim.batch_norm)
        batch_norm_params = scope[_get_scope_key(slim.batch_norm)]
        self.assertAlmostEqual(batch_norm_params['decay'], 0.7)
        self.assertAlmostEqual(batch_norm_params['epsilon'], 0.03)
        self.assertFalse(batch_norm_params['center'])
        self.assertTrue(batch_norm_params['scale'])
        self.assertFalse(batch_norm_params['is_training'])

    def test_do_not_use_batch_norm_if_default(self):
        if False:
            for i in range(10):
                print('nop')
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
        self.assertEqual(conv_scope_arguments['normalizer_fn'], None)

    def test_do_not_use_batch_norm_if_default_keras(self):
        if False:
            i = 10
            return i + 15
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        keras_config = hyperparams_builder.KerasLayerHyperparams(conv_hyperparams_proto)
        self.assertFalse(keras_config.use_batch_norm())
        self.assertEqual(keras_config.batch_norm_params(), {})
        identity_layer = keras_config.build_batch_norm()
        self.assertTrue(isinstance(identity_layer, tf.keras.layers.Lambda))

    def test_use_none_activation(self):
        if False:
            for i in range(10):
                print('nop')
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n      activation: NONE\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
        self.assertEqual(conv_scope_arguments['activation_fn'], None)

    def test_use_none_activation_keras(self):
        if False:
            print('Hello World!')
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n      activation: NONE\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        keras_config = hyperparams_builder.KerasLayerHyperparams(conv_hyperparams_proto)
        self.assertEqual(keras_config.params()['activation'], None)
        self.assertEqual(keras_config.params(include_activation=True)['activation'], None)
        activation_layer = keras_config.build_activation_layer()
        self.assertTrue(isinstance(activation_layer, tf.keras.layers.Lambda))
        self.assertEqual(activation_layer.function, tf.identity)

    def test_use_relu_activation(self):
        if False:
            return 10
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n      activation: RELU\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
        self.assertEqual(conv_scope_arguments['activation_fn'], tf.nn.relu)

    def test_use_relu_activation_keras(self):
        if False:
            i = 10
            return i + 15
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n      activation: RELU\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        keras_config = hyperparams_builder.KerasLayerHyperparams(conv_hyperparams_proto)
        self.assertEqual(keras_config.params()['activation'], None)
        self.assertEqual(keras_config.params(include_activation=True)['activation'], tf.nn.relu)
        activation_layer = keras_config.build_activation_layer()
        self.assertTrue(isinstance(activation_layer, tf.keras.layers.Lambda))
        self.assertEqual(activation_layer.function, tf.nn.relu)

    def test_use_relu_6_activation(self):
        if False:
            for i in range(10):
                print('nop')
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n      activation: RELU_6\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
        self.assertEqual(conv_scope_arguments['activation_fn'], tf.nn.relu6)

    def test_use_relu_6_activation_keras(self):
        if False:
            i = 10
            return i + 15
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n      activation: RELU_6\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        keras_config = hyperparams_builder.KerasLayerHyperparams(conv_hyperparams_proto)
        self.assertEqual(keras_config.params()['activation'], None)
        self.assertEqual(keras_config.params(include_activation=True)['activation'], tf.nn.relu6)
        activation_layer = keras_config.build_activation_layer()
        self.assertTrue(isinstance(activation_layer, tf.keras.layers.Lambda))
        self.assertEqual(activation_layer.function, tf.nn.relu6)

    def test_override_activation_keras(self):
        if False:
            i = 10
            return i + 15
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n      activation: RELU_6\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        keras_config = hyperparams_builder.KerasLayerHyperparams(conv_hyperparams_proto)
        new_params = keras_config.params(activation=tf.nn.relu)
        self.assertEqual(new_params['activation'], tf.nn.relu)

    def _assert_variance_in_range(self, initializer, shape, variance, tol=0.01):
        if False:
            i = 10
            return i + 15
        with tf.Graph().as_default() as g:
            with self.test_session(graph=g) as sess:
                var = tf.get_variable(name='test', shape=shape, dtype=tf.float32, initializer=initializer)
                sess.run(tf.global_variables_initializer())
                values = sess.run(var)
                self.assertAllClose(np.var(values), variance, tol, tol)

    def test_variance_in_range_with_variance_scaling_initializer_fan_in(self):
        if False:
            return 10
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        variance_scaling_initializer {\n          factor: 2.0\n          mode: FAN_IN\n          uniform: false\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
        initializer = conv_scope_arguments['weights_initializer']
        self._assert_variance_in_range(initializer, shape=[100, 40], variance=2.0 / 100.0)

    def test_variance_in_range_with_variance_scaling_initializer_fan_in_keras(self):
        if False:
            i = 10
            return i + 15
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        variance_scaling_initializer {\n          factor: 2.0\n          mode: FAN_IN\n          uniform: false\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        keras_config = hyperparams_builder.KerasLayerHyperparams(conv_hyperparams_proto)
        initializer = keras_config.params()['kernel_initializer']
        self._assert_variance_in_range(initializer, shape=[100, 40], variance=2.0 / 100.0)

    def test_variance_in_range_with_variance_scaling_initializer_fan_out(self):
        if False:
            while True:
                i = 10
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        variance_scaling_initializer {\n          factor: 2.0\n          mode: FAN_OUT\n          uniform: false\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
        initializer = conv_scope_arguments['weights_initializer']
        self._assert_variance_in_range(initializer, shape=[100, 40], variance=2.0 / 40.0)

    def test_variance_in_range_with_variance_scaling_initializer_fan_out_keras(self):
        if False:
            i = 10
            return i + 15
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        variance_scaling_initializer {\n          factor: 2.0\n          mode: FAN_OUT\n          uniform: false\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        keras_config = hyperparams_builder.KerasLayerHyperparams(conv_hyperparams_proto)
        initializer = keras_config.params()['kernel_initializer']
        self._assert_variance_in_range(initializer, shape=[100, 40], variance=2.0 / 40.0)

    def test_variance_in_range_with_variance_scaling_initializer_fan_avg(self):
        if False:
            for i in range(10):
                print('nop')
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        variance_scaling_initializer {\n          factor: 2.0\n          mode: FAN_AVG\n          uniform: false\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
        initializer = conv_scope_arguments['weights_initializer']
        self._assert_variance_in_range(initializer, shape=[100, 40], variance=4.0 / (100.0 + 40.0))

    def test_variance_in_range_with_variance_scaling_initializer_fan_avg_keras(self):
        if False:
            print('Hello World!')
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        variance_scaling_initializer {\n          factor: 2.0\n          mode: FAN_AVG\n          uniform: false\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        keras_config = hyperparams_builder.KerasLayerHyperparams(conv_hyperparams_proto)
        initializer = keras_config.params()['kernel_initializer']
        self._assert_variance_in_range(initializer, shape=[100, 40], variance=4.0 / (100.0 + 40.0))

    def test_variance_in_range_with_variance_scaling_initializer_uniform(self):
        if False:
            return 10
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        variance_scaling_initializer {\n          factor: 2.0\n          mode: FAN_IN\n          uniform: true\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
        initializer = conv_scope_arguments['weights_initializer']
        self._assert_variance_in_range(initializer, shape=[100, 40], variance=2.0 / 100.0)

    def test_variance_in_range_with_variance_scaling_initializer_uniform_keras(self):
        if False:
            return 10
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        variance_scaling_initializer {\n          factor: 2.0\n          mode: FAN_IN\n          uniform: true\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        keras_config = hyperparams_builder.KerasLayerHyperparams(conv_hyperparams_proto)
        initializer = keras_config.params()['kernel_initializer']
        self._assert_variance_in_range(initializer, shape=[100, 40], variance=2.0 / 100.0)

    def test_variance_in_range_with_truncated_normal_initializer(self):
        if False:
            while True:
                i = 10
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n          mean: 0.0\n          stddev: 0.8\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
        initializer = conv_scope_arguments['weights_initializer']
        self._assert_variance_in_range(initializer, shape=[100, 40], variance=0.49, tol=0.1)

    def test_variance_in_range_with_truncated_normal_initializer_keras(self):
        if False:
            return 10
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n          mean: 0.0\n          stddev: 0.8\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        keras_config = hyperparams_builder.KerasLayerHyperparams(conv_hyperparams_proto)
        initializer = keras_config.params()['kernel_initializer']
        self._assert_variance_in_range(initializer, shape=[100, 40], variance=0.49, tol=0.1)

    def test_variance_in_range_with_random_normal_initializer(self):
        if False:
            while True:
                i = 10
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        random_normal_initializer {\n          mean: 0.0\n          stddev: 0.8\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        scope_fn = hyperparams_builder.build(conv_hyperparams_proto, is_training=True)
        scope = scope_fn()
        conv_scope_arguments = scope[_get_scope_key(slim.conv2d)]
        initializer = conv_scope_arguments['weights_initializer']
        self._assert_variance_in_range(initializer, shape=[100, 40], variance=0.64, tol=0.1)

    def test_variance_in_range_with_random_normal_initializer_keras(self):
        if False:
            i = 10
            return i + 15
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        random_normal_initializer {\n          mean: 0.0\n          stddev: 0.8\n        }\n      }\n    '
        conv_hyperparams_proto = hyperparams_pb2.Hyperparams()
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams_proto)
        keras_config = hyperparams_builder.KerasLayerHyperparams(conv_hyperparams_proto)
        initializer = keras_config.params()['kernel_initializer']
        self._assert_variance_in_range(initializer, shape=[100, 40], variance=0.64, tol=0.1)
if __name__ == '__main__':
    tf.test.main()