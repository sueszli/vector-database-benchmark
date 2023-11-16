"""Tests for object_detection.core.freezable_batch_norm."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import zip
import tensorflow as tf
from object_detection.core import freezable_batch_norm

class FreezableBatchNormTest(tf.test.TestCase):
    """Tests for FreezableBatchNorm operations."""

    def _build_model(self, training=None):
        if False:
            for i in range(10):
                print('nop')
        model = tf.keras.models.Sequential()
        norm = freezable_batch_norm.FreezableBatchNorm(training=training, input_shape=(10,), momentum=0.8)
        model.add(norm)
        return (model, norm)

    def _train_freezable_batch_norm(self, training_mean, training_var):
        if False:
            return 10
        (model, _) = self._build_model()
        model.compile(loss='mse', optimizer='sgd')
        train_data = np.random.normal(loc=training_mean, scale=training_var, size=(1000, 10))
        model.fit(train_data, train_data, epochs=4, verbose=0)
        return model.weights

    def _test_batchnorm_layer(self, norm, should_be_training, test_data, testing_mean, testing_var, training_arg, training_mean, training_var):
        if False:
            print('Hello World!')
        out_tensor = norm(tf.convert_to_tensor(test_data, dtype=tf.float32), training=training_arg)
        out = tf.keras.backend.eval(out_tensor)
        out -= tf.keras.backend.eval(norm.beta)
        out /= tf.keras.backend.eval(norm.gamma)
        if not should_be_training:
            out *= training_var
            out += training_mean - testing_mean
            out /= testing_var
        np.testing.assert_allclose(out.mean(), 0.0, atol=0.15)
        np.testing.assert_allclose(out.std(), 1.0, atol=0.15)

    def test_batchnorm_freezing_training_none(self):
        if False:
            while True:
                i = 10
        with self.test_session():
            training_mean = 5.0
            training_var = 10.0
            testing_mean = -10.0
            testing_var = 5.0
            trained_weights = self._train_freezable_batch_norm(training_mean, training_var)
            (model, norm) = self._build_model(training=True)
            for (trained_weight, blank_weight) in zip(trained_weights, model.weights):
                weight_copy = blank_weight.assign(tf.keras.backend.eval(trained_weight))
                tf.keras.backend.eval(weight_copy)
            test_data = np.random.normal(loc=testing_mean, scale=testing_var, size=(1000, 10))
            training_arg = True
            should_be_training = True
            self._test_batchnorm_layer(norm, should_be_training, test_data, testing_mean, testing_var, training_arg, training_mean, training_var)
            training_arg = False
            should_be_training = False
            self._test_batchnorm_layer(norm, should_be_training, test_data, testing_mean, testing_var, training_arg, training_mean, training_var)
            training_arg = None
            should_be_training = False
            self._test_batchnorm_layer(norm, should_be_training, test_data, testing_mean, testing_var, training_arg, training_mean, training_var)
            tf.keras.backend.set_learning_phase(True)
            should_be_training = True
            self._test_batchnorm_layer(norm, should_be_training, test_data, testing_mean, testing_var, training_arg, training_mean, training_var)
            tf.keras.backend.set_learning_phase(False)
            should_be_training = False
            self._test_batchnorm_layer(norm, should_be_training, test_data, testing_mean, testing_var, training_arg, training_mean, training_var)

    def test_batchnorm_freezing_training_false(self):
        if False:
            print('Hello World!')
        with self.test_session():
            training_mean = 5.0
            training_var = 10.0
            testing_mean = -10.0
            testing_var = 5.0
            trained_weights = self._train_freezable_batch_norm(training_mean, training_var)
            (model, norm) = self._build_model(training=False)
            for (trained_weight, blank_weight) in zip(trained_weights, model.weights):
                weight_copy = blank_weight.assign(tf.keras.backend.eval(trained_weight))
                tf.keras.backend.eval(weight_copy)
            test_data = np.random.normal(loc=testing_mean, scale=testing_var, size=(1000, 10))
            training_arg = True
            should_be_training = False
            self._test_batchnorm_layer(norm, should_be_training, test_data, testing_mean, testing_var, training_arg, training_mean, training_var)
            training_arg = False
            should_be_training = False
            self._test_batchnorm_layer(norm, should_be_training, test_data, testing_mean, testing_var, training_arg, training_mean, training_var)
            training_arg = None
            should_be_training = False
            self._test_batchnorm_layer(norm, should_be_training, test_data, testing_mean, testing_var, training_arg, training_mean, training_var)
            tf.keras.backend.set_learning_phase(True)
            should_be_training = False
            self._test_batchnorm_layer(norm, should_be_training, test_data, testing_mean, testing_var, training_arg, training_mean, training_var)
            tf.keras.backend.set_learning_phase(False)
            should_be_training = False
            self._test_batchnorm_layer(norm, should_be_training, test_data, testing_mean, testing_var, training_arg, training_mean, training_var)
if __name__ == '__main__':
    tf.test.main()