"""This module tests generic behavior of reference data tests.

This test is not intended to test every layer of interest, and models should
test the layers that affect them. This test is primarily focused on ensuring
that reference_data.BaseTest functions as intended. If there is a legitimate
change such as a change to TensorFlow which changes graph construction, tests
can be regenerated with the following command:

  $ python3 reference_data_test.py -regen
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import unittest
import tensorflow as tf
from official.utils.misc import keras_utils
from official.utils.testing import reference_data

class GoldenBaseTest(reference_data.BaseTest):
    """Class to ensure that reference data testing runs properly."""

    def setUp(self):
        if False:
            while True:
                i = 10
        if keras_utils.is_v2_0():
            tf.compat.v1.disable_eager_execution()
        super(GoldenBaseTest, self).setUp()

    @property
    def test_name(self):
        if False:
            i = 10
            return i + 15
        return 'reference_data_test'

    def _uniform_random_ops(self, test=False, wrong_name=False, wrong_shape=False, bad_seed=False, bad_function=False):
        if False:
            while True:
                i = 10
        'Tests number generation and failure modes.\n\n    This test is of a very simple graph: the generation of a 1x1 random tensor.\n    However, it is also used to confirm that the tests are actually checking\n    properly by failing in predefined ways.\n\n    Args:\n      test: Whether or not to run as a test case.\n      wrong_name: Whether to assign the wrong name to the tensor.\n      wrong_shape: Whether to create a tensor with the wrong shape.\n      bad_seed: Whether or not to perturb the random seed.\n      bad_function: Whether to perturb the correctness function.\n    '
        name = 'uniform_random'
        g = tf.Graph()
        with g.as_default():
            seed = self.name_to_seed(name)
            seed = seed + 1 if bad_seed else seed
            tf.compat.v1.set_random_seed(seed)
            tensor_name = 'wrong_tensor' if wrong_name else 'input_tensor'
            tensor_shape = (1, 2) if wrong_shape else (1, 1)
            input_tensor = tf.compat.v1.get_variable(tensor_name, dtype=tf.float32, initializer=tf.random.uniform(tensor_shape, maxval=1))

        def correctness_function(tensor_result):
            if False:
                print('Hello World!')
            result = float(tensor_result[0, 0])
            result = result + 0.1 if bad_function else result
            return [result]
        self._save_or_test_ops(name=name, graph=g, ops_to_eval=[input_tensor], test=test, correctness_function=correctness_function)

    def _dense_ops(self, test=False):
        if False:
            i = 10
            return i + 15
        name = 'dense'
        g = tf.Graph()
        with g.as_default():
            tf.compat.v1.set_random_seed(self.name_to_seed(name))
            input_tensor = tf.compat.v1.get_variable('input_tensor', dtype=tf.float32, initializer=tf.random.uniform((1, 2), maxval=1))
            layer = tf.compat.v1.layers.dense(inputs=input_tensor, units=4)
            layer = tf.compat.v1.layers.dense(inputs=layer, units=1)
        self._save_or_test_ops(name=name, graph=g, ops_to_eval=[layer], test=test, correctness_function=self.default_correctness_function)

    def test_uniform_random(self):
        if False:
            i = 10
            return i + 15
        self._uniform_random_ops(test=True)

    def test_tensor_name_error(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(AssertionError):
            self._uniform_random_ops(test=True, wrong_name=True)

    @unittest.skipIf(keras_utils.is_v2_0(), 'TODO:(b/136010138) Fails on TF 2.0.')
    def test_tensor_shape_error(self):
        if False:
            print('Hello World!')
        with self.assertRaises(AssertionError):
            self._uniform_random_ops(test=True, wrong_shape=True)

    def test_incorrectness_function(self):
        if False:
            return 10
        with self.assertRaises(AssertionError):
            self._uniform_random_ops(test=True, bad_function=True)

    def test_dense(self):
        if False:
            while True:
                i = 10
        self._dense_ops(test=True)

    def regenerate(self):
        if False:
            print('Hello World!')
        self._uniform_random_ops(test=False)
        self._dense_ops(test=False)
if __name__ == '__main__':
    reference_data.main(argv=sys.argv, test_class=GoldenBaseTest)