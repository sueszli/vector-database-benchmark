"""Test configs for splitv."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_splitv_tests(options):
    if False:
        for i in range(10):
            print('nop')
    'Make a set of tests to do tf.split_v.'
    test_parameters = [{'input_shape': [[1, 3, 4, 6], [2, 4, 1], [6, 4], [8]], 'size_splits': [[2, 2], [1, 3], [4, 2], [5, 3], [-1, 1], [-1, 2], [-1, 4]], 'axis': [0, 1, 2, 3, -4, -3, -2, -1]}]

    def build_graph(parameters):
        if False:
            return 10
        input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, name='input', shape=parameters['input_shape'])
        out = tf.split(input_tensor, parameters['size_splits'], parameters['axis'])
        return ([input_tensor], [out[0]])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            print('Hello World!')
        values = [create_tensor_data(np.float32, parameters['input_shape'])]
        return (values, sess.run(outputs, feed_dict=dict(zip(inputs, values))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs, expected_tf_failures=158)