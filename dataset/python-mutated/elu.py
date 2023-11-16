"""Test configs for elu."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_elu_tests(options):
    if False:
        return 10
    'Make a set of tests to do (float) tf.nn.elu.'
    test_parameters = [{'input_shape': [[], [1], [2, 3], [1, 1, 1, 1], [1, 3, 4, 3], [3, 15, 14, 3], [3, 1, 2, 4, 6], [2, 2, 3, 4, 5, 6]]}]

    def build_graph(parameters):
        if False:
            return 10
        'Build the graph for the test case.'
        input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, name='input', shape=parameters['input_shape'])
        out = tf.nn.elu(input_tensor)
        return ([input_tensor], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            i = 10
            return i + 15
        'Build the inputs for the test case.'
        input_values = create_tensor_data(np.float32, parameters['input_shape'], min_value=-4, max_value=10)
        return ([input_values], sess.run(outputs, feed_dict=dict(zip(inputs, [input_values]))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)