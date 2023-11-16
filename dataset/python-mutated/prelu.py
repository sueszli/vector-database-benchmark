"""Test configs for prelu."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_prelu_tests(options):
    if False:
        i = 10
        return i + 15
    'Make a set of tests to do PReLU.'
    test_parameters = [{'input_shape': [[1, 10, 10, 3], [3, 3, 3, 3]], 'shared_axes': [[1, 2], [1]], 'fully_quantize': [False], 'input_range': [(-10, 10)]}, {'input_shape': [[20, 20], [20, 20, 20]], 'shared_axes': [[1]], 'fully_quantize': [False], 'input_range': [(-10, 10)]}, {'input_shape': [[1, 10, 10, 3], [3, 3, 3, 3]], 'shared_axes': [[1, 2], [1]], 'fully_quantize': [True], 'input_range': [(-10, 10)]}, {'input_shape': [[20, 20], [20, 20, 20]], 'shared_axes': [[1]], 'fully_quantize': [True], 'input_range': [(-10, 10)]}]

    def build_graph(parameters):
        if False:
            print('Hello World!')
        'Build the graph for the test case.'
        input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, name='input', shape=parameters['input_shape'])
        prelu = tf.keras.layers.PReLU(shared_axes=parameters['shared_axes'])
        out = prelu(input_tensor)
        return ([input_tensor], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            i = 10
            return i + 15
        'Build the inputs for the test case.'
        input_shape = parameters['input_shape']
        input_values = create_tensor_data(np.float32, input_shape, min_value=-10, max_value=10)
        shared_axes = parameters['shared_axes']
        alpha_shape = []
        for dim in range(1, len(input_shape)):
            alpha_shape.append(1 if dim in shared_axes else input_shape[dim])
        alpha_values = create_tensor_data(np.float32, alpha_shape, min_value=-5, max_value=5)
        variables = tf.compat.v1.all_variables()
        assert len(variables) == 1
        sess.run(variables[0].assign(alpha_values))
        return ([input_values], sess.run(outputs, feed_dict=dict(zip(inputs, [input_values]))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs, use_frozen_graph=True)