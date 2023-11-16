"""Test configs for relu6."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_relu6_tests(options):
    if False:
        while True:
            i = 10
    'Make a set of tests to do relu6.'
    test_parameters = [{'input_shape': [[], [1, 1, 1, 1], [1, 3, 4, 3], [3, 15, 14, 3], [3, 1, 2, 4, 6], [2, 2, 3, 4, 5, 6]], 'fully_quantize': [True, False], 'input_range': [(-2, 8)]}]

    def build_graph(parameters):
        if False:
            return 10
        input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, name='input', shape=parameters['input_shape'])
        out = tf.nn.relu6(input_tensor)
        return ([input_tensor], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            for i in range(10):
                print('nop')
        (min_value, max_value) = parameters['input_range']
        input_values = create_tensor_data(np.float32, parameters['input_shape'], min_value, max_value)
        return ([input_values], sess.run(outputs, feed_dict=dict(zip(inputs, [input_values]))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)