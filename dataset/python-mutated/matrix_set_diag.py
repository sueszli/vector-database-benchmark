"""Test configs for matrix_set_diag."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_matrix_set_diag_tests(options):
    if False:
        i = 10
        return i + 15
    'Make a set of tests for tf.linalg.set_diag op.'
    test_parameters = [{'input_diag_shapes': [([3, 3], [3]), ([2, 3], [2]), ([2, 4, 4], [2, 4]), ([3, 4, 5, 6], [3, 4, 5])], 'input_dtype': [tf.int32, tf.float32, tf.uint8]}]

    def build_graph(parameters):
        if False:
            for i in range(10):
                print('nop')
        input_shape = parameters['input_diag_shapes'][0]
        diag_shape = parameters['input_diag_shapes'][1]
        input_tensor = tf.compat.v1.placeholder(dtype=parameters['input_dtype'], name='input', shape=input_shape)
        diag_tensor = tf.compat.v1.placeholder(dtype=parameters['input_dtype'], name='diagonal', shape=diag_shape)
        outs = tf.linalg.set_diag(input_tensor, diag_tensor)
        return ([input_tensor, diag_tensor], [outs])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            return 10
        input_shape = parameters['input_diag_shapes'][0]
        diag_shape = parameters['input_diag_shapes'][1]
        input_values = create_tensor_data(parameters['input_dtype'], input_shape)
        diag_values = create_tensor_data(parameters['input_dtype'], diag_shape)
        return ([input_values, diag_values], sess.run(outputs, feed_dict=dict(zip(inputs, [input_values, diag_values]))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)