"""Test configs for floor."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_floor_tests(options):
    if False:
        i = 10
        return i + 15
    'Make a set of tests to do floor.'
    test_parameters = [{'input_dtype': [tf.float32], 'input_shape': [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]]}]

    def build_graph(parameters):
        if False:
            while True:
                i = 10
        'Build the floor op testing graph.'
        input_value = tf.compat.v1.placeholder(dtype=parameters['input_dtype'], name='input1', shape=parameters['input_shape'])
        out = tf.floor(input_value)
        return ([input_value], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            return 10
        input_value = create_tensor_data(parameters['input_dtype'], parameters['input_shape'])
        return ([input_value], sess.run(outputs, feed_dict={inputs[0]: input_value}))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)