"""Test configs for exp."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_conv3d_tests(options):
    if False:
        return 10
    'Make a set of tests to do conv3d.'
    test_parameters = [{'input_dtype': [tf.float32], 'input_shape': [[2, 3, 4, 5, 3], [2, 5, 6, 8, 3]], 'filter_shape': [[2, 2, 2, 3, 2], [1, 2, 2, 3, 2]], 'strides': [(1, 1, 1, 1, 1), (1, 1, 1, 2, 1), (1, 1, 2, 2, 1), (1, 2, 1, 2, 1), (1, 2, 2, 2, 1)], 'dilations': [(1, 1, 1, 1, 1)], 'padding': ['SAME', 'VALID']}]

    def build_graph(parameters):
        if False:
            while True:
                i = 10
        'Build the exp op testing graph.'
        input_tensor = tf.compat.v1.placeholder(dtype=parameters['input_dtype'], name='input', shape=parameters['input_shape'])
        filter_tensor = tf.compat.v1.placeholder(dtype=parameters['input_dtype'], name='filter', shape=parameters['filter_shape'])
        out = tf.nn.conv3d(input_tensor, filter_tensor, strides=parameters['strides'], dilations=parameters['dilations'], padding=parameters['padding'])
        return ([input_tensor, filter_tensor], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            while True:
                i = 10
        values = [create_tensor_data(parameters['input_dtype'], parameters['input_shape'], min_value=-100, max_value=9), create_tensor_data(parameters['input_dtype'], parameters['filter_shape'], min_value=-3, max_value=3)]
        return (values, sess.run(outputs, feed_dict=dict(zip(inputs, values))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)