"""Test configs for conv_with_shared_weights."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_conv_with_shared_weights_tests(options):
    if False:
        i = 10
        return i + 15
    'Make a test where 2 Conv ops shared the same constant weight tensor.'
    test_parameters = [{'input_shape': [[1, 10, 10, 3]], 'filter_shape': [[3, 3]], 'strides': [[1, 1, 1, 1]], 'dilations': [[1, 1, 1, 1]], 'padding': ['SAME'], 'data_format': ['NHWC'], 'channel_multiplier': [1], 'dynamic_range_quantize': [False, True]}]

    def get_tensor_shapes(parameters):
        if False:
            print('Hello World!')
        input_shape = parameters['input_shape']
        filter_size = parameters['filter_shape']
        filter_shape = filter_size + [input_shape[3], parameters['channel_multiplier']]
        return [input_shape, filter_shape]

    def build_graph(parameters):
        if False:
            return 10
        'Build a conv graph given `parameters`.'
        (input_shape, filter_shape) = get_tensor_shapes(parameters)
        input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, name='input', shape=input_shape)
        input_tensors = [input_tensor]
        filter_tensor = tf.constant(create_tensor_data(np.float32, filter_shape), dtype=tf.float32)
        conv_input = input_tensor + 0.1
        result1 = tf.nn.conv2d(input=conv_input, filters=filter_tensor, strides=parameters['strides'], dilations=parameters['dilations'], padding=parameters['padding'], data_format=parameters['data_format'])
        result2 = tf.nn.conv2d(input=conv_input, filters=filter_tensor, strides=parameters['strides'], dilations=parameters['dilations'], padding=parameters['padding'], data_format=parameters['data_format'])
        result1 = result1 * 2
        result2 = result2 * 3
        out = result1 + result2
        return (input_tensors, [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            print('Hello World!')
        (input_shape, unused_filter_shape) = get_tensor_shapes(parameters)
        values = [create_tensor_data(np.float32, input_shape)]
        return (values, sess.run(outputs, feed_dict=dict(zip(inputs, values))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)