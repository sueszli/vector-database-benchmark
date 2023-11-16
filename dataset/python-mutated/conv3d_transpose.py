"""Test configs for conv3d_transpose."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_conv3d_transpose_tests(options):
    if False:
        return 10
    'Make a set of tests to do conv3d_transpose.'
    test_parameters = [{'shape_dtype': [tf.int32, tf.int64], 'input_dtype': [tf.float32], 'input_shape': [[2, 3, 4, 5, 2], [2, 5, 6, 8, 2]], 'filter_shape': [[2, 2, 2, 3, 2], [1, 2, 2, 3, 2]], 'strides': [(1, 1, 1, 1, 1), (1, 1, 1, 2, 1), (1, 1, 2, 2, 1), (1, 2, 1, 2, 1), (1, 2, 2, 2, 1)], 'dilations': [(1, 1, 1, 1, 1)], 'padding': ['SAME', 'VALID']}]

    def build_graph(parameters):
        if False:
            i = 10
            return i + 15
        'Build the exp op testing graph.'
        output_shape = tf.compat.v1.placeholder(dtype=parameters['shape_dtype'], name='input', shape=[5])
        input_tensor = tf.compat.v1.placeholder(dtype=parameters['input_dtype'], name='input', shape=parameters['input_shape'])
        filter_tensor = tf.compat.v1.placeholder(dtype=parameters['input_dtype'], name='filter', shape=parameters['filter_shape'])
        out = tf.nn.conv3d_transpose(input_tensor, filter_tensor, output_shape, strides=parameters['strides'], dilations=parameters['dilations'], padding=parameters['padding'])
        return ([input_tensor, filter_tensor, output_shape], [out])

    def calculate_output_shape(parameters):
        if False:
            print('Hello World!')

        def calculate_shape(idx):
            if False:
                i = 10
                return i + 15
            input_size = parameters['input_shape'][idx]
            filter_size = parameters['filter_shape'][idx - 1]
            stride = parameters['strides'][idx]
            if parameters['padding'] == 'SAME':
                return (input_size - 1) * stride + 1
            else:
                return (input_size - 1) * stride + filter_size
        output_shape_values = [parameters['input_shape'][0]]
        output_shape_values.append(calculate_shape(1))
        output_shape_values.append(calculate_shape(2))
        output_shape_values.append(calculate_shape(3))
        output_shape_values.append(parameters['filter_shape'][3])
        return np.dtype(parameters['shape_dtype'].as_numpy_dtype()).type(output_shape_values)

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            print('Hello World!')
        values = [create_tensor_data(parameters['input_dtype'], parameters['input_shape'], min_value=-100, max_value=9), create_tensor_data(parameters['input_dtype'], parameters['filter_shape'], min_value=-3, max_value=3), calculate_output_shape(parameters)]
        return (values, sess.run(outputs, feed_dict=dict(zip(inputs, values))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)