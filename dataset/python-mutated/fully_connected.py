"""Test configs for fully_connected."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_fully_connected_tests(options):
    if False:
        while True:
            i = 10
    'Make a set of tests to do fully_connected.'
    test_parameters = [{'shape1': [[3, 3]], 'shape2': [[3, 3]], 'transpose_a': [True, False], 'transpose_b': [True, False], 'constant_filter': [True, False], 'fully_quantize': [False], 'quant_16x8': [False]}, {'shape1': [[4, 4], [1, 4], [4]], 'shape2': [[4, 4], [4, 1], [4]], 'transpose_a': [False], 'transpose_b': [False], 'constant_filter': [True, False], 'fully_quantize': [False], 'quant_16x8': [False]}, {'shape1': [[40, 37]], 'shape2': [[37, 40]], 'transpose_a': [False], 'transpose_b': [False], 'constant_filter': [True, False], 'fully_quantize': [False], 'quant_16x8': [False]}, {'shape1': [[40, 37]], 'shape2': [[40, 37]], 'transpose_a': [False], 'transpose_b': [True], 'constant_filter': [True, False], 'fully_quantize': [False], 'quant_16x8': [False]}, {'shape1': [[5, 3]], 'shape2': [[5, 3]], 'transpose_a': [True], 'transpose_b': [False], 'constant_filter': [True, False], 'fully_quantize': [False], 'quant_16x8': [False]}, {'shape1': [[1, 3]], 'shape2': [[3, 3]], 'transpose_a': [False], 'transpose_b': [False], 'constant_filter': [True], 'fully_quantize': [True], 'quant_16x8': [False]}, {'shape1': [[1, 4], [4]], 'shape2': [[4, 4], [4, 1], [4]], 'transpose_a': [False], 'transpose_b': [False], 'constant_filter': [True], 'fully_quantize': [True], 'quant_16x8': [False]}, {'shape1': [[1, 37], [2, 37]], 'shape2': [[37, 40]], 'transpose_a': [False], 'transpose_b': [False], 'constant_filter': [True], 'fully_quantize': [True], 'quant_16x8': [False]}, {'shape1': [[1, 3], [2, 3]], 'shape2': [[3, 5], [3, 1]], 'transpose_a': [False], 'transpose_b': [False], 'constant_filter': [True], 'fully_quantize': [True], 'quant_16x8': [False]}, {'shape1': [[2, 3]], 'shape2': [[3, 5]], 'transpose_a': [False], 'transpose_b': [False], 'constant_filter': [True], 'fully_quantize': [True], 'quant_16x8': [True]}, {'shape1': [[0, 3]], 'shape2': [[3, 3]], 'transpose_a': [False], 'transpose_b': [False], 'constant_filter': [True, False], 'fully_quantize': [False], 'quant_16x8': [False]}, {'shape1': [[3, 0]], 'shape2': [[0, 3]], 'transpose_a': [False], 'transpose_b': [False], 'constant_filter': [True, False], 'fully_quantize': [False], 'quant_16x8': [False]}]

    def build_graph(parameters):
        if False:
            return 10
        'Build a matmul graph given `parameters`.'
        input_tensor1 = tf.compat.v1.placeholder(dtype=tf.float32, name='input1', shape=parameters['shape1'])
        if parameters['constant_filter']:
            input_tensor2 = create_tensor_data(np.float32, parameters['shape2'], min_value=-1, max_value=1)
            input_tensors = [input_tensor1]
        else:
            input_tensor2 = tf.compat.v1.placeholder(dtype=tf.float32, name='input2', shape=parameters['shape2'])
            input_tensors = [input_tensor1, input_tensor2]
        out = tf.matmul(input_tensor1, input_tensor2, transpose_a=parameters['transpose_a'], transpose_b=parameters['transpose_b'])
        return (input_tensors, [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            while True:
                i = 10
        'Build list of input values.\n\n    It either contains 1 tensor (input_values1) or\n    2 tensors (input_values1, input_values2) based on whether the second input\n    is a constant or variable input.\n    '
        values = [create_tensor_data(np.float32, shape=parameters['shape1'], min_value=-1, max_value=1)]
        if not parameters['constant_filter']:
            values.append(create_tensor_data(np.float32, parameters['shape2'], min_value=-1, max_value=1))
        return (values, sess.run(outputs, feed_dict=dict(zip(inputs, values))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs, expected_tf_failures=14)