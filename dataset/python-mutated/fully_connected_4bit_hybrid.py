"""Test configs for fully_connected_4bit."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_fully_connected_4bit_hybrid_tests(options):
    if False:
        return 10
    'Make a set of tests to do fully_connected.'
    test_parameters = [{'shape1': [[3, 3]], 'shape2': [[3, 3]], 'dynamic_range_quantize': [True]}, {'shape1': [[40, 42]], 'shape2': [[42, 40]], 'dynamic_range_quantize': [True]}, {'shape1': [[1, 40]], 'shape2': [[40, 3]], 'dynamic_range_quantize': [True]}]

    def build_graph(parameters):
        if False:
            return 10
        'Build a matmul graph given `parameters`.'
        input_tensor1 = tf.compat.v1.placeholder(dtype=tf.float32, name='input1', shape=parameters['shape1'])
        float_data = np.random.uniform(-1, 1, parameters['shape2'])
        scale = np.abs(float_data).max() / 7.0
        int_data = np.round(float_data / scale)
        input_tensor2 = tf.constant(int_data, dtype=tf.float32)
        quantized = tf.quantization.fake_quant_with_min_max_vars(input_tensor2, min=-7, max=7, num_bits=4, narrow_range=True)
        out = tf.matmul(input_tensor1, quantized)
        return ([input_tensor1], [out])

    def create_input_data(parameters):
        if False:
            for i in range(10):
                print('nop')
        'Create a float input with no quantization loss.'
        float_data = np.random.random(parameters['shape1']).astype(np.float32)
        scale = np.abs(float_data).max(axis=1, keepdims=True) / 127.0
        return np.round(float_data / scale)

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            i = 10
            return i + 15
        'Build list of input values.\n\n    Use the specialized method, as dynamic range quantization will cause\n    differing outputs from TF, which does not quantize inputs.\n    '
        values = [create_input_data(parameters)]
        return (values, sess.run(outputs, feed_dict=dict(zip(inputs, values))))
    options.experimental_low_bit_qat = True
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs, expected_tf_failures=0)