"""Test configs for bitwise_xor operator."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_bitwise_xor_tests(options):
    if False:
        return 10
    'Generate examples for bitwise_xor.'
    test_parameters = [{'input_dtype': [tf.uint8, tf.int8, tf.uint16, tf.int16, tf.uint32, tf.int32], 'input_shape_pair': [([], []), ([2, 3, 4], [2, 3, 4]), ([1, 1, 1, 3], [1, 1, 1, 3]), ([5, 5], [1]), ([10], [2, 4, 10]), ([2, 3, 3], [2, 3])]}]

    def build_graph(parameters):
        if False:
            while True:
                i = 10
        'Build the bitwise_xor testing graph.'
        input_value1 = tf.compat.v1.placeholder(dtype=parameters['input_dtype'], name='input1', shape=parameters['input_shape_pair'][0])
        input_value2 = tf.compat.v1.placeholder(dtype=parameters['input_dtype'], name='input2', shape=parameters['input_shape_pair'][1])
        out = tf.bitwise.bitwise_xor(input_value1, input_value2)
        return ([input_value1, input_value2], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            i = 10
            return i + 15
        input_value1 = create_tensor_data(parameters['input_dtype'], parameters['input_shape_pair'][0])
        input_value2 = create_tensor_data(parameters['input_dtype'], parameters['input_shape_pair'][1])
        return ([input_value1, input_value2], sess.run(outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2]))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs, expected_tf_failures=6)