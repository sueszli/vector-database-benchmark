"""Test configs for reverse_sequence."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_reverse_sequence_tests(options):
    if False:
        for i in range(10):
            print('nop')
    'Make a set of tests to do reverse_sequence.'
    test_parameters = [{'input_dtype': [tf.float32, tf.int32, tf.int64], 'input_shape': [[8, 4, 5, 5, 6], [4, 4, 3, 5]], 'seq_lengths': [[2, 2, 2, 2], [2, 1, 1, 0]], 'seq_axis': [0, 3], 'batch_axis': [1]}, {'input_dtype': [tf.float32], 'input_shape': [[2, 4, 5, 5, 6]], 'seq_lengths': [[2, 1]], 'seq_axis': [2], 'batch_axis': [0]}, {'input_dtype': [tf.float32], 'input_shape': [[4, 2]], 'seq_lengths': [[3, 1]], 'seq_axis': [0], 'batch_axis': [1]}]

    def build_graph(parameters):
        if False:
            print('Hello World!')
        'Build the graph for reverse_sequence tests.'
        input_value = tf.compat.v1.placeholder(dtype=parameters['input_dtype'], name='input', shape=parameters['input_shape'])
        outs = tf.reverse_sequence(input=input_value, seq_lengths=parameters['seq_lengths'], batch_axis=parameters['batch_axis'], seq_axis=parameters['seq_axis'])
        return ([input_value], [outs])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            print('Hello World!')
        input_value = create_tensor_data(parameters['input_dtype'], parameters['input_shape'])
        return ([input_value], sess.run(outputs, feed_dict=dict(zip(inputs, [input_value]))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)