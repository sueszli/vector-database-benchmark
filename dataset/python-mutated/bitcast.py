"""Test configs for bitcast."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_bitcast_tests(options):
    if False:
        i = 10
        return i + 15
    'Generate examples for bitcast.'
    test_parameters = [{'input_dtype': [tf.int32], 'output_dtype': [tf.uint32], 'input_shape': [[], [1], [1, 2], [3, 4, 5, 6]]}, {'input_dtype': [tf.uint32], 'output_dtype': [tf.int32], 'input_shape': [[], [1], [1, 2], [3, 4, 5, 6]]}, {'input_dtype': [tf.uint32], 'output_dtype': [tf.int16], 'input_shape': [[], [1], [1, 2], [3, 4, 5, 6]]}, {'input_dtype': [tf.int16], 'output_dtype': [tf.uint32], 'input_shape': [[2], [1, 2], [1, 2, 2], [3, 4, 5, 6, 2]]}, {'input_dtype': [tf.int32], 'output_dtype': [tf.int16], 'input_shape': [[], [1], [1, 2], [3, 4, 5, 6]]}, {'input_dtype': [tf.uint16], 'output_dtype': [tf.uint32], 'input_shape': [[2], [1, 2], [1, 2, 2], [3, 4, 5, 6, 2]]}, {'input_dtype': [tf.float32], 'output_dtype': [tf.int16], 'input_shape': [[], [1], [1, 2], [3, 4, 5, 6]]}, {'input_dtype': [tf.int16], 'output_dtype': [tf.float32], 'input_shape': [[2], [1, 2], [1, 2, 2], [3, 4, 5, 6, 2]]}]

    def build_graph(parameters):
        if False:
            return 10
        'Build the bitcast testing graph.'
        input_value = tf.compat.v1.placeholder(dtype=parameters['input_dtype'], name='input', shape=parameters['input_shape'])
        out = tf.bitcast(input_value, parameters['output_dtype'])
        return ([input_value], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            print('Hello World!')
        input_value = create_tensor_data(parameters['input_dtype'], parameters['input_shape'])
        return ([input_value], sess.run(outputs, feed_dict=dict(zip(inputs, [input_value]))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)