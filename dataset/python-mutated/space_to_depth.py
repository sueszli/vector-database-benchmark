"""Test configs for space_to_depth."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_space_to_depth_tests(options):
    if False:
        for i in range(10):
            print('nop')
    'Make a set of tests to do space_to_depth.'
    test_parameters = [{'dtype': [tf.float32, tf.int32, tf.uint8, tf.int64], 'input_shape': [[2, 12, 24, 1]], 'block_size': [2, 3, 4], 'fully_quantize': [False]}, {'dtype': [tf.float32], 'input_shape': [[2, 12, 24, 1], [1, 12, 24, 1]], 'block_size': [2, 3, 4], 'fully_quantize': [True]}]

    def build_graph(parameters):
        if False:
            i = 10
            return i + 15
        input_tensor = tf.compat.v1.placeholder(dtype=parameters['dtype'], name='input', shape=parameters['input_shape'])
        out = tf.compat.v1.space_to_depth(input=input_tensor, block_size=parameters['block_size'])
        return ([input_tensor], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            for i in range(10):
                print('nop')
        input_values = create_tensor_data(parameters['dtype'], parameters['input_shape'], min_value=-1, max_value=1)
        return ([input_values], sess.run(outputs, feed_dict=dict(zip(inputs, [input_values]))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)