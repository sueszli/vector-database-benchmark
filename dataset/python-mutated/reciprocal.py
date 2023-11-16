"""Test configs for reciprocal."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function('make_reciprocal_tests')
def make_reciprocal_tests(options):
    if False:
        while True:
            i = 10
    'Make a set of tests to do reciprocal.'
    test_parameters = [{'input_dtype': [tf.float32, tf.int32, tf.int64], 'input_shape': [[1, 2], [1, 2, 3, 4], [10]]}]

    def build_graph(parameters):
        if False:
            for i in range(10):
                print('nop')
        'Build the graph for cond tests.'
        input_tensor = tf.compat.v1.placeholder(dtype=parameters['input_dtype'], name='input', shape=parameters['input_shape'])
        out = tf.math.reciprocal(input_tensor)
        return ([input_tensor], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            return 10
        input_values = [create_tensor_data(parameters['input_dtype'], parameters['input_shape'])]
        return (input_values, sess.run(outputs, feed_dict=dict(zip(inputs, input_values))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs, expected_tf_failures=6)