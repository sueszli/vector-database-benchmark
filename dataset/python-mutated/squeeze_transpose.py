"""Test configs for squeeze_transpose."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_squeeze_transpose_tests(options):
    if False:
        print('Hello World!')
    'Make a set of tests to do squeeze followed by transpose.'
    test_parameters = [{'dtype': [tf.int32, tf.float32, tf.int64], 'input_shape': [[1, 4, 10, 1]], 'axis': [[-1], [3]]}]

    def build_graph(parameters):
        if False:
            for i in range(10):
                print('nop')
        input_tensor = tf.compat.v1.placeholder(dtype=parameters['dtype'], name='input', shape=parameters['input_shape'])
        out = tf.squeeze(input_tensor, axis=parameters['axis'])
        out = tf.transpose(a=out, perm=[1, 2])
        return ([input_tensor], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            print('Hello World!')
        input_values = create_tensor_data(parameters['dtype'], parameters['input_shape'])
        return ([input_values], sess.run(outputs, feed_dict=dict(zip(inputs, [input_values]))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs, expected_tf_failures=0)