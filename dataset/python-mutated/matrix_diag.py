"""Test configs for matrix_diag."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_matrix_diag_tests(options):
    if False:
        print('Hello World!')
    'Make a set of tests for tf.linalg.diag op.'
    test_parameters = [{'input_shape': [[3], [2, 3], [3, 4, 5], [2, 4, 6, 8]], 'input_dtype': [tf.int32, tf.float32]}]

    def build_graph(parameters):
        if False:
            while True:
                i = 10
        input_tensor = tf.compat.v1.placeholder(dtype=parameters['input_dtype'], name='input', shape=parameters['input_shape'])
        outs = tf.linalg.diag(input_tensor)
        return ([input_tensor], [outs])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            return 10
        input_values = create_tensor_data(parameters['input_dtype'], parameters['input_shape'])
        return ([input_values], sess.run(outputs, feed_dict=dict(zip(inputs, [input_values]))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)