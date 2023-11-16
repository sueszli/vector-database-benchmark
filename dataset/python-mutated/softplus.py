"""Test configs for exp."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_softplus_tests(options):
    if False:
        while True:
            i = 10
    'Make a set of tests to do softplus.'
    test_parameters = [{'input_dtype': [tf.float32], 'input_shape': [[], [3], [1, 100], [4, 2, 3], [5, 224, 224, 3]]}]

    def build_graph(parameters):
        if False:
            return 10
        'Build the exp op testing graph.'
        input_tensor = tf.compat.v1.placeholder(dtype=parameters['input_dtype'], name='input', shape=parameters['input_shape'])
        out = tf.math.softplus(input_tensor)
        return ([input_tensor], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            while True:
                i = 10
        values = [create_tensor_data(parameters['input_dtype'], parameters['input_shape'], min_value=-100, max_value=9)]
        return (values, sess.run(outputs, feed_dict=dict(zip(inputs, values))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)