"""Test configs for rfft2d."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import ExtraConvertOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_rfft_tests(options):
    if False:
        i = 10
        return i + 15
    'Make a set of tests to do rfft.'
    test_parameters = [{'input_dtype': [tf.float32], 'input_shape': [[8], [8, 8], [3, 8, 8], [3, 8]], 'fft_length': [None, [4], [8], [16]]}]

    def build_graph(parameters):
        if False:
            print('Hello World!')
        input_value = tf.compat.v1.placeholder(dtype=parameters['input_dtype'], name='input', shape=parameters['input_shape'])
        outs = tf.signal.rfft(input_value, fft_length=parameters['fft_length'])
        return ([input_value], [outs])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            print('Hello World!')
        input_value = create_tensor_data(parameters['input_dtype'], parameters['input_shape'])
        return ([input_value], sess.run(outputs, feed_dict=dict(zip(inputs, [input_value]))))
    extra_convert_options = ExtraConvertOptions()
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs, extra_convert_options)