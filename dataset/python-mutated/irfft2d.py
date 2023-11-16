"""Test configs for irfft2d."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import ExtraConvertOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_irfft2d_tests(options):
    if False:
        i = 10
        return i + 15
    'Make a set of tests to do irfft2d.'
    test_parameters = [{'input_dtype': [tf.complex64], 'input_shape': [[4, 3]], 'fft_length': [[4, 4], [2, 2], [2, 4]]}, {'input_dtype': [tf.complex64], 'input_shape': [[3, 8, 5]], 'fft_length': [[2, 4], [2, 8], [8, 8]]}, {'input_dtype': [tf.complex64], 'input_shape': [[3, 1, 9]], 'fft_length': [[1, 8], [1, 16]]}]

    def build_graph(parameters):
        if False:
            return 10
        input_value = tf.compat.v1.placeholder(dtype=parameters['input_dtype'], name='input', shape=parameters['input_shape'])
        outs = tf.signal.irfft2d(input_value, fft_length=parameters['fft_length'])
        return ([input_value], [outs])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            while True:
                i = 10
        rfft_length = []
        rfft_length.append(parameters['input_shape'][-2])
        rfft_length.append((parameters['input_shape'][-1] - 1) * 2)
        rfft_input = create_tensor_data(np.float32, parameters['input_shape'])
        rfft_result = np.fft.rfft2(rfft_input, rfft_length)
        return ([rfft_result], sess.run(outputs, feed_dict=dict(zip(inputs, [rfft_result]))))
    extra_convert_options = ExtraConvertOptions()
    extra_convert_options.allow_custom_ops = True
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs, extra_convert_options)