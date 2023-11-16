"""Test configs for complex abs."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_complex_abs_tests(options):
    if False:
        for i in range(10):
            print('nop')
    'Make a set of tests to do complex abs.'
    test_parameters = [{'dtype': [tf.complex64], 'input_shape': [[], [1], [2, 3], [1, 3, 4, 3], [2, 2, 3, 4, 5, 6]], 'Tout': [tf.float32]}, {'dtype': [tf.complex128], 'input_shape': [[], [1], [2, 3], [1, 3, 4, 3], [2, 2, 3, 4, 5, 6]], 'Tout': [tf.float64]}]

    def build_graph(parameters):
        if False:
            while True:
                i = 10
        input_tensor = tf.compat.v1.placeholder(dtype=parameters['dtype'], name='input', shape=parameters['input_shape'])
        out = tf.raw_ops.ComplexAbs(x=input_tensor, Tout=parameters['Tout'])
        return ([input_tensor], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            return 10
        input_values = create_tensor_data(parameters['dtype'].as_numpy_dtype, parameters['input_shape'], min_value=-10, max_value=10)
        return ([input_values], sess.run(outputs, feed_dict=dict(zip(inputs, [input_values]))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)