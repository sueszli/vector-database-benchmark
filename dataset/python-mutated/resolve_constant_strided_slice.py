"""Test configs for resolve_constant_strided_slice."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_resolve_constant_strided_slice_tests(options):
    if False:
        print('Hello World!')
    'Make a set of tests to show strided_slice yields incorrect results.'
    test_parameters = [{'unused_iteration_counter': [1]}]

    def build_graph(parameters):
        if False:
            while True:
                i = 10
        'Build the strided_slice op testing graph.'
        del parameters
        input_values = tf.compat.v1.placeholder(dtype=tf.float32, shape=[4, 2])
        data = tf.constant([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], tf.float32)
        return ([input_values], [input_values + data[:, :2]])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            for i in range(10):
                print('nop')
        del parameters
        input_values = np.zeros([4, 2], dtype=np.float32)
        return ([input_values], sess.run(outputs, feed_dict={inputs[0]: input_values}))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)