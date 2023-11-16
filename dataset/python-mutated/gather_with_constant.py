"""Test configs for gather_with_constant."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_gather_with_constant_tests(options):
    if False:
        print('Hello World!')
    'Make a set of test which feed a constant to gather.'
    test_parameters = [{'input_shape': [[3]], 'reference_shape': [[2]]}, {'input_shape': [[2, 3]], 'reference_shape': [[2, 3]]}]

    def build_graph(parameters):
        if False:
            for i in range(10):
                print('nop')
        'Build a graph where the inputs to Gather are constants.'
        reference = tf.compat.v1.placeholder(dtype=tf.int32, shape=parameters['reference_shape'])
        gather_input = tf.constant(create_tensor_data(tf.int32, parameters['input_shape']))
        gather_indices = tf.constant([0, 1], tf.int32)
        out = tf.equal(reference, tf.gather(gather_input, gather_indices))
        return ([reference], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            while True:
                i = 10
        reference_values = np.zeros(parameters['reference_shape'], dtype=np.int32)
        return ([reference_values], sess.run(outputs, feed_dict={inputs[0]: reference_values}))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)