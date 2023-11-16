"""Test configs for add_n."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_add_n_tests(options):
    if False:
        for i in range(10):
            print('nop')
    'Make a set of tests for AddN op.'
    test_parameters = [{'dtype': [tf.float32, tf.int32], 'input_shape': [[2, 5, 3, 1]], 'num_inputs': [2, 3, 4, 5], 'dynamic_range_quantize': [False]}, {'dtype': [tf.float32, tf.int32], 'input_shape': [[5]], 'num_inputs': [2, 3, 4, 5], 'dynamic_range_quantize': [False]}, {'dtype': [tf.float32, tf.int32], 'input_shape': [[]], 'num_inputs': [2, 3, 4, 5], 'dynamic_range_quantize': [False]}, {'dtype': [tf.float32], 'input_shape': [[]], 'num_inputs': [2, 3, 4, 5], 'dynamic_range_quantize': [True]}]

    def build_graph(parameters):
        if False:
            while True:
                i = 10
        'Builds the graph given the current parameters.'
        input_tensors = []
        for i in range(parameters['num_inputs']):
            input_tensors.append(tf.compat.v1.placeholder(dtype=parameters['dtype'], name='input_{}'.format(i), shape=parameters['input_shape']))
        out = tf.add_n(input_tensors)
        return (input_tensors, [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            print('Hello World!')
        'Builds operand inputs for op.'
        input_data = []
        for _ in range(parameters['num_inputs']):
            input_data.append(create_tensor_data(parameters['dtype'], parameters['input_shape']))
        return (input_data, sess.run(outputs, feed_dict={i: d for (i, d) in zip(inputs, input_data)}))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)