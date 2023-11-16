"""Test configs for unroll_batch_matmul."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_unroll_batch_matmul_tests(options):
    if False:
        return 10
    'Make a set of tests to test unroll_batch_matmul.'
    broadcast_shape_params = [[(1, 2, 3), (3, 5), False, False], [(2, 5, 3), (3, 7), False, False], [(1, 5, 3), (4, 3, 7), False, False], [(3, 1, 5, 3), (1, 4, 3, 7), False, False]]
    test_parameters = [{'dtype': [tf.float32], 'shape': [[(2, 2, 3), (2, 3, 2), False, False], [(2, 2, 3), (2, 3, 2), True, True], [(2, 2, 3), (2, 2, 3), False, True], [(2, 2, 3), (2, 2, 3), True, False], [(4, 2, 2, 3), (4, 2, 3, 2), False, False], [(4, 2, 2, 3), (4, 2, 3, 2), True, True], [(4, 2, 2, 3), (4, 2, 2, 3), False, True], [(4, 2, 2, 3), (4, 2, 2, 3), True, False]] + broadcast_shape_params}]

    def build_graph(parameters):
        if False:
            i = 10
            return i + 15
        'Build the batch_matmul op testing graph.'

        def _build_graph():
            if False:
                while True:
                    i = 10
            'Build the graph.'
            input_tensor1 = tf.compat.v1.placeholder(dtype=parameters['dtype'], shape=parameters['shape'][0])
            input_tensor2 = tf.compat.v1.placeholder(dtype=parameters['dtype'], shape=parameters['shape'][1])
            out = tf.matmul(input_tensor1, input_tensor2, transpose_a=parameters['shape'][2], transpose_b=parameters['shape'][3])
            return ([input_tensor1, input_tensor2], [out])
        return _build_graph()

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            for i in range(10):
                print('nop')
        input_value1 = create_tensor_data(parameters['dtype'], shape=parameters['shape'][0])
        input_value2 = create_tensor_data(parameters['dtype'], shape=parameters['shape'][1])
        return ([input_value1, input_value2], sess.run(outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2]))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)