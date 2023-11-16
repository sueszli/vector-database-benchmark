"""Test configs for broadcast_args."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function('make_broadcast_args_tests')
def make_broadcast_args_tests(options):
    if False:
        print('Hello World!')
    'Make a set of tests to do broadcast_args.'
    test_parameters = [{'dtype': [tf.int64, tf.int32], 'input1_shape': [[1], [4], [3, 4], [1, 3, 4]], 'input2_shape': [[6, 4, 3, 4]]}, {'dtype': [tf.int64, tf.int32], 'input1_shape': [[1, 4, 0]], 'input2_shape': [[3, 1, 0], [3, 4, 1]]}]

    def build_graph(parameters):
        if False:
            i = 10
            return i + 15
        'Build the graph for broadcast_args tests.'
        shape1_tensor = tf.compat.v1.placeholder(dtype=parameters['dtype'], name='input1', shape=[len(parameters['input1_shape'])])
        shape2_tensor = tf.compat.v1.placeholder(dtype=parameters['dtype'], name='input2', shape=[len(parameters['input2_shape'])])
        out = tf.raw_ops.BroadcastArgs(s0=shape1_tensor, s1=shape2_tensor)
        return ([shape1_tensor, shape2_tensor], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            for i in range(10):
                print('nop')
        input_values = [np.array(parameters['input1_shape']).astype(parameters['dtype'].as_numpy_dtype), np.array(parameters['input2_shape']).astype(parameters['dtype'].as_numpy_dtype)]
        return (input_values, sess.run(outputs, feed_dict=dict(zip(inputs, input_values))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)