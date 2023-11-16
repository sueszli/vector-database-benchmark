"""Test configs for padv2."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_padv2_tests(options):
    if False:
        return 10
    'Make a set of tests to do padv2.'
    test_parameters = [{'dtype': [tf.int32, tf.int64, tf.float32], 'input_shape': [[1, 1, 2, 1, 1], [2, 1, 1, 1, 1]], 'paddings': [[[0, 0], [0, 1], [2, 3], [0, 0], [1, 0]], [[0, 1], [0, 0], [0, 0], [2, 3], [1, 0]]], 'constant_paddings': [True, False], 'constant_values': [0, 2]}, {'dtype': [tf.int32, tf.int64, tf.float32], 'input_shape': [[1, 1, 2, 1], [2, 1, 1, 1]], 'paddings': [[[0, 0], [0, 1], [2, 3], [0, 0]], [[0, 1], [0, 0], [0, 0], [2, 3]]], 'constant_paddings': [True, False], 'constant_values': [0, 2]}, {'dtype': [tf.int32, tf.int64, tf.float32], 'input_shape': [[1, 2]], 'paddings': [[[0, 1], [2, 3]]], 'constant_paddings': [True, False], 'constant_values': [0, 2]}, {'dtype': [tf.int32], 'input_shape': [[1]], 'paddings': [[[0, 1]]], 'constant_paddings': [False], 'constant_values': [0, 2]}]

    def build_graph(parameters):
        if False:
            print('Hello World!')
        'Build a pad graph given `parameters`.'
        input_tensor = tf.compat.v1.placeholder(dtype=parameters['dtype'], name='input', shape=parameters['input_shape'])
        if parameters['constant_paddings']:
            paddings = parameters['paddings']
            input_tensors = [input_tensor]
        else:
            shape = [len(parameters['paddings']), 2]
            paddings = tf.compat.v1.placeholder(dtype=tf.int32, name='padding', shape=shape)
            input_tensors = [input_tensor, paddings]
        out = tf.pad(tensor=input_tensor, paddings=paddings, constant_values=parameters['constant_values'])
        return (input_tensors, [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            return 10
        values = [create_tensor_data(parameters['dtype'], parameters['input_shape'])]
        if not parameters['constant_paddings']:
            values.append(np.array(parameters['paddings']))
        return (values, sess.run(outputs, feed_dict=dict(zip(inputs, values))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)