"""Test configs for batch_to_space_nd."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_batch_to_space_nd_tests(options):
    if False:
        i = 10
        return i + 15
    'Make a set of tests to do batch_to_space_nd.'
    test_parameters = [{'dtype': [tf.float32, tf.int64, tf.int32], 'input_shape': [[12, 3, 3, 1]], 'block_shape': [[1, 4], [2, 2], [3, 4]], 'crops': [[[0, 0], [0, 0]], [[1, 1], [1, 1]]], 'constant_block_shape': [True, False], 'constant_crops': [True, False], 'dynamic_range_quantize': [False]}, {'dtype': [tf.float32], 'input_shape': [[1, 3, 3, 1]], 'block_shape': [[1, 1]], 'crops': [[[0, 0], [0, 0]], [[1, 1], [1, 1]]], 'constant_block_shape': [True], 'constant_crops': [True], 'dynamic_range_quantize': [True, False]}, {'dtype': [tf.float32], 'input_shape': [[1, 3, 3, 1]], 'block_shape': [[1, 1]], 'crops': [[[0, 0], [0, 0]], [[1, 1], [1, 1]]], 'constant_block_shape': [True], 'constant_crops': [True], 'fully_quantize': [True], 'quant_16x8': [False, True]}, {'dtype': [tf.float32], 'input_shape': [[1, 3, 3]], 'block_shape': [[1]], 'crops': [[[0, 0]], [[1, 1]]], 'constant_block_shape': [True], 'constant_crops': [True], 'dynamic_range_quantize': [True, False]}, {'dtype': [tf.float32], 'input_shape': [[1, 3, 3]], 'block_shape': [[1]], 'crops': [[[0, 0]], [[1, 1]]], 'constant_block_shape': [True], 'constant_crops': [True], 'fully_quantize': [True], 'quant_16x8': [False, True]}]
    if options.run_with_flex:
        test_parameters = test_parameters + [{'dtype': [tf.float32], 'input_shape': [[8, 2, 2, 2, 1, 1]], 'block_shape': [[2, 2, 2]], 'crops': [[[0, 0], [0, 0], [0, 0]]], 'constant_block_shape': [True, False], 'constant_crops': [True, False], 'dynamic_range_quantize': [False]}]

    def build_graph(parameters):
        if False:
            print('Hello World!')
        'Build a batch_to_space graph given `parameters`.'
        input_tensor = tf.compat.v1.placeholder(dtype=parameters['dtype'], name='input', shape=parameters['input_shape'])
        input_tensors = [input_tensor]
        if parameters['constant_block_shape']:
            block_shape = parameters['block_shape']
        else:
            shape = [len(parameters['block_shape'])]
            block_shape = tf.compat.v1.placeholder(dtype=tf.int32, name='shape', shape=shape)
            input_tensors.append(block_shape)
        if parameters['constant_crops']:
            crops = parameters['crops']
        else:
            shape = [len(parameters['crops']), 2]
            crops = tf.compat.v1.placeholder(dtype=tf.int32, name='crops', shape=shape)
            input_tensors.append(crops)
        out = tf.batch_to_space(input_tensor, block_shape, crops)
        return (input_tensors, [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            return 10
        values = [create_tensor_data(parameters['dtype'], parameters['input_shape'], min_value=-1.0, max_value=1.0)]
        if not parameters['constant_block_shape']:
            values.append(np.array(parameters['block_shape']))
        if not parameters['constant_crops']:
            values.append(np.array(parameters['crops']))
        return (values, sess.run(outputs, feed_dict=dict(zip(inputs, values))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)