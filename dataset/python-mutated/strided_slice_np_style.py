"""Test configs for strided_slice_np_style."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_strided_slice_np_style_tests(options):
    if False:
        return 10
    'Make a set of tests to test strided_slice in np style.'
    test_parameters = [{'dtype': [tf.float32], 'shape': [[12, 7], [33, 1]], 'spec': [[slice(3, 7, 2), slice(None)], [tf.newaxis, slice(3, 7, 1), tf.newaxis, slice(None)], [slice(1, 5, 1), slice(None)]]}, {'dtype': [tf.float32], 'shape': [[44]], 'spec': [[slice(3, 7, 2)], [tf.newaxis, slice(None)]]}, {'dtype': [tf.float32], 'shape': [[21, 15, 7]], 'spec': [[slice(3, 7, 2), slice(None), 2]]}, {'dtype': [tf.float32], 'shape': [[21, 15, 7]], 'spec': [[slice(3, 7, 2), Ellipsis], [slice(1, 11, 3), Ellipsis, slice(3, 7, 2)]]}, {'dtype': [tf.float32], 'shape': [[21, 15, 7, 9]], 'spec': [[slice(3, 7, 2), Ellipsis]]}, {'dtype': [tf.float32], 'shape': [[11, 21, 15, 7, 9]], 'spec': [[slice(3, 7, 2), slice(None), slice(None), slice(None), slice(None)]]}, {'dtype': [tf.float32], 'shape': [[22, 15, 7]], 'spec': [[2, Ellipsis]]}, {'dtype': [tf.float32], 'shape': [[23, 15, 7]], 'spec': [[tf.newaxis, slice(3, 7, 2), slice(None), Ellipsis], [tf.newaxis, slice(3, 7, 2), slice(None), Ellipsis, tf.newaxis]]}, {'dtype': [tf.float32], 'shape': [[21, 15, 7]], 'spec': [[Ellipsis, slice(3, 7, 2)]]}, {'dtype': [tf.float32], 'shape': [[21, 15, 7, 9]], 'spec': [[Ellipsis, slice(3, 7, 2)], [slice(1, 11, 3), Ellipsis, slice(3, 7, 2)]]}, {'dtype': [tf.float32], 'shape': [[11, 21, 15, 7, 9]], 'spec': [[Ellipsis, slice(3, 7, 2)]]}, {'dtype': [tf.float32], 'shape': [[22, 15, 7]], 'spec': [[Ellipsis, 2]]}]

    def build_graph(parameters):
        if False:
            print('Hello World!')
        'Build a simple graph with np style strided_slice.'
        input_value = tf.compat.v1.placeholder(dtype=parameters['dtype'], shape=parameters['shape'])
        out = input_value.__getitem__(parameters['spec'])
        return ([input_value], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            print('Hello World!')
        input_value = create_tensor_data(parameters['dtype'], parameters['shape'])
        return ([input_value], sess.run(outputs, feed_dict=dict(zip(inputs, [input_value]))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)