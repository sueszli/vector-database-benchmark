"""Test configs for strided_slice operators."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

def _make_shape_to_strided_slice_test(options, test_parameters, expected_tf_failures=0):
    if False:
        for i in range(10):
            print('nop')
    'Utility function to make shape_to_strided_slice_tests.'

    def build_graph(parameters):
        if False:
            for i in range(10):
                print('nop')
        'Build graph for shape_stride_slice test.'
        input_tensor = tf.compat.v1.placeholder(dtype=parameters['dtype'], name='input', shape=parameters['dynamic_input_shape'])
        begin = parameters['begin']
        end = parameters['end']
        strides = parameters['strides']
        tensors = [input_tensor]
        out = tf.strided_slice(tf.shape(input=input_tensor), begin, end, strides, begin_mask=parameters['begin_mask'], end_mask=parameters['end_mask'])
        return (tensors, [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            i = 10
            return i + 15
        'Build inputs for stride_slice test.'
        input_values = create_tensor_data(parameters['dtype'], parameters['input_shape'], min_value=-1, max_value=1)
        values = [input_values]
        return (values, sess.run(outputs, feed_dict=dict(zip(inputs, values))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs, expected_tf_failures=expected_tf_failures)

@register_make_test_function()
def make_shape_to_strided_slice_tests(options):
    if False:
        while True:
            i = 10
    'Make a set of tests to do shape op into strided_slice.'
    test_parameters = [{'dtype': [tf.float32], 'dynamic_input_shape': [[None, 2, 2, 5]], 'input_shape': [[12, 2, 2, 5]], 'strides': [[1]], 'begin': [[0]], 'end': [[1]], 'begin_mask': [0], 'end_mask': [0], 'fully_quantize': [False, True], 'dynamic_range_quantize': [False]}, {'dtype': [tf.float32], 'dynamic_input_shape': [[None, 2, 2, 5]], 'input_shape': [[12, 2, 2, 5]], 'strides': [[1]], 'begin': [[0]], 'end': [[1]], 'begin_mask': [0], 'end_mask': [0], 'fully_quantize': [False], 'dynamic_range_quantize': [True]}]
    _make_shape_to_strided_slice_test(options, test_parameters, expected_tf_failures=0)