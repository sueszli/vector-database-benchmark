"""Test configs for segment_sum."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_segment_sum_tests(options):
    if False:
        print('Hello World!')
    'Make a set of tests to do segment_sum.'
    test_parameters = [{'data_shape': [[4, 4], [4], [4, 3, 2]], 'data_dtype': [tf.float32, tf.int32], 'segment_ids': [[0, 0, 1, 1], [0, 1, 2, 2], [0, 1, 2, 3], [0, 0, 0, 0]]}]

    def build_graph(parameters):
        if False:
            i = 10
            return i + 15
        'Build the segment_sum op testing graph.'
        data = tf.compat.v1.placeholder(dtype=parameters['data_dtype'], name='data', shape=parameters['data_shape'])
        segment_ids = tf.constant(parameters['segment_ids'], dtype=tf.int32)
        out = tf.math.segment_sum(data, segment_ids)
        return ([data], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            print('Hello World!')
        data = create_tensor_data(parameters['data_dtype'], parameters['data_shape'])
        return ([data], sess.run(outputs, feed_dict=dict(zip(inputs, [data]))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs, expected_tf_failures=0)