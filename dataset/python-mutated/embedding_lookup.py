"""Test configs for embedding_lookup."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

@register_make_test_function()
def make_embedding_lookup_tests(options):
    if False:
        while True:
            i = 10
    'Make a set of tests to do gather.'
    test_parameters = [{'params_dtype': [tf.float32], 'params_shape': [[10], [10, 10]], 'ids_dtype': [tf.int32], 'ids_shape': [[3], [5]]}]

    def build_graph(parameters):
        if False:
            i = 10
            return i + 15
        'Build the gather op testing graph.'
        params = tf.compat.v1.placeholder(dtype=parameters['params_dtype'], name='params', shape=parameters['params_shape'])
        ids = tf.compat.v1.placeholder(dtype=parameters['ids_dtype'], name='ids', shape=parameters['ids_shape'])
        out = tf.nn.embedding_lookup(params=params, ids=ids)
        return ([params, ids], [out])

    def build_inputs(parameters, sess, inputs, outputs):
        if False:
            for i in range(10):
                print('nop')
        params = create_tensor_data(parameters['params_dtype'], parameters['params_shape'])
        ids = create_tensor_data(parameters['ids_dtype'], parameters['ids_shape'], 0, parameters['params_shape'][0] - 1)
        return ([params, ids], sess.run(outputs, feed_dict=dict(zip(inputs, [params, ids]))))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)