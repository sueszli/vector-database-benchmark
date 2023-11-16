"""Remote Predict Op client example.

Example client code which calls the Remote Predict Op directly.
"""
from __future__ import print_function
import tensorflow.compat.v1 as tf
from tensorflow_serving.experimental.tensorflow.ops.remote_predict.python.ops import remote_predict_ops
tf.app.flags.DEFINE_string('input_tensor_aliases', 'x', 'Aliases of input tensors')
tf.app.flags.DEFINE_float('input_value', 1.0, 'input value')
tf.app.flags.DEFINE_string('output_tensor_aliases', 'y', 'Aliases of output tensors')
tf.app.flags.DEFINE_string('target_address', 'localhost:8500', 'PredictionService address host:port')
tf.app.flags.DEFINE_string('model_name', 'half_plus_two', 'Name of the model')
tf.app.flags.DEFINE_integer('model_version', -1, 'Version of the model')
tf.app.flags.DEFINE_boolean('fail_op_on_rpc_error', True, 'Failure handling')
tf.app.flags.DEFINE_integer('rpc_deadline_millis', 30000, 'rpc deadline in milliseconds')
FLAGS = tf.app.flags.FLAGS

def main(unused_argv):
    if False:
        for i in range(10):
            print('nop')
    print('Call remote_predict_op')
    results = remote_predict_ops.run([FLAGS.input_tensor_aliases], [tf.constant(FLAGS.input_value, dtype=tf.float32)], [FLAGS.output_tensor_aliases], target_address=FLAGS.target_address, model_name=FLAGS.model_name, model_version=FLAGS.model_version, fail_op_on_rpc_error=FLAGS.fail_op_on_rpc_error, max_rpc_deadline_millis=FLAGS.rpc_deadline_millis, output_types=[tf.float32])
    print('Done remote_predict_op')
    print('Returned Result:', results.output_tensors[0].numpy())
if __name__ == '__main__':
    tf.app.run()