"""Demo of the tfdbg readline UI: A TF network computing Fibonacci sequence."""
import argparse
import sys
import numpy as np
import tensorflow
from tensorflow.python import debug as tf_debug
tf = tensorflow.compat.v1
FLAGS = None

def main(_):
    if False:
        i = 10
        return i + 15
    sess = tf.Session()
    n0 = tf.Variable(np.ones([FLAGS.tensor_size] * 2), dtype=tf.int32, name='node_00')
    n1 = tf.Variable(np.ones([FLAGS.tensor_size] * 2), dtype=tf.int32, name='node_01')
    for i in range(2, FLAGS.length):
        (n0, n1) = (n1, tf.add(n0, n1, name='node_%.2d' % i))
    sess.run(tf.global_variables_initializer())
    if FLAGS.debug and FLAGS.tensorboard_debug_address:
        raise ValueError('The --debug and --tensorboard_debug_address flags are mutually exclusive.')
    if FLAGS.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        def has_negative(_, tensor):
            if False:
                i = 10
                return i + 15
            return np.any(tensor < 0)
        sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)
        sess.add_tensor_filter('has_negative', has_negative)
    elif FLAGS.tensorboard_debug_address:
        sess = tf_debug.TensorBoardDebugWrapperSession(sess, FLAGS.tensorboard_debug_address)
    print('Fibonacci number at position %d:\n%s' % (FLAGS.length, sess.run(n1)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--tensor_size', type=int, default=1, help='      Size of tensor. E.g., if the value is 30, the tensors will have shape\n      [30, 30].      ')
    parser.add_argument('--length', type=int, default=20, help='Length of the fibonacci sequence to compute.')
    parser.add_argument('--ui_type', type=str, default='readline', help='Command-line user interface type (only readline is supported)')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Use TensorFlow Debugger (tfdbg). Mutually exclusive with the --tensorboard_debug_address flag.')
    parser.add_argument('--tensorboard_debug_address', type=str, default=None, help='Connect to the TensorBoard Debugger Plugin backend specified by the gRPC address (e.g., localhost:1234). Mutually exclusive with the --debug flag.')
    (FLAGS, unparsed) = parser.parse_known_args()
    with tf.Graph().as_default():
        tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)