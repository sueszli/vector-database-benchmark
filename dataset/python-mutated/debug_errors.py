"""Example of debugging TensorFlow runtime errors using tfdbg."""
import argparse
import sys
import tempfile
import numpy as np
import tensorflow
from tensorflow.python import debug as tf_debug
tf = tensorflow.compat.v1

def main(_):
    if False:
        while True:
            i = 10
    sess = tf.Session()
    ph_float = tf.placeholder(tf.float32, name='ph_float')
    x = tf.transpose(ph_float, name='x')
    v = tf.Variable(np.array([[-2.0], [-3.0], [6.0]], dtype=np.float32), name='v')
    m = tf.constant(np.array([[0.0, 1.0, 2.0], [-4.0, -1.0, 0.0]]), dtype=tf.float32, name='m')
    y = tf.matmul(m, x, name='y')
    z = tf.matmul(m, v, name='z')
    if FLAGS.debug:
        if FLAGS.use_random_config_path:
            (_, config_file_path) = tempfile.mkstemp('.tfdbg_config')
        else:
            config_file_path = None
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type=FLAGS.ui_type, config_file_path=config_file_path)
    if FLAGS.error == 'shape_mismatch':
        print(sess.run(y, feed_dict={ph_float: np.array([[0.0], [1.0], [2.0]])}))
    elif FLAGS.error == 'uninitialized_variable':
        print(sess.run(z))
    elif FLAGS.error == 'no_error':
        print(sess.run(y, feed_dict={ph_float: np.array([[0.0, 1.0, 2.0]])}))
    else:
        raise ValueError('Unrecognized error type: ' + FLAGS.error)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--error', type=str, default='shape_mismatch', help='      Type of the error to generate (shape_mismatch | uninitialized_variable |\n      no_error).      ')
    parser.add_argument('--ui_type', type=str, default='readline', help='Command-line user interface type (only readline is supported)')
    parser.add_argument('--debug', type='bool', nargs='?', const=True, default=False, help='Use debugger to track down bad values during training')
    parser.add_argument('--use_random_config_path', type='bool', nargs='?', const=True, default=False, help='If set, set config file path to a random file in the temporary\n      directory.')
    (FLAGS, unparsed) = parser.parse_known_args()
    with tf.Graph().as_default():
        tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)