"""Demo of the tfdbg curses UI: A TF v2 network computing Fibonacci sequence."""
import argparse
import sys
from absl import app
import numpy as np
import tensorflow.compat.v2 as tf
FLAGS = None
tf.compat.v1.enable_v2_behavior()

def main(_):
    if False:
        i = 10
        return i + 15
    if FLAGS.debug and FLAGS.tensorboard_debug_address:
        raise ValueError('The --debug and --tensorboard_debug_address flags are mutually exclusive.')
    if FLAGS.debug:
        raise NotImplementedError('tfdbg v2 support for debug_fibonacci is not implemented yet')
    elif FLAGS.tensorboard_debug_address:
        raise NotImplementedError('Tensorboard Debugger Plugin support for debug_fibonacci_v2 is not implemented yet')
    n0 = tf.constant(np.ones([FLAGS.tensor_size] * 2), dtype=tf.int32)
    n1 = tf.constant(np.ones([FLAGS.tensor_size] * 2), dtype=tf.int32)
    for _ in range(2, FLAGS.length):
        (n0, n1) = (n1, tf.add(n0, n1))
    print('Fibonacci number at position %d:\n%s' % (FLAGS.length, n1.numpy()))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--tensor_size', type=int, default=1, help='      Size of tensor. E.g., if the value is 30, the tensors will have shape\n      [30, 30].      ')
    parser.add_argument('--length', type=int, default=20, help='Length of the fibonacci sequence to compute.')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Use TensorFlow Debugger (tfdbg). Mutually exclusive with the --tensorboard_debug_address flag.')
    parser.add_argument('--tensorboard_debug_address', type=str, default=None, help='Connect to the TensorBoard Debugger Plugin backend specified by the gRPC address (e.g., localhost:1234). Mutually exclusive with the --debug flag.')
    (FLAGS, unparsed) = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)