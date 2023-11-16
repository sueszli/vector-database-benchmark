"""Tests that an error is raised when numpy functions are called."""
import tensorflow.compat.v2 as tf
from tensorflow.python.ops.numpy_ops import np_config

class ConfigTest(tf.test.TestCase):

    def testMethods(self):
        if False:
            for i in range(10):
                print('nop')
        a = tf.constant(1.0)
        for name in {'T', 'astype', 'ravel', 'transpose', 'reshape', 'clip', 'size', 'tolist'}:
            with self.assertRaisesRegex(AttributeError, 'enable_numpy_behavior'):
                getattr(a, name)
        np_config.enable_numpy_behavior()
        for name in {'T', 'astype', 'ravel', 'transpose', 'reshape', 'clip', 'size', 'tolist'}:
            _ = getattr(a, name)
if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    tf.test.main()