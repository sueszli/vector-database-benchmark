"""Tests using module `tf.experimental.numpy` via an alias."""
import numpy as onp
import tensorflow as tf
np = tf.experimental.numpy

class PublicSymbolTest(tf.test.TestCase):

    def testSimple(self):
        if False:
            print('Hello World!')
        a = 0.1
        b = 0.2
        self.assertAllClose(onp.add(a, b), np.add(a, b))
if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    tf.test.main()