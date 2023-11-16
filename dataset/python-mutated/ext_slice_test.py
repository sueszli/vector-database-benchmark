"""Extended slice operations."""
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def basic_ext_slice(n):
    if False:
        while True:
            i = 10
    return (n[:, :], n[0, :], n[:, 0])

def basic_expand_dims(n):
    if False:
        return 10
    return n[:, tf.newaxis] - n[tf.newaxis, :]

def slice_of_application(n, x):
    if False:
        while True:
            i = 10
    return n(x)[:, tf.newaxis] - n(x)[tf.newaxis, :]

class ReferenceTest(reference_test_base.TestCase):

    def test_basic_ext_slice(self):
        if False:
            i = 10
            return i + 15
        self.assertFunctionMatchesEager(basic_ext_slice, tf.eye(3))

    def test_basic_expand_dims(self):
        if False:
            i = 10
            return i + 15
        self.assertFunctionMatchesEager(basic_expand_dims, tf.eye(3))

    def test_slice_of_application(self):
        if False:
            return 10
        self.assertFunctionMatchesEager(slice_of_application, lambda x: x, tf.eye(3))
if __name__ == '__main__':
    tf.test.main()