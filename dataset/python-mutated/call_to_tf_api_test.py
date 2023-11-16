"""Simple call to a TF API function.

The call will remain unchanged.
"""
import tensorflow.compat.v1 as tf
from tensorflow.python.autograph.tests import reference_test_base

def core_tf_call(x):
    if False:
        print('Hello World!')
    return x * tf.constant(2)

class ReferenceTest(reference_test_base.TestCase):

    def test_basic(self):
        if False:
            return 10
        self.assertFunctionMatchesEager(core_tf_call, 1)
        self.assertFunctionMatchesEager(core_tf_call, tf.constant(1))
if __name__ == '__main__':
    tf.test.main()