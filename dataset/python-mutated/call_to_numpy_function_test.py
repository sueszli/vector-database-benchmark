"""Simple call to a whitelisted Numpy function.

The call should be wrapped in py_func.
"""
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.autograph.tests import reference_test_base

def f():
    if False:
        print('Hello World!')
    np.random.seed(1)
    return 2 * np.random.binomial(1, 0.5, size=(10,)) - 1

class ReferenceTest(reference_test_base.TestCase):

    def test_basic(self):
        if False:
            print('Hello World!')
        self.assertFunctionMatchesEager(f)
if __name__ == '__main__':
    tf.test.main()