"""Simple call to a print function preceding other computations.

The call may be wrapped inside a py_func, but tf.Print should be used if
possible. The subsequent computations will be gated by the print function
execution.
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def lone_print(x):
    if False:
        i = 10
        return i + 15
    print(x)

def print_multiple_values(x):
    if False:
        return 10
    print('x is', x)

def multiple_prints(x, y):
    if False:
        while True:
            i = 10
    tf.print('x is', x)
    tf.print('y is', y)

def print_with_nontf_values(x):
    if False:
        return 10
    print('x is', x, {'foo': 'bar'})

def print_in_cond(x):
    if False:
        for i in range(10):
            print('nop')
    if x == 0:
        print(x)

def tf_print(x):
    if False:
        i = 10
        return i + 15
    tf.print(x)

class ReferenceTest(reference_test_base.TestCase):

    def setUp(self):
        if False:
            return 10
        super(ReferenceTest, self).setUp()
        self.autograph_opts = tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS

    def test_lone_print(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFunctionMatchesEager(lone_print, 1)
        self.assertFunctionMatchesEager(lone_print, np.array([1, 2, 3]))

    def test_print_multiple_values(self):
        if False:
            i = 10
            return i + 15
        self.assertFunctionMatchesEager(print_multiple_values, 1)
        self.assertFunctionMatchesEager(print_multiple_values, np.array([1, 2, 3]))

    def test_multiple_prints(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFunctionMatchesEager(multiple_prints, 1, 2)
        self.assertFunctionMatchesEager(multiple_prints, np.array([1, 2, 3]), 4)

    def test_print_with_nontf_values(self):
        if False:
            print('Hello World!')
        self.assertFunctionMatchesEager(print_with_nontf_values, 1)
        self.assertFunctionMatchesEager(print_with_nontf_values, np.array([1, 2, 3]))

    def test_print_in_cond(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFunctionMatchesEager(print_in_cond, 0)
        self.assertFunctionMatchesEager(print_in_cond, 1)

    def test_tf_print(self):
        if False:
            print('Hello World!')
        self.assertFunctionMatchesEager(tf_print, 0)
if __name__ == '__main__':
    tf.test.main()