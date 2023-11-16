"""Loop control statements (e.g. break, return) in illegal patterns.

Meant to verify that:
  * break/return on a dynamic condition raises error inside Python loop
"""
import itertools
import re
from absl.testing import parameterized
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def tf_break_in_py_for(l):
    if False:
        return 10
    s = 0
    for c in l:
        if tf.greater(c % 2, 0):
            break
        s += c
    return s

def tf_return_in_py_for(l):
    if False:
        while True:
            i = 10
    s = 0
    for c in l:
        if tf.greater(c % 2, 0):
            return s
        else:
            return s
        s += c
    return s

def tf_break_in_py_while(x):
    if False:
        print('Hello World!')
    s = 0
    while x > 0:
        x -= 1
        if tf.greater(x % 2, 0):
            break
        s += x
    return s

def tf_return_in_py_while(x):
    if False:
        while True:
            i = 10
    s = 0
    while x > 0:
        x -= 1
        if tf.greater(x % 2, 0):
            return s
        else:
            return s
        s += x
    return s

class LoopControlFlowIllegalCasesTest(reference_test_base.TestCase, parameterized.TestCase):

    @parameterized.parameters(*itertools.product(([1], [1, 2], [1, 2, 3]), (tf_break_in_py_for, tf_return_in_py_for)))
    def test_tf_control_flow_in_py_for(self, l, target):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(NotImplementedError, 'not supported in Python for'):
            tf.function(target)(l)

    @parameterized.parameters(*itertools.product((1, 2, 3), (tf_break_in_py_while, tf_return_in_py_while)))
    def test_tf_control_flow_in_py_while(self, n, target):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(NotImplementedError, re.compile('.*condition of while loop started as non\\-Tensor, then changed to Tensor.*', re.DOTALL)):
            tf.function(target)(n)
if __name__ == '__main__':
    tf.test.main()