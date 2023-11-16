"""Loops which create variables, with or without shape invariants."""
import itertools
from absl.testing import parameterized
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def while_creates_var_static_shape(n):
    if False:
        print('Hello World!')
    i = 0
    while i < n:
        v = tf.zeros([1, 2, 3])
        i += 1
    return v

def while_creates_var_dynamic_shape(n):
    if False:
        print('Hello World!')
    i = 0
    while i < n:
        v = tf.zeros([1, tf.random.uniform((), i, i + 1, tf.int32), 2])
        i += 1
    return v

def while_creates_var_dynamic_rank(n):
    if False:
        return 10
    i = 0
    while i < n:
        v = tf.zeros(tf.range(tf.random.uniform((), i, i + 1, tf.int32)))
        i += 1
    return v

def while_creates_var_dynamic_shape_py_init_var(n):
    if False:
        while True:
            i = 10
    i = 0
    while i < n:
        v = tf.range(i)
        i += 1
    return v

def while_creates_nested_var_static_shape(n):
    if False:
        return 10
    i = 0
    while i < n:
        v = {'a': tf.zeros([1, 2, 3]), 'b': tf.ones([1, 2, 3])}
        i += 1
    return (v['a'], v['b'])

def while_creates_nested_var_dynamic_shape(n):
    if False:
        i = 10
        return i + 15
    i = 0
    while i < n:
        v = {'a': tf.zeros([1, tf.random.uniform((), i, i + 1, tf.int32)]), 'b': tf.ones([tf.random.uniform((), i, i + 1, tf.int32), 2])}
        i += 1
    return (v['a'], v['b'])

def while_creates_nested_var_dynamic_rank(n):
    if False:
        while True:
            i = 10
    i = 0
    while i < n:
        v = {'a': tf.ones(tf.range(tf.random.uniform((), i, i + 1, tf.int32))), 'b': tf.ones([1, 2, 3])}
        i += 1
    return (v['a'], v['b'])

class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):

    @parameterized.parameters(while_creates_var_static_shape, while_creates_var_dynamic_shape, while_creates_var_dynamic_rank, while_creates_var_dynamic_shape_py_init_var, while_creates_nested_var_static_shape, while_creates_nested_var_dynamic_shape, while_creates_nested_var_dynamic_rank)
    def test_while_creates_var_illegal_tf(self, target):
        if False:
            print('Hello World!')
        with self.assertRaises(tf.errors.InvalidArgumentError):
            tf.function(target)(tf.constant(0))

    @parameterized.parameters(while_creates_var_static_shape, while_creates_var_dynamic_shape, while_creates_var_dynamic_rank, while_creates_var_dynamic_shape_py_init_var, while_creates_nested_var_static_shape, while_creates_nested_var_dynamic_shape, while_creates_nested_var_dynamic_rank)
    def test_while_creates_var_illegal_py(self, target):
        if False:
            return 10
        with self.assertRaises(UnboundLocalError):
            tf.function(target)(0)

    @parameterized.parameters(*itertools.product((1, 2), (int, tf.constant), (while_creates_var_static_shape, while_creates_var_dynamic_shape, while_creates_var_dynamic_rank, while_creates_var_dynamic_shape_py_init_var, while_creates_nested_var_static_shape, while_creates_nested_var_dynamic_shape, while_creates_nested_var_dynamic_rank)))
    def test_while_creates_var(self, n, type_, target):
        if False:
            i = 10
            return i + 15
        n = type_(n)
        self.assertFunctionMatchesEager(target, n)
if __name__ == '__main__':
    tf.test.main()