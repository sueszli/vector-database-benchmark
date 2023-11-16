"""Loops with type changing variables."""
import re
from absl.testing import parameterized
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def while_with_variable_py_type():
    if False:
        print('Hello World!')
    n = tf.constant(0, dtype=tf.int32)
    c = True
    while c:
        c = tf.constant(True)
    return n

def while_with_variable_dtype():
    if False:
        return 10
    n = tf.constant(0, dtype=tf.int32)
    while tf.constant(True):
        n = tf.constant(0, dtype=tf.float32)
    return n

def while_with_variable_dtype_and_early_stopping():
    if False:
        while True:
            i = 10
    n = tf.constant(0, dtype=tf.int32)
    while tf.constant(True):
        n = tf.constant(0, dtype=tf.float32)
        break
    return n

def for_with_variable_dtype(l):
    if False:
        for i in range(10):
            print('nop')
    n = tf.constant(0, dtype=tf.int32)
    for _ in l:
        n = tf.constant(0, dtype=tf.float32)
    return n

def for_with_variable_dtype_and_early_stopping(l):
    if False:
        for i in range(10):
            print('nop')
    n = tf.constant(0, dtype=tf.int32)
    for _ in l:
        n = tf.constant(0, dtype=tf.float32)
        break
    return n

def while_with_variable_shape():
    if False:
        i = 10
        return i + 15
    t = tf.constant([1])
    while tf.constant(True):
        t = tf.constant([1, 1])
    return t

def for_with_variable_shape(l):
    if False:
        return 10
    t = tf.constant([1])
    for _ in l:
        t = tf.constant([1, 1])
    return t

def while_with_shape_erasure():
    if False:
        i = 10
        return i + 15
    t = tf.constant([1])
    while tf.constant(True):
        t = tf.range(tf.random.uniform((), 2, 3, dtype=tf.int32))
    return t

def for_with_shape_erasure(l):
    if False:
        print('Hello World!')
    t = tf.constant([1])
    for _ in l:
        t = tf.range(tf.random.uniform((), 2, 3, dtype=tf.int32))
    return t

def while_with_shape_invariant_violation():
    if False:
        for i in range(10):
            print('nop')
    t = tf.constant([1])
    while tf.constant(True):
        tf.autograph.experimental.set_loop_options(shape_invariants=((t, tf.TensorShape([1])),))
        t = tf.range(tf.random.uniform((), 2, 3, dtype=tf.int32))
    return t

def for_with_shape_invariant_violation(l):
    if False:
        print('Hello World!')
    t = tf.constant([1])
    for _ in l:
        tf.autograph.experimental.set_loop_options(shape_invariants=((t, tf.TensorShape([1])),))
        t = tf.range(tf.random.uniform((), 2, 3, dtype=tf.int32))
    return t

def while_with_variable_structure():
    if False:
        i = 10
        return i + 15
    s = {'a': tf.constant(0)}
    while tf.constant(True):
        s = tf.constant(7.0)
    return s

def for_with_variable_structure(l):
    if False:
        return 10
    s = [tf.constant(0)]
    for _ in l:
        s = s + [tf.constant(0)]
    return s

def _tf_range(l):
    if False:
        while True:
            i = 10
    return tf.range(len(l))

def _dataset(l):
    if False:
        for i in range(10):
            print('nop')
    return tf.data.Dataset.from_tensor_slices(l)

def _dataset_iterator(l):
    if False:
        while True:
            i = 10
    return iter(tf.data.Dataset.from_tensor_slices(l))

def _distributed_dataset(l):
    if False:
        i = 10
        return i + 15
    ds = tf.data.Dataset.from_tensor_slices([l] * 2)
    return tf.distribute.MirroredStrategy().experimental_distribute_dataset(ds)

class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):

    def test_while_with_variable_py_type(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(NotImplementedError, re.compile('.*condition of while loop started as non\\-Tensor, then changed to Tensor.*', re.DOTALL)):
            tf.function(while_with_variable_py_type)()

    def test_while_with_variable_dtype(self):
        if False:
            return 10
        with self.assertRaisesRegex(TypeError, "'n' has dtype int32 before the loop, but dtype float32 after"):
            tf.function(while_with_variable_dtype)()

    def test_while_with_variable_dtype_and_early_stopping(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(TypeError, "'n' has dtype int32 before the loop, but dtype float32 after"):
            tf.function(while_with_variable_dtype_and_early_stopping)()

    @parameterized.parameters((tf.constant,), (_tf_range,), (_dataset,), (_dataset_iterator,), (_distributed_dataset,))
    def test_for_with_variable_dtype(self, type_):
        if False:
            i = 10
            return i + 15
        l = type_([1, 2, 3])
        with self.assertRaisesRegex(TypeError, "'n' has dtype int32 before the loop, but dtype float32 after"):
            tf.function(for_with_variable_dtype)(l)

    @parameterized.parameters((tf.constant,), (_tf_range,), (_dataset,), (_dataset_iterator,))
    def test_for_with_variable_dtype_and_early_stopping(self, type_):
        if False:
            while True:
                i = 10
        l = type_([1, 2, 3])
        with self.assertRaisesRegex(TypeError, "'n' has dtype int32 before the loop, but dtype float32 after"):
            tf.function(for_with_variable_dtype_and_early_stopping)(l)

    def test_while_with_variable_shape(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, "'t' has shape \\(1,\\) before the loop, but shape \\(2,\\) after"):
            tf.function(while_with_variable_shape)()

    @parameterized.parameters((tf.constant,), (_tf_range,), (_dataset_iterator,), (_distributed_dataset,))
    def test_for_with_variable_shape(self, type_):
        if False:
            for i in range(10):
                print('nop')
        l = type_([1, 2, 3])
        with self.assertRaisesRegex(ValueError, "'t' has shape \\(1,\\) before the loop, but shape \\(2,\\) after"):
            tf.function(for_with_variable_shape)(l)

    def test_while_with_shape_erasure(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(ValueError, "'t' has shape \\(1,\\) before the loop, but shape \\(None,\\) after"):
            tf.function(while_with_shape_erasure)()

    @parameterized.parameters((tf.constant,), (_tf_range,), (_dataset_iterator,), (_distributed_dataset,))
    def test_for_with_shape_erasure(self, type_):
        if False:
            while True:
                i = 10
        l = type_([1, 2, 3])
        with self.assertRaisesRegex(ValueError, "'t' has shape \\(1,\\) before the loop, but shape \\(None,\\) after"):
            tf.function(for_with_shape_erasure)(l)

    def test_while_with_shape_invariant_violation(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, "'t' has shape \\(None,\\) after one iteration, which does not conform"):
            tf.function(while_with_shape_invariant_violation)()

    @parameterized.parameters((tf.constant,), (_tf_range,), (_dataset_iterator,), (_distributed_dataset,))
    def test_for_with_shape_invariant_violation(self, type_):
        if False:
            i = 10
            return i + 15
        l = type_([1, 2, 3])
        with self.assertRaisesRegex(ValueError, "'t' has shape \\(None,\\) after one iteration, which does not conform"):
            tf.function(for_with_shape_invariant_violation)(l)

    def test_while_with_variable_structure(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(TypeError, "'s' does not have the same nested structure"):
            tf.function(while_with_variable_structure)()

    @parameterized.parameters((tf.constant,), (_tf_range,), (_dataset,), (_dataset_iterator,), (_distributed_dataset,))
    def test_for_with_variable_structure(self, type_):
        if False:
            print('Hello World!')
        l = type_([1, 2, 3])
        with self.assertRaisesRegex(TypeError, "'s' does not have the same nested structure"):
            tf.function(for_with_variable_structure)(l)
if __name__ == '__main__':
    tf.test.main()