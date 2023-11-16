"""Loops with type changing variables."""
import collections
import itertools
from absl.testing import parameterized
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def while_with_variable_shape_growing_vector(n):
    if False:
        while True:
            i = 10
    v = tf.constant([0, 0])
    i = 0
    while i < n:
        tf.autograph.experimental.set_loop_options(shape_invariants=[(v, tf.TensorShape([None]))])
        v = tf.concat((v, [i]), 0)
        i += 1
    return v

def for_with_variable_shape_growing_vector(l):
    if False:
        print('Hello World!')
    v = tf.constant([0, 0])
    for i in l:
        tf.autograph.experimental.set_loop_options(shape_invariants=[(v, tf.TensorShape([None]))])
        v = tf.concat((v, [i]), 0)
    return v

def while_with_variable_shape_growing_matrix_rows(n):
    if False:
        print('Hello World!')
    m = tf.constant([[0]])
    i = 0
    while i < n:
        tf.autograph.experimental.set_loop_options(shape_invariants=[(m, tf.TensorShape([None, 1]))])
        m = tf.concat((m, [[i]]), 0)
        i += 1
    return m

def for_with_variable_shape_growing_matrix_rows(l):
    if False:
        print('Hello World!')
    m = tf.constant([[0]])
    for i in l:
        tf.autograph.experimental.set_loop_options(shape_invariants=[(m, tf.TensorShape([None, 1]))])
        m = tf.concat((m, [[i]]), 0)
    return m

def while_with_variable_shape_growing_matrix_cols(n):
    if False:
        return 10
    m = tf.constant([[0, 0]])
    i = 0
    while i < n:
        tf.autograph.experimental.set_loop_options(shape_invariants=[(m, tf.TensorShape([1, None]))])
        m = tf.concat((m, [[i]]), 1)
        i += 1
    return m

def for_with_variable_shape_growing_matrix_cols(l):
    if False:
        return 10
    m = tf.constant([[0, 0]])
    for i in l:
        tf.autograph.experimental.set_loop_options(shape_invariants=[(m, tf.TensorShape([1, None]))])
        m = tf.concat((m, [[i]]), 1)
    return m

def while_with_variable_shape_growing_matrix(n):
    if False:
        return 10
    m = tf.constant([[0, 0], [0, 0]])
    i = 0
    while i < n:
        tf.autograph.experimental.set_loop_options(shape_invariants=[(m, tf.TensorShape(None))])
        m = tf.pad(m, [[1, 1], [1, 1]], constant_values=i)
        i += 1
    return m

def for_with_variable_shape_growing_matrix(l):
    if False:
        i = 10
        return i + 15
    m = tf.constant([[0, 0], [0, 0]])
    for i in l:
        tf.autograph.experimental.set_loop_options(shape_invariants=[(m, tf.TensorShape(None))])
        m = tf.pad(m, [[1, 1], [1, 1]], constant_values=i)
    return m

def while_with_variable_shape_inside_if(n):
    if False:
        return 10
    v = tf.constant([0, 0])
    i = 0
    if n > 1:
        while i < n:
            tf.autograph.experimental.set_loop_options(shape_invariants=[(v, tf.TensorShape([None]))])
            v = tf.concat((v, [i]), 0)
            i += 1
    else:
        v = tf.constant([1, 2, 3])
    return v

def for_with_variable_shape_inside_if(n):
    if False:
        while True:
            i = 10
    v = tf.constant([0, 0])
    if n > 1:
        for i in range(n):
            tf.autograph.experimental.set_loop_options(shape_invariants=[(v, tf.TensorShape([None]))])
            v = tf.concat((v, [i]), 0)
            i += 1
    else:
        v = tf.constant([1, 2, 3])
    return v

def for_with_nested_variable_shape_inside_if(n):
    if False:
        return 10
    Test = collections.namedtuple('Test', ['var'])
    t = Test(var=tf.constant([0]))
    v = tf.constant([0, 0])
    if n > 1:
        for i in range(n):
            tf.autograph.experimental.set_loop_options(shape_invariants=[(v, tf.TensorShape([None]))])
            v = tf.concat((v, [i]), 0)
            t = Test(var=t.var + 1)
            i += 1
    else:
        v = tf.constant([1, 2, 3])
        t = Test(var=tf.constant([3]))
    return v

def while_with_variable_shape_and_break(n):
    if False:
        i = 10
        return i + 15
    v = tf.constant([0, 0])
    i = 0
    if n > 1:
        while i < n:
            tf.autograph.experimental.set_loop_options(shape_invariants=[(v, tf.TensorShape([None]))])
            v = tf.concat((v, [i]), 0)
            i += 1
            if i > 3:
                break
    else:
        v = tf.constant([1, 2, 3])
    return v

def for_with_variable_shape_and_break(n):
    if False:
        for i in range(10):
            print('nop')
    v = tf.constant([0, 0])
    if n > 1:
        for i in range(n):
            tf.autograph.experimental.set_loop_options(shape_invariants=[(v, tf.TensorShape([None]))])
            v = tf.concat((v, [i]), 0)
            i += 1
            if i > 3:
                break
    else:
        v = tf.constant([1, 2, 3])
    return v

def while_with_composite_tensor_shape_invariant(n):
    if False:
        print('Hello World!')
    v = tf.SparseTensor(indices=[[0, 0], [1, 1]], values=[1, 2], dense_shape=[3, 3])
    i = 0
    while i < n:
        tf.autograph.experimental.set_loop_options(shape_invariants=[(v, tf.TensorShape(None))])
        v = tf.sparse.expand_dims(v)
        i += 1
    return v

def for_with_composite_tensor_shape_invariant(l):
    if False:
        print('Hello World!')
    v = tf.SparseTensor(indices=[[0, 0], [1, 1]], values=[1, 2], dense_shape=[3, 3])
    for _ in l:
        tf.autograph.experimental.set_loop_options(shape_invariants=[(v, tf.TensorShape(None))])
        v = tf.sparse.expand_dims(v)
    return v

def _int_dataset_range(n):
    if False:
        while True:
            i = 10
    return tf.data.Dataset.range(n).map(lambda x: tf.cast(x, tf.int32))

class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, tf.constant)))
    def test_while_with_variable_shape_growing_vector(self, n, type_):
        if False:
            return 10
        n = type_(n)
        self.assertFunctionMatchesEager(while_with_variable_shape_growing_vector, n)

    @parameterized.parameters(*itertools.product((0, 1, 2), (range, tf.range, tf.data.Dataset.range)))
    def test_for_with_variable_shape_growing_vector(self, n, list_type):
        if False:
            return 10
        l = list_type(n)
        self.assertFunctionMatchesEager(for_with_variable_shape_growing_vector, l)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, tf.constant)))
    def test_while_with_variable_shape_growing_matrix_rows(self, n, type_):
        if False:
            i = 10
            return i + 15
        n = type_(n)
        self.assertFunctionMatchesEager(while_with_variable_shape_growing_matrix_rows, n)

    @parameterized.parameters(*itertools.product((0, 1, 2), (range, tf.range, _int_dataset_range)))
    def test_for_with_variable_shape_growing_matrix_rows(self, l, type_):
        if False:
            print('Hello World!')
        l = type_(l)
        self.assertFunctionMatchesEager(for_with_variable_shape_growing_matrix_rows, l)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, tf.constant)))
    def test_while_with_variable_shape_growing_matrix_cols(self, n, type_):
        if False:
            return 10
        n = type_(n)
        self.assertFunctionMatchesEager(while_with_variable_shape_growing_matrix_cols, n)

    @parameterized.parameters(*itertools.product((0, 1, 2), (range, tf.range, tf.data.Dataset.range)))
    def test_for_with_variable_shape_growing_matrix_cols(self, l, type_):
        if False:
            i = 10
            return i + 15
        l = type_(l)
        self.assertFunctionMatchesEager(for_with_variable_shape_growing_matrix_cols, l)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, tf.constant)))
    def test_while_with_variable_shape_growing_matrix(self, n, type_):
        if False:
            while True:
                i = 10
        n = type_(n)
        self.assertFunctionMatchesEager(while_with_variable_shape_growing_matrix, n)

    @parameterized.parameters(*itertools.product((0, 1, 2), (range, tf.range, _int_dataset_range)))
    def test_for_with_variable_shape_growing_matrix(self, n, type_):
        if False:
            i = 10
            return i + 15
        l = type_(n)
        self.assertFunctionMatchesEager(for_with_variable_shape_growing_matrix, l)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, tf.constant)))
    def test_while_with_variable_shape_inside_if(self, n, type_):
        if False:
            while True:
                i = 10
        n = type_(n)
        self.assertFunctionMatchesEager(while_with_variable_shape_inside_if, n)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, tf.constant)))
    def test_for_with_variable_shape_inside_if(self, n, type_):
        if False:
            for i in range(10):
                print('nop')
        n = type_(n)
        self.assertFunctionMatchesEager(for_with_variable_shape_inside_if, n)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, tf.constant)))
    def test_for_with_nested_variable_shape_inside_if(self, n, type_):
        if False:
            return 10
        n = type_(n)
        self.assertFunctionMatchesEager(for_with_nested_variable_shape_inside_if, n)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, tf.constant)))
    def test_while_with_variable_shape_and_break(self, n, type_):
        if False:
            return 10
        n = type_(n)
        self.assertFunctionMatchesEager(while_with_variable_shape_and_break, n)

    @parameterized.parameters(*itertools.product((0, 1, 2, 5), (int, tf.constant)))
    def test_for_with_variable_shape_and_break(self, n, type_):
        if False:
            i = 10
            return i + 15
        n = type_(n)
        self.assertFunctionMatchesEager(for_with_variable_shape_and_break, n)

    @parameterized.parameters(*itertools.product((0, 1, 2, 5), (int, tf.constant)))
    def test_while_with_composite_tensor_shape_invariant(self, n, type_):
        if False:
            for i in range(10):
                print('nop')
        n = type_(n)
        self.assertFunctionMatchesEager(while_with_composite_tensor_shape_invariant, n)

    @parameterized.parameters(*itertools.product((0, 1, 2), (range, tf.range, _int_dataset_range)))
    def test_for_with_composite_tensor_shape_invariant(self, n, type_):
        if False:
            i = 10
            return i + 15
        l = type_(n)
        self.assertFunctionMatchesEager(for_with_composite_tensor_shape_invariant, l)
if __name__ == '__main__':
    tf.test.main()