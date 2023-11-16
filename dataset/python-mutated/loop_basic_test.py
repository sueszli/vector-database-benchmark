"""Basic loops iterating over various types."""
import functools
import itertools
from absl.testing import parameterized
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def while_no_vars(n, v):
    if False:
        while True:
            i = 10
    v.assign(0)
    while v.read_value() < n:
        v.assign_add(1)
    return v.read_value()

def for_no_vars(l, v):
    if False:
        while True:
            i = 10
    v.assign(0)
    for _ in l:
        v.assign_add(1)
    return v.read_value()

def while_one_var(n):
    if False:
        print('Hello World!')
    i = 0
    while i < n:
        i = i * 10 + 1
    return i

def for_one_var(l):
    if False:
        i = 10
        return i + 15
    i = 0
    for i in l:
        pass
    return i

def while_two_vars(n):
    if False:
        while True:
            i = 10
    s = 0
    i = 0
    while i < n:
        s = s * 10 + i
        i += 1
    return s

def for_two_vars(l):
    if False:
        i = 10
        return i + 15
    s = 0
    for i in l:
        s = s * 10 + i
    return s

def while_else(x, y):
    if False:
        for i in range(10):
            print('nop')
    s = 0
    while x > 0:
        x -= 1
        if x > y:
            break
        s += x
    else:
        s = -100
    return s

def for_else(l1, l2):
    if False:
        print('Hello World!')
    s = 0
    for c in l1:
        if c in l2:
            break
        s = s * 10 + c
    else:
        s = -1000
    return s

def while_creates_var(n):
    if False:
        print('Hello World!')
    i = 0
    while i < n:
        i = n + 1
        j = i + 1
    return (i, j)

def for_creates_var(l):
    if False:
        while True:
            i = 10
    for i in l:
        pass
    return i

def successive_while_loops(n1, n2):
    if False:
        print('Hello World!')
    s = 0
    i = 0
    while i < n1:
        s = s * 10 + i
        i += 1
    i = 0
    while i < n2:
        s = s * 10 + i
        i += 1
    return s

def successive_for_loops(l1, l2):
    if False:
        i = 10
        return i + 15
    s = 0
    for i in l1:
        s = s * 10 + i
    for i in l2:
        s = s * 10 + i
    return s

def nested_while_loops(n1, n2):
    if False:
        while True:
            i = 10
    i = 0
    l = tf.TensorArray(tf.int32, size=0, dynamic_size=True, element_shape=())
    while i < n1:
        j = 0
        s = 0
        while j < n2:
            s = s * 10 + i * j
            j += 1
        l = l.write(i, s)
        i += 1
    return l.stack()

def nested_for_loops(m):
    if False:
        for i in range(10):
            print('nop')
    l = tf.TensorArray(tf.int32, size=0, dynamic_size=True, element_shape=())
    for i in m:
        s = 0
        for j in i:
            s = s * 10 + j
        l = l.write(l.size(), s)
    return l.stack()

def _int_tensor(x):
    if False:
        while True:
            i = 10
    return tf.constant(x, dtype=tf.int32)

def _int_dataset(l):
    if False:
        while True:
            i = 10
    return tf.data.Dataset.from_tensor_slices(tf.constant(l, dtype=tf.int32))

def double_product(l1, l2):
    if False:
        for i in range(10):
            print('nop')
    for i in l1:
        for j in l2:
            for k in l1:
                for l in l2:
                    yield (i, j, k, l)

class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, _int_tensor)))
    def test_while_no_vars(self, n, type_):
        if False:
            print('Hello World!')
        n = type_(n)
        self.assertFunctionMatchesEager(while_no_vars, n, tf.Variable(0))

    @parameterized.parameters(*itertools.product(([], [1], [1, 2]), (list, _int_tensor), (True, False)))
    def test_for_no_vars(self, l, type_, xla):
        if False:
            for i in range(10):
                print('nop')
        if type_ is _int_tensor and xla and (not l):
            self.skipTest('Empty loops not supported in XLA')
        l = type_(l)
        self.assertFunctionMatchesEager(for_no_vars, l, tf.Variable(0), xla=xla)

    @parameterized.parameters(([],), ([1],), ([1, 2],))
    def test_for_no_vars_ds_iterator(self, l):
        if False:
            return 10
        inputs_ = lambda : (iter(_int_dataset(l)), tf.Variable(0))
        self.assertFunctionMatchesEagerStatefulInput(for_no_vars, inputs_)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, _int_tensor)))
    def test_while_one_var(self, n, type_):
        if False:
            i = 10
            return i + 15
        n = type_(n)
        self.assertFunctionMatchesEager(while_one_var, n)

    @parameterized.parameters(*itertools.product(([], [1], [1, 2]), (list, _int_tensor, _int_dataset), (True, False)))
    def test_for_one_var(self, l, type_, xla):
        if False:
            while True:
                i = 10
        if type_ is _int_dataset and xla:
            self.skipTest('Datsets not supported in XLA')
        if type_ is _int_tensor and xla and (not l):
            self.skipTest('Empty loops not supported in XLA')
        l = type_(l)
        self.assertFunctionMatchesEager(for_one_var, l, xla=xla)

    @parameterized.parameters(([],), ([1],), ([1, 2],))
    def test_for_one_var_ds_iterator(self, l):
        if False:
            print('Hello World!')
        inputs_ = lambda : (iter(_int_dataset(l)), tf.Variable(0))
        self.assertFunctionMatchesEagerStatefulInput(for_one_var, inputs_)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, _int_tensor)))
    def test_while_two_vars(self, n, type_):
        if False:
            while True:
                i = 10
        n = type_(n)
        self.assertFunctionMatchesEager(while_two_vars, n)

    @parameterized.parameters(*itertools.product(([], [1], [1, 2]), (list, _int_tensor, _int_dataset), (True, False)))
    def test_for_two_vars(self, l, type_, xla):
        if False:
            for i in range(10):
                print('nop')
        if type_ is _int_dataset and xla:
            self.skipTest('Datsets not supported in XLA')
        if type_ is _int_tensor and xla and (not l):
            self.skipTest('Empty loops not supported in XLA')
        l = type_(l)
        self.assertFunctionMatchesEager(for_two_vars, l, xla=xla)

    @parameterized.parameters(([],), ([1],), ([1, 2],))
    def test_for_two_vars_ds_iterator(self, l):
        if False:
            return 10
        inputs_ = lambda : (iter(_int_dataset(l)), tf.Variable(0))
        self.assertFunctionMatchesEagerStatefulInput(for_two_vars, inputs_)

    @parameterized.parameters(*itertools.product(([], [1, 2]), ([], [1, 2]), (list, _int_tensor)))
    def test_for_else(self, l1, l2, type_):
        if False:
            print('Hello World!')
        l1 = type_(l1)
        l2 = type_(l2)
        with self.assertRaisesRegex(NotImplementedError, 'for/else'):
            tf.function(for_else)(l1, l2)

    @parameterized.parameters(*itertools.product((0, 2), (0, 2), (int, _int_tensor)))
    def test_while_else(self, n1, n2, type_):
        if False:
            i = 10
            return i + 15
        n1 = type_(n1)
        n2 = type_(n2)
        with self.assertRaisesRegex(NotImplementedError, 'while/else'):
            tf.function(while_else)(n1, n2)

    @parameterized.parameters(*itertools.product((1, 2), (int, _int_tensor)))
    def test_while_creates_var_legal(self, n, type_):
        if False:
            for i in range(10):
                print('nop')
        n = type_(n)
        self.assertFunctionMatchesEager(while_creates_var, n)

    def test_while_creates_var_illegal(self):
        if False:
            return 10
        with self.assertRaisesRegex(UnboundLocalError, 'used before assignment'):
            tf.function(while_creates_var)(0)
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'loop must iterate at least once'):
            tf.function(while_creates_var)(tf.constant(0))

    @parameterized.parameters(*itertools.product(([1], [1, 2]), (list, _int_tensor)))
    def test_for_creates_var_legal(self, l, type_):
        if False:
            i = 10
            return i + 15
        l = type_(l)
        self.assertFunctionMatchesEager(for_creates_var, l)

    def test_for_creates_var_illegal(self):
        if False:
            return 10
        with self.assertRaisesRegex(UnboundLocalError, 'used before assignment'):
            tf.function(for_creates_var)([])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'loop must iterate at least once'):
            tf.function(for_creates_var)(tf.constant([]))

    @parameterized.parameters(*double_product((0, 1, 2), (int, _int_tensor)))
    def test_successive_while_loops(self, n1, type1, n2, type2):
        if False:
            i = 10
            return i + 15
        n1 = type1(n1)
        n2 = type1(n2)
        self.assertFunctionMatchesEager(successive_while_loops, n1, n2)

    @parameterized.parameters(*double_product(([], [1], [1, 2]), (list, _int_tensor, _int_dataset)))
    def test_successive_for_loops(self, l1, type1, l2, type2):
        if False:
            return 10
        l1 = type1(l1)
        l2 = type1(l2)
        self.assertFunctionMatchesEager(successive_for_loops, l1, l2)

    @parameterized.parameters(*double_product(([], [1], [1, 2]), (list, _int_dataset)))
    def test_successive_for_loops_iterators(self, l1, type1, l2, type2):
        if False:
            for i in range(10):
                print('nop')
        inputs_ = lambda : (iter(type1(l1)), iter(type2(l2)))
        self.assertFunctionMatchesEagerStatefulInput(successive_for_loops, inputs_)

    @parameterized.parameters(*double_product((0, 1, 2), (int, _int_tensor)))
    def test_nested_while_loops(self, n1, type1, n2, type2):
        if False:
            for i in range(10):
                print('nop')
        n1 = type1(n1)
        n2 = type1(n2)
        self.assertFunctionMatchesEager(nested_while_loops, n1, n2)

    @parameterized.parameters(*itertools.product(([[]], [[], []], [[1]], [[1], [2]], [[1, 2]], [[1, 2], [3, 4]]), (_int_tensor, _int_dataset)))
    def test_nested_for_loops_dense(self, m, type_):
        if False:
            print('Hello World!')
        m = type_(m)
        self.assertFunctionMatchesEager(nested_for_loops, m)

    @parameterized.parameters(*itertools.product(([[]], [[], [1]], [[], [1], [1, 2]]), (list, functools.partial(tf.ragged.constant, dtype=tf.int32))))
    def test_nested_for_loops_ragged(self, m, type_):
        if False:
            print('Hello World!')
        m = type_(m)
        self.assertFunctionMatchesEager(nested_for_loops, m)

    def test_nested_for_loops_mixed_list(self):
        if False:
            for i in range(10):
                print('nop')
        m = [[], _int_tensor([]), [1], _int_tensor([1]), [1, 2]]
        self.assertFunctionMatchesEager(nested_for_loops, m)
if __name__ == '__main__':
    tf.test.main()