"""Tests that verify scoping around loops."""
import itertools
from absl.testing import parameterized
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def for_with_local_var(l):
    if False:
        return 10
    s = 0
    for i in l:
        x = i + 2
        s = s * 10 + x
    return s

def while_with_local_var(x):
    if False:
        return 10
    s = 0
    while x > 0:
        y = x + 2
        s = s * 10 + y
        x -= 1
    return s

def for_with_lambda_iter(l):
    if False:
        return 10
    fns = []
    results = []
    for i in l:
        fns.append(lambda : i)
    for f in fns:
        results.append(f())
    return results

def for_with_lambda_object():
    if False:
        return 10

    class SomeRandomObject:

        def bar(self, n):
            if False:
                print('Hello World!')
            return n + 1

    def foo_init():
        if False:
            print('Hello World!')
        return tf.constant(0)
    fns = []
    results = []
    foo = foo_init()
    for i in tf.range(3):
        foo = SomeRandomObject()
        fns.append(lambda i=i: foo.bar(i))
    for f in fns:
        results.append(f())
    return results

def for_with_lambda_iter_local_var(l):
    if False:
        return 10
    fns = []
    results = []
    for i in l:
        fns.append(lambda i=i: i)
    for f in fns:
        results.append(f())
    return results

def for_initializes_local_var(l):
    if False:
        for i in range(10):
            print('nop')
    s = 0
    for i in l:
        if i == l[0]:
            x = 0
        else:
            x += 1
        s = s * 10 + x
    return s

def while_initializes_local_var(x):
    if False:
        for i in range(10):
            print('nop')
    s = 0
    while x > 0:
        if x > 0:
            y = 0
        else:
            y += 1
        s = s * 10 + y
        x -= 1
    return s

def for_defines_var(l):
    if False:
        print('Hello World!')
    for i in l:
        x = i + 2
    return x

def while_defines_var(x):
    if False:
        for i in range(10):
            print('nop')
    while x > 0:
        y = x + 2
        x -= 1
    return y

def for_defines_iterate(n, fn):
    if False:
        print('Hello World!')
    s = 0
    for i in fn(n):
        s = s * 10 + i
    return (i, s)

def for_reuses_iterate(n, fn):
    if False:
        while True:
            i = 10
    i = 7
    s = 0
    for i in fn(n):
        s = s * 10 + i
    return (i, s)

def for_alters_iterate(n, fn):
    if False:
        while True:
            i = 10
    i = 7
    s = 0
    for i in fn(n):
        i = 3 * i + 1
        s = s * 10 + i
    return (i, s)

def _int_tensor(x):
    if False:
        for i in range(10):
            print('nop')
    return tf.constant(x, dtype=tf.int32)

class LoopScopingTest(reference_test_base.TestCase, parameterized.TestCase):

    @parameterized.parameters(*itertools.product(([], [1], [1, 2]), (list, _int_tensor)))
    def test_for_with_local_var(self, l, type_):
        if False:
            return 10
        l = type_(l)
        self.assertFunctionMatchesEager(for_with_local_var, l)

    @parameterized.parameters(*itertools.product((0, 1, 2), (range, tf.range)))
    def test_for_with_local_var_range(self, l, type_):
        if False:
            while True:
                i = 10
        l = type_(l)
        self.assertFunctionMatchesEager(for_with_local_var, l)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, _int_tensor)))
    def test_while_with_local_var(self, x, type_):
        if False:
            return 10
        x = type_(x)
        self.assertFunctionMatchesEager(while_with_local_var, x)

    @parameterized.parameters(([],), ([1],), ([1, 2],))
    def test_for_initializes_local_var_legal_cases(self, l):
        if False:
            return 10
        self.assertFunctionMatchesEager(for_initializes_local_var, l)

    @parameterized.parameters(([],), ([1],), ([1, 2],))
    def test_for_initializes_local_var_illegal_cases(self, l):
        if False:
            while True:
                i = 10
        self.skipTest('TODO(mdanatg): Check')
        l = tf.constant(l)
        with self.assertRaisesRegex(ValueError, '"x" must be defined'):
            tf.function(for_initializes_local_var)(l)

    @parameterized.parameters(0, 1, 2)
    def test_while_initializes_local_var_legal_cases(self, x):
        if False:
            print('Hello World!')
        self.assertFunctionMatchesEager(while_initializes_local_var, x)

    @parameterized.parameters(0, 1, 2)
    def test_while_initializes_local_var_illegal_cases(self, x):
        if False:
            i = 10
            return i + 15
        self.skipTest('TODO(mdanatg): check')
        x = tf.constant(x)
        with self.assertRaisesRegex(ValueError, '"y" must be defined'):
            tf.function(while_initializes_local_var)(x)

    @parameterized.parameters(([1],), ([1, 2],))
    def test_for_defines_var_legal_cases(self, l):
        if False:
            print('Hello World!')
        self.assertFunctionMatchesEager(for_defines_var, l)

    @parameterized.parameters(([],), ([1],), ([1, 2],))
    def test_for_defines_var_illegal_cases(self, l):
        if False:
            return 10
        self.skipTest('TODO(mdanatg): check')
        l = tf.constant(l)
        with self.assertRaisesRegex(ValueError, '"x" must be defined'):
            tf.function(for_defines_var)(l)

    @parameterized.parameters((1,), (2,))
    def test_while_defines_var_legal_cases(self, x):
        if False:
            print('Hello World!')
        self.assertFunctionMatchesEager(while_defines_var, x)

    @parameterized.parameters((0,), (1,), (2,))
    def test_while_defines_var_illegal_cases(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.skipTest('TODO(mdanatg): check')
        x = tf.constant(x)
        with self.assertRaisesRegex(ValueError, '"y" must be defined'):
            tf.function(while_defines_var)(x)

    @parameterized.parameters(*itertools.product((1, 2), (range, tf.range)))
    def test_for_defines_iterate_legal_cases(self, n, fn):
        if False:
            return 10
        self.assertFunctionMatchesEager(for_defines_iterate, n, fn)

    def test_for_defines_iterate_range(self):
        if False:
            return 10
        self.skipTest('b/155171694')

    def test_for_defines_iterate_tf_range(self):
        if False:
            i = 10
            return i + 15
        self.assertAllEqual(tf.function(for_defines_iterate)(0, tf.range), (0, 0))

    @parameterized.parameters(*itertools.product(([], [1], [1, 2]), (list, _int_tensor)))
    def test_for_reuses_iterate(self, l, fn):
        if False:
            print('Hello World!')
        self.assertFunctionMatchesEager(for_reuses_iterate, l, fn)

    @parameterized.parameters(*itertools.product((0, 1, 2), (range, tf.range)))
    def test_for_reuses_iterate_range(self, n, fn):
        if False:
            print('Hello World!')
        self.assertFunctionMatchesEager(for_reuses_iterate, n, fn)

    @parameterized.parameters(*itertools.product(([], [1], [1, 2]), (list, _int_tensor)))
    def test_for_alters_iterate(self, l, fn):
        if False:
            print('Hello World!')
        self.assertFunctionMatchesEager(for_alters_iterate, l, fn)

    @parameterized.parameters(*itertools.product((0, 1, 2), (range, tf.range)))
    def test_for_alters_iterate_range(self, n, fn):
        if False:
            return 10
        self.assertFunctionMatchesEager(for_alters_iterate, n, fn)

class LoopLambdaScopingTest(reference_test_base.TestCase, parameterized.TestCase):

    @parameterized.parameters(*itertools.product(([], [1], [1, 2], [(1, 2), (3, 4)]), (list, list)))
    def test_for_with_lambda_iter(self, l, type_):
        if False:
            i = 10
            return i + 15
        self.skipTest('https://github.com/tensorflow/tensorflow/issues/56089')
        l = type_(l)
        self.assertFunctionMatchesEager(for_with_lambda_iter, l)

    def test_for_with_lambda_object(self):
        if False:
            print('Hello World!')
        self.skipTest('https://github.com/tensorflow/tensorflow/issues/56089')
        self.assertFunctionMatchesEager(for_with_lambda_object)

    @parameterized.parameters(*itertools.product(([], [1], [1, 2], [(1, 2), (3, 4)]), (list, list)))
    def test_for_with_lambda_iter_local_var(self, l, type_):
        if False:
            for i in range(10):
                print('nop')
        self.skipTest('https://github.com/tensorflow/tensorflow/issues/56089')
        l = type_(l)
        self.assertFunctionMatchesEager(for_with_lambda_iter_local_var, l)
if __name__ == '__main__':
    tf.test.main()