"""Function calls inside the while loop body."""
import itertools
from absl.testing import parameterized
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def while_with_call_in_cond(n, fn):
    if False:
        return 10
    i = 0
    s = 0
    while i < fn(n):
        s = s * 10 + i
        i += 1
    return s

def for_with_call_in_target(l, fn):
    if False:
        while True:
            i = 10
    s = 0
    for i in fn(l):
        s = s * 10 + i
    return s

def while_with_local_call_in_cond(n):
    if False:
        return 10

    def local_fn(x):
        if False:
            while True:
                i = 10
        return x * 3
    i = 0
    s = 0
    while i < local_fn(n):
        s = s * 10 + i
        i += 1
    return s

def for_with_local_call_in_target(l):
    if False:
        while True:
            i = 10

    def local_fn(l):
        if False:
            while True:
                i = 10
        return l * 1
    s = 0
    for i in local_fn(l):
        s = s * 10 + i
    return s

def while_with_call(n, fn):
    if False:
        for i in range(10):
            print('nop')
    i = 0
    s = 0
    while i < n:
        s = s * 10 + fn(i)
        i += 1
    return s

def for_with_call(l, fn):
    if False:
        for i in range(10):
            print('nop')
    s = 0
    for i in l:
        s = s * 10 + fn(i)
    return s

def while_with_local_call(n):
    if False:
        while True:
            i = 10

    def local_fn(x):
        if False:
            print('Hello World!')
        return x * 3
    i = 0
    s = 0
    while i < n:
        s = s * 10 + local_fn(i)
        i += 1
    return s

def for_with_local_call(l):
    if False:
        print('Hello World!')

    def local_fn(x):
        if False:
            i = 10
            return i + 15
        return x * 3
    s = 0
    for i in l:
        s = s * 10 + local_fn(i)
    return s

def while_with_closure_call(n):
    if False:
        for i in range(10):
            print('nop')
    i = 0

    def i_via_closure():
        if False:
            i = 10
            return i + 15
        return i + 2
    i = 0
    s = 0
    while i < n:
        s = s * 10 + i_via_closure()
        i += 1
    return s

def for_with_closure_call(l):
    if False:
        i = 10
        return i + 15
    i = 0

    def i_via_closure():
        if False:
            i = 10
            return i + 15
        return i + 2
    s = 0
    for i in l:
        s = s * 10 + i_via_closure()
    return (s, i)

def while_with_lambda_closure_call(n):
    if False:
        print('Hello World!')
    i = 0
    s = 0
    i_via_closure = lambda : i + 2
    while i < n:
        s = s * 10 + i_via_closure()
        i += 1
    return s

def for_with_lambda_closure_call(l):
    if False:
        while True:
            i = 10
    i = 0
    s = 0
    i_via_closure = lambda : i + 2
    for i in l:
        s = s * 10 + i_via_closure()
    return (s, i)

def while_with_method_closure_call(n):
    if False:
        print('Hello World!')
    i = 0

    class Callable(object):

        def __call__(self):
            if False:
                i = 10
                return i + 15
            return i
    i_via_closure = Callable()
    i = 0
    s = 0
    while i < n:
        s = s * 10 + i_via_closure()
        i += 1
    return s

def for_with_method_closure_call(l):
    if False:
        return 10
    i = 0

    class Callable(object):

        def __call__(self):
            if False:
                return 10
            return i
    i_via_closure = Callable()
    i = 0
    s = 0
    for i in l:
        s = s * 10 + i_via_closure()
    return (s, i)

def global_fn(x):
    if False:
        while True:
            i = 10
    return x * 2

class TestClass(object):

    def method(self, x):
        if False:
            while True:
                i = 10
        return x * 4

def _int_tensor(x):
    if False:
        for i in range(10):
            print('nop')
    return tf.constant(x, dtype=tf.int32)

class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, tf.constant), (global_fn, lambda x: x * 1, TestClass().method, abs)))
    def test_while_with_call_in_cond(self, n, type_, fn):
        if False:
            print('Hello World!')
        n = type_(n)
        self.assertFunctionMatchesEager(while_with_call_in_cond, n, fn)

    @parameterized.parameters(*itertools.product(([], [1], [1, 2]), (list, _int_tensor), (global_fn, lambda x: x * 1, TestClass().method, tf.abs)))
    def test_for_with_call_in_target(self, l, type_, fn):
        if False:
            i = 10
            return i + 15
        if fn is tf.abs and type_ is list:
            self.skipTest('tf.abs([]) defaults to float32')
        l = type_(l)
        self.assertFunctionMatchesEager(for_with_call_in_target, l, fn)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, _int_tensor), (range, tf.range)))
    def test_for_with_range_call_in_target(self, l, type_, fn):
        if False:
            i = 10
            return i + 15
        l = type_(l)
        self.assertFunctionMatchesEager(for_with_call_in_target, l, fn)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, tf.constant), (global_fn, lambda x: x * 1, TestClass().method, abs)))
    def test_while_with_call(self, n, type_, fn):
        if False:
            for i in range(10):
                print('nop')
        n = type_(n)
        self.assertFunctionMatchesEager(while_with_call, n, fn)

    @parameterized.parameters(*itertools.product(([], [1], [1, 2]), (list, _int_tensor), (global_fn, lambda x: x * 1, TestClass().method, abs)))
    def test_for_with_call(self, l, type_, fn):
        if False:
            while True:
                i = 10
        l = type_(l)
        self.assertFunctionMatchesEager(for_with_call, l, fn)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, tf.constant)))
    def test_while_with_local_call(self, n, type_):
        if False:
            while True:
                i = 10
        n = type_(n)
        self.assertFunctionMatchesEager(while_with_local_call, n)

    @parameterized.parameters(*itertools.product(([], [1], [1, 2]), (list, _int_tensor)))
    def test_for_with_local_call(self, l, type_):
        if False:
            i = 10
            return i + 15
        l = type_(l)
        self.assertFunctionMatchesEager(for_with_local_call, l)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, tf.constant)))
    def test_while_with_closure_call(self, n, type_):
        if False:
            while True:
                i = 10
        n = type_(n)
        self.assertFunctionMatchesEager(while_with_closure_call, n)

    @parameterized.parameters(*itertools.product(([], [1], [1, 2]), (list, _int_tensor)))
    def test_for_with_closure_call(self, l, type_):
        if False:
            i = 10
            return i + 15
        l = type_(l)
        self.assertFunctionMatchesEager(for_with_closure_call, l)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, tf.constant)))
    def test_while_with_lambda_closure_call(self, n, type_):
        if False:
            print('Hello World!')
        n = type_(n)
        self.assertFunctionMatchesEager(while_with_lambda_closure_call, n)

    @parameterized.parameters(*itertools.product(([], [1], [1, 2]), (list, _int_tensor)))
    def test_for_with_lambda_closure_call(self, l, type_):
        if False:
            print('Hello World!')
        l = type_(l)
        self.assertFunctionMatchesEager(for_with_lambda_closure_call, l)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, tf.constant)))
    def test_while_with_method_closure_call(self, n, type_):
        if False:
            while True:
                i = 10
        self.skipTest('fix static analysis for nested classes')
        n = type_(n)
        self.assertFunctionMatchesEager(while_with_method_closure_call, n)

    @parameterized.parameters(*itertools.product(([], [1], [1, 2]), (list, _int_tensor)))
    def test_for_with_method_closure_call(self, l, type_):
        if False:
            for i in range(10):
                print('nop')
        self.skipTest('fix static analysis for nested classes')
        l = type_(l)
        self.assertFunctionMatchesEager(for_with_method_closure_call, l)
if __name__ == '__main__':
    tf.test.main()