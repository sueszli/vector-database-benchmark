"""Composite names (attributes) in control flow.

Generally, composite symbols should be treated like regular ones.
"""
import itertools
from absl.testing import parameterized
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def if_basic_dict(a):
    if False:
        for i in range(10):
            print('nop')
    x = {'a': a}
    if x['a'] > 0:
        x['b'] = 1
    else:
        x['b'] = -1
    return x

def if_basic_list(a):
    if False:
        return 10
    x = [a, 0]
    if x[0] > 0:
        x[1] = 1
    else:
        x[1] = -1
    return x

def if_imbalanced(a):
    if False:
        print('Hello World!')
    x = {'a': a}
    if x['a'] > 0:
        x['b'] = 1
    return x

def else_imbalanced(a):
    if False:
        print('Hello World!')
    x = {'a': a}
    if x['a'] > 0:
        pass
    else:
        x['b'] = 1
    return x

def if_imbalanced_nested(a):
    if False:
        print('Hello World!')
    x = {'a': {'a': a}}
    if x['a']['a'] > 0:
        x['a']['b'] = 1
    return x

def else_imbalanced_nested(a):
    if False:
        return 10
    x = {'a': {'a': a}}
    if x['a']['a'] > 0:
        pass
    else:
        x['a']['b'] = 1
    return x

def if_buggy(a):
    if False:
        return 10
    x = {'a': {'a': a}}
    if x['a']['a'] > 0:
        x['b']['a'] = 1
    else:
        x['b']['a'] = -1
    return x

def if_imbalanced_buggy(a):
    if False:
        for i in range(10):
            print('nop')
    x = {'a': {'a': a}}
    if x['a']['a'] > 0:
        x['b']['a'] = 1
    return x

def while_basic_dict(x, a, b):
    if False:
        while True:
            i = 10
    y = {'a': a, 'b': b}
    while x > 0:
        x -= 1
        y['a'] += 1
    return y

def while_basic_list(x, a, b):
    if False:
        print('Hello World!')
    y = [a, b]
    while x > 0:
        x -= 1
        y[0] += 1
    return y

def while_state_only_dict(a, b):
    if False:
        return 10
    y = {'a': a, 'b': b}
    while y['b'] <= 10:
        y['a'] += 1
        y['b'] *= 2
    return y

def while_state_only_list(a, b):
    if False:
        while True:
            i = 10
    y = [a, b]
    while y[1] <= 10:
        y[0] += 1
        y[1] *= 2
    return y

def while_imbalanced(b):
    if False:
        while True:
            i = 10
    y = {'b': b}
    while y['b'] <= 10:
        y['a'] = y['b'] + 1
        y['b'] *= 2
    return y

def for_basic_dict(n, x, a, b):
    if False:
        i = 10
        return i + 15
    y = {'a': a, 'b': b}
    for i in range(n):
        x -= 1
        y['a'] += i
    return y

def for_basic_list(n, x, a, b):
    if False:
        print('Hello World!')
    y = [a, b]
    for i in range(n):
        x -= 1
        y[0] += i
    return y

def for_state_only_dict(n, a, b):
    if False:
        return 10
    y = {'a': a, 'b': b}
    for _ in range(n):
        y['a'] += 1
    return y

def for_state_only_list(n, a, b):
    if False:
        return 10
    y = [a, b]
    for _ in range(n):
        y[0] += 1
    return y

def for_imbalanced(n, x):
    if False:
        return 10
    y = {}
    for i in range(n):
        x -= i
        y['a'] = x
    return y

class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):

    @parameterized.parameters(*itertools.product((if_basic_dict, if_basic_list), (0, 1), (bool, tf.constant)))
    def test_if_basic(self, target, a, type_):
        if False:
            i = 10
            return i + 15
        a = type_(a)
        self.assertFunctionMatchesEager(target, a)

    @parameterized.parameters(*itertools.product((if_imbalanced, else_imbalanced, if_imbalanced_nested, else_imbalanced_nested), (0, 1)))
    def test_if_imbalanced_legal(self, target, a):
        if False:
            return 10
        self.assertFunctionMatchesEager(target, a)

    @parameterized.parameters((if_imbalanced, 0, tf.constant, ValueError, "'x\\['b'\\]' must also be initialized in the else"), (if_imbalanced, 1, tf.constant, ValueError, "'x\\['b'\\]' must also be initialized in the else"), (else_imbalanced, 0, tf.constant, ValueError, "'x\\['b'\\]' must also be initialized in the main"), (else_imbalanced, 1, tf.constant, ValueError, "'x\\['b'\\]' must also be initialized in the main"), (if_imbalanced_nested, 0, tf.constant, ValueError, "'x\\['a'\\]\\['b'\\]' must also be initialized in the else"), (if_imbalanced_nested, 1, tf.constant, ValueError, "'x\\['a'\\]\\['b'\\]' must also be initialized in the else"), (else_imbalanced_nested, 0, tf.constant, ValueError, "'x\\['a'\\]\\['b'\\]' must also be initialized in the main"), (else_imbalanced_nested, 1, tf.constant, ValueError, "'x\\['a'\\]\\['b'\\]' must also be initialized in the main"), (if_buggy, 1, int, KeyError, "'b'"), (if_buggy, 1, tf.constant, KeyError, "'b'"), (if_buggy, 0, tf.constant, KeyError, "'b'"), (if_imbalanced_buggy, 1, int, KeyError, "'b'"), (if_imbalanced_buggy, 1, tf.constant, KeyError, "'b'"), (if_imbalanced_buggy, 0, tf.constant, KeyError, "'b'"))
    def test_if_imbalanced_illegal(self, target, a, type_, exc_type, exc_regex):
        if False:
            return 10
        a = type_(a)
        with self.assertRaisesRegex(exc_type, exc_regex):
            tf.function(target)(a)

    @parameterized.parameters(*itertools.product((while_basic_dict, while_basic_list), (0, 1, 2), (bool, tf.constant), (3, 7)))
    def test_while_basic(self, target, x, type_, a):
        if False:
            i = 10
            return i + 15
        x = type_(x)
        self.assertFunctionMatchesEager(target, x, a, 0)

    @parameterized.parameters(*itertools.product((while_state_only_dict, while_state_only_list), (bool, tf.constant), (3, 4, 5, 6)))
    def test_while_state_only(self, target, type_, b):
        if False:
            return 10
        b = type_(b)
        self.assertFunctionMatchesEager(target, 0, b)

    @parameterized.parameters(*itertools.product((5, 10, 11), (int, tf.constant)))
    def test_while_imbalanced_legal(self, b, type_):
        if False:
            while True:
                i = 10
        if b == 11 and type_ is tf.constant:
            self.skipTest("TF loop must initialize y['a']")
        b = type_(b)
        self.assertFunctionMatchesEager(while_imbalanced, b)

    def test_while_imbalanced_illegal(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "loop must iterate at least once to initialize y\\[\\\\'a\\\\'\\]"):
            tf.function(while_imbalanced)(tf.constant(11))

    @parameterized.parameters(*itertools.product((for_basic_dict, for_basic_list), (0, 1, 2), (bool, tf.constant)))
    def test_for_basic(self, target, n, type_):
        if False:
            print('Hello World!')
        n = type_(n)
        self.assertFunctionMatchesEager(target, n, 1, 1, 1)

    @parameterized.parameters(*itertools.product((for_state_only_dict, for_state_only_list), (0, 1, 2), (bool, tf.constant)))
    def test_for_state_only(self, target, n, type_):
        if False:
            for i in range(10):
                print('nop')
        n = type_(n)
        self.assertFunctionMatchesEager(target, n, 1, 1)

    @parameterized.parameters(*itertools.product((0, 1, 2), (int, tf.constant)))
    def test_for_imbalanced_legal(self, n, type_):
        if False:
            while True:
                i = 10
        if n == 0 and type_ is tf.constant:
            self.skipTest("TF loop must initialize y['a']")
        n = type_(n)
        self.assertFunctionMatchesEager(for_imbalanced, n, 0)

    def test_for_imbalanced_illegal(self):
        if False:
            return 10
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "loop must iterate at least once to initialize y\\[\\\\'a\\\\'\\]"):
            tf.function(for_imbalanced)(tf.constant(0), 0)
if __name__ == '__main__':
    tf.test.main()