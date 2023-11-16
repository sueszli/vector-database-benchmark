"""Multiple returns, some in conditionals."""
import itertools
from absl.testing import parameterized
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def return_with_default(x):
    if False:
        i = 10
        return i + 15
    if x > 0:
        tf.print('x', x)
        return x
    return x * x

def return_dependent_on_local(c):
    if False:
        i = 10
        return i + 15
    t = tf.constant(1)
    if c:
        return t
    t = tf.stack([t, t])
    return tf.reduce_sum(t)

def return_possibly_undefined(x):
    if False:
        return 10
    if x > 0:
        if x < 5:
            return x
    else:
        return x * x * x

def nested_ifs(x):
    if False:
        while True:
            i = 10
    if x > 0:
        if x < 5:
            return x
        else:
            return x * x
    else:
        return x * x * x

def possible_return_before_loop(c1, c2, n):
    if False:
        return 10
    if c1:
        if c2:
            return 1
    for _ in range(n):
        pass
    return 2

def nested_ifs_and_context_managers(x):
    if False:
        for i in range(10):
            print('nop')
    with tf.name_scope(''):
        if x > 0:
            if x < 5:
                with tf.name_scope(''):
                    return x
            else:
                return x * x
        else:
            return x * x * x

def unreachable_return(x):
    if False:
        return 10
    with tf.name_scope(''):
        if x > 0:
            if x < 5:
                with tf.name_scope(''):
                    return x
            else:
                return x * x
        else:
            return x * x * x
    return x * x * x * x

def return_with_default_in_contexmanager(x):
    if False:
        print('Hello World!')
    with tf.name_scope(''):
        if x > 0:
            return 1
        return 0

def return_in_try_with_finally(x):
    if False:
        return 10
    try:
        if x > 0:
            return 1
        else:
            return 0
    finally:
        x = x + 1

def return_with_default_in_try_with_finally(x):
    if False:
        while True:
            i = 10
    try:
        if x > 0:
            return 1
        return 0
    finally:
        x = x + 1

def return_in_finally(x):
    if False:
        i = 10
        return i + 15
    try:
        return 2
    finally:
        if x > 0:
            return 1
        else:
            return 0

def return_with_default_in_finally(x):
    if False:
        i = 10
        return i + 15
    try:
        return 2
    finally:
        if x > 0:
            return 1
        return 0

def return_in_finally_default_in_try(x):
    if False:
        while True:
            i = 10
    try:
        if x > 0:
            return 0
    finally:
        return 1

def _raising_helper():
    if False:
        while True:
            i = 10
    raise ValueError()

def raise_during_return_caught():
    if False:
        for i in range(10):
            print('nop')
    try:
        return _raising_helper()
    except ValueError:
        pass
    return 1

def raise_during_return_caught_in_tail_branch(c):
    if False:
        while True:
            i = 10
    if c:
        return 2
    try:
        return _raising_helper()
    except ValueError:
        pass
    return 1

class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):
    """Base class for the reference tests."""

    @parameterized.parameters(*itertools.product((0, 1), (int, tf.constant)))
    def test_return_with_default(self, n, type_):
        if False:
            return 10
        self.assertFunctionMatchesEager(return_with_default, type_(n))

    @parameterized.parameters(*itertools.product((True, False), (int, tf.constant)))
    def test_return_dependent_on_local(self, c, type_):
        if False:
            while True:
                i = 10
        self.assertFunctionMatchesEager(return_dependent_on_local, type_(c))

    @parameterized.parameters((0,), (3,), (5,))
    def test_return_possibly_undefined_legal(self, n):
        if False:
            while True:
                i = 10
        self.assertFunctionMatchesEager(return_possibly_undefined, n)

    @parameterized.parameters((0,), (3,), (5,))
    def test_return_possibly_undefined_illegal(self, n):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'else branch must also have a return'):
            tf.function(return_possibly_undefined)(tf.constant(n))

    @parameterized.parameters(*itertools.product((-1, 3, 6), (int, tf.constant)))
    def test_nested_ifs(self, n, type_):
        if False:
            for i in range(10):
                print('nop')
        self.assertFunctionMatchesEager(nested_ifs, type_(n))

    @parameterized.parameters(*itertools.product((True, False), (True, False), (0, 1, 2)))
    def test_possible_return_before_loop(self, c1, c2, n):
        if False:
            i = 10
            return i + 15
        self.assertFunctionMatchesEager(possible_return_before_loop, c1, c2, n)

    @parameterized.parameters(*itertools.product((0, 3, 5), (int, tf.constant)))
    def test_nested_ifs_and_context_managers(self, x, type_):
        if False:
            return 10
        self.assertFunctionMatchesEager(nested_ifs_and_context_managers, type_(x))

    @parameterized.parameters(*itertools.product((0, 3, 5), (int, tf.constant)))
    def test_unreachable_return(self, x, type_):
        if False:
            while True:
                i = 10
        self.assertFunctionMatchesEager(unreachable_return, type_(x))

    @parameterized.parameters(*itertools.product((0, 1), (int, tf.constant)))
    def test_return_with_default_in_contexmanager(self, x, type_):
        if False:
            i = 10
            return i + 15
        self.assertFunctionMatchesEager(return_with_default_in_contexmanager, type_(x))

    @parameterized.parameters(*itertools.product((0, 1), (int, tf.constant)))
    def test_return_in_try_finally(self, x, type_):
        if False:
            print('Hello World!')
        self.assertFunctionMatchesEager(return_in_try_with_finally, type_(x))

    @parameterized.parameters(*itertools.product((0, 1), (int, tf.constant)))
    def test_return_with_default_try_finally(self, x, type_):
        if False:
            while True:
                i = 10
        self.assertFunctionMatchesEager(return_with_default_in_try_with_finally, type_(x))

    @parameterized.parameters(*itertools.product((0, 1), (int, tf.constant)))
    def test_return_in_finally(self, x, type_):
        if False:
            for i in range(10):
                print('nop')
        self.assertFunctionMatchesEager(return_in_finally, type_(x))

    @parameterized.parameters(*itertools.product((0, 1), (int, tf.constant)))
    def test_return_with_default_in_finally(self, x, type_):
        if False:
            while True:
                i = 10
        self.assertFunctionMatchesEager(return_with_default_in_finally, type_(x))

    @parameterized.parameters(*itertools.product((0, 1), (int, tf.constant)))
    def test_return_in_finally_default_in_try(self, x, type_):
        if False:
            i = 10
            return i + 15
        self.assertFunctionMatchesEager(return_in_finally_default_in_try, type_(x))

    def test_raise_during_return_caught(self):
        if False:
            i = 10
            return i + 15
        self.assertFunctionMatchesEager(raise_during_return_caught)

    @parameterized.parameters(*itertools.product((True, False), (int, tf.constant)))
    def test_raise_during_return_caught_in_tail_branch(self, c, type_):
        if False:
            print('Hello World!')
        self.assertFunctionMatchesEager(raise_during_return_caught_in_tail_branch, type_(c))
if __name__ == '__main__':
    tf.test.main()