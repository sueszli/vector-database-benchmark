"""Basic conditionals."""
import itertools
import re
from absl.testing import parameterized
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def if_no_vars(c, v):
    if False:
        return 10
    v.assign(0)
    if c:
        v.assign_add(1)
    return v.read_value()

def if_else_no_vars(c, v):
    if False:
        i = 10
        return i + 15
    v.assign(0)
    if c:
        v.assign_add(1)
    else:
        v.assign_add(2)
    return v.read_value()

def if_one_var(n):
    if False:
        while True:
            i = 10
    i = 0
    if i < n:
        i += 1
    return i

def if_else_one_var(n):
    if False:
        i = 10
        return i + 15
    i = 0
    if i < n:
        i += 1
    else:
        i += 2
    return i

def if_two_vars(n):
    if False:
        while True:
            i = 10
    i = 0
    j = 1
    if i < n:
        i += 1
        j *= 10
    return (i, j)

def if_else_two_vars(n):
    if False:
        for i in range(10):
            print('nop')
    i = 0
    j = 1
    if i < n:
        i += 1
        j *= 10
    else:
        i += 2
        j *= 20
    return (i, j)

def if_creates_var(c):
    if False:
        i = 10
        return i + 15
    if c:
        i = 1
    return i

def if_else_creates_var(c):
    if False:
        i = 10
        return i + 15
    if c:
        i = 1
    else:
        i = 2
    return i

def else_creates_var(c):
    if False:
        return 10
    if c:
        pass
    else:
        i = 2
    return i

def if_destroys_var(c):
    if False:
        i = 10
        return i + 15
    i = 1
    if c:
        del i
    return i

def if_else_destroys_var(c):
    if False:
        return 10
    i = 1
    if c:
        del i
    else:
        del i
    return i

def else_destroys_var(c):
    if False:
        i = 10
        return i + 15
    i = 2
    if c:
        pass
    else:
        del i
    return i

def if_returns_none(c):
    if False:
        while True:
            i = 10
    i = 0
    j = 1
    if c:
        i = None
        j = 2
    return (i, j)

def if_else_returns_none(c):
    if False:
        print('Hello World!')
    if c:
        i = None
        j = 1
    else:
        i = None
        j = 2
    return (i, j)

def else_returns_none(c):
    if False:
        for i in range(10):
            print('nop')
    i = 1
    j = 1
    if c:
        pass
    else:
        i = None
        j = 2
    return (i, j)

def if_local_var(c):
    if False:
        i = 10
        return i + 15
    i = 0
    if c:
        j = 1
        i = j + 1
    return i

def if_else_local_var(c):
    if False:
        print('Hello World!')
    i = 0
    if c:
        j = 1
    else:
        j = 2
        i = j + 1
    return i

def if_locally_modified_var(c):
    if False:
        i = 10
        return i + 15
    i = 0
    j = 2
    if c:
        j = j + 1
        i = j + 1
    return i

def successive_ifs(n1, n2):
    if False:
        while True:
            i = 10
    s = 0
    i = 0
    if i < n1:
        s = s * 10 + i
        i += 1
    i = 0
    if i < n2:
        s = s * 10 + i
        i += 1
    return s

def successive_if_elses(n1, n2):
    if False:
        for i in range(10):
            print('nop')
    s = 0
    i = 0
    if i < n1:
        s = s * 10 + i
        i += 1
    else:
        s = s * 11 + i
        i += 2
    i = 0
    if i < n2:
        s = s * 10 + i
        i += 1
    else:
        s = s * 11 + i
        i += 2
    return s

def nested_ifs(n1, n2):
    if False:
        return 10
    i = 0
    l = 0
    if i < n1:
        j = 0
        s = 0
        if j < n2:
            s = s * 10 + i * j
            j += 1
        l = l * 1000 + s
        i += 1
    return l

def nested_if_temporarily_undefined_return(c1, c2):
    if False:
        while True:
            i = 10
    if c1:
        if c2:
            return 1
    return 2

def nested_if_elses(n1, n2):
    if False:
        print('Hello World!')
    i = 0
    l = 0
    if i < n1:
        j = 0
        s = 0
        if j < n2:
            s = s * 10 + i * j
            j += 1
        else:
            s = s * 11 + i * j
            j += 2
        l = l * 1000 + s
        i += 1
    else:
        j = 0
        s = 0
        if j < n2:
            s = s * 12 + i * j
            j += 3
        else:
            s = s * 13 + i * j
            j += 4
        l = l * 2000 + s
        i += 1
    return l

class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):

    @parameterized.parameters(*itertools.product((if_no_vars, if_else_no_vars), (True, False), (bool, tf.constant)))
    def test_no_vars(self, target, c, type_):
        if False:
            for i in range(10):
                print('nop')
        c = type_(c)
        self.assertFunctionMatchesEager(target, c, tf.Variable(0))

    @parameterized.parameters(*itertools.product((if_one_var, if_else_one_var, if_two_vars, if_else_two_vars), (0, 1), (int, tf.constant)))
    def test_several_vars(self, target, n, type_):
        if False:
            i = 10
            return i + 15
        n = type_(n)
        self.assertFunctionMatchesEager(target, n)

    def test_var_lifetime_imbalanced_legal(self):
        if False:
            return 10
        self.assertFunctionMatchesEager(if_creates_var, True)
        self.assertFunctionMatchesEager(else_creates_var, False)
        self.assertFunctionMatchesEager(if_destroys_var, False)
        self.assertFunctionMatchesEager(else_destroys_var, True)

    @parameterized.parameters(*itertools.product((True, False), (int, tf.constant)))
    def test_if_else_var_lifetime(self, c, type_):
        if False:
            for i in range(10):
                print('nop')
        c = type_(c)
        self.assertFunctionMatchesEager(if_else_creates_var, c)
        if type_ is int:
            with self.assertRaisesRegex(UnboundLocalError, "'i'"):
                tf.function(if_else_destroys_var)(c)
        else:
            with self.assertRaisesRegex(ValueError, "'i' must also be initialized"):
                tf.function(if_else_destroys_var)(c)

    @parameterized.parameters((if_creates_var, False, bool, UnboundLocalError, "'i' is used before assignment"), (if_creates_var, True, tf.constant, ValueError, "'i' must also be initialized in the else branch"), (if_creates_var, False, tf.constant, ValueError, "'i' must also be initialized in the else branch"), (else_creates_var, True, bool, UnboundLocalError, "'i' is used before assignment"), (else_creates_var, True, tf.constant, ValueError, "'i' must also be initialized in the main branch"), (else_creates_var, False, tf.constant, ValueError, "'i' must also be initialized in the main branch"))
    def test_creates_var_imbalanced_illegal(self, target, c, type_, exc_type, exc_regex):
        if False:
            while True:
                i = 10
        c = type_(c)
        with self.assertRaisesRegex(exc_type, exc_regex):
            tf.function(target)(c)

    def test_returns_none_legal(self):
        if False:
            while True:
                i = 10
        self.assertFunctionMatchesEager(if_returns_none, True)
        self.assertFunctionMatchesEager(if_else_returns_none, False)
        self.assertFunctionMatchesEager(else_returns_none, False)

    @parameterized.parameters((if_returns_none, True), (if_returns_none, False), (else_returns_none, True), (else_returns_none, False), (if_else_returns_none, True), (if_else_returns_none, False))
    def test_returns_none_illegal(self, target, c):
        if False:
            print('Hello World!')
        c = tf.constant(c)
        with self.assertRaisesRegex(ValueError, re.compile("'i' is None", re.DOTALL)):
            tf.function(target)(c)

    @parameterized.parameters(*itertools.product((if_local_var, if_else_local_var, if_locally_modified_var), (True, False), (bool, tf.constant)))
    def test_local_vars(self, target, c, type_):
        if False:
            for i in range(10):
                print('nop')
        c = type_(c)
        self.assertFunctionMatchesEager(target, c)

    @parameterized.parameters(*itertools.product((True, False), (True, False)))
    def test_nested_if_temporarily_undefined_return_legal(self, c1, c2):
        if False:
            return 10
        self.assertFunctionMatchesEager(nested_if_temporarily_undefined_return, c1, c2)

    @parameterized.parameters(*itertools.product((True, False), (True, False)))
    def test_nested_if_temporarily_undefined_return_illegal(self, c1, c2):
        if False:
            i = 10
            return i + 15
        c1 = tf.constant(c1)
        c2 = tf.constant(c2)
        with self.assertRaisesRegex(ValueError, 'must also have a return'):
            tf.function(nested_if_temporarily_undefined_return)(c1, c2)

    @parameterized.parameters(*itertools.product((successive_ifs, successive_if_elses, nested_ifs, nested_if_elses), (0, 1), (bool, tf.constant), (0, 1), (bool, tf.constant)))
    def test_composition(self, target, n1, n1_type, n2, n2_type):
        if False:
            i = 10
            return i + 15
        n1 = n1_type(n1)
        n2 = n2_type(n2)
        self.assertFunctionMatchesEager(target, n1, n2)
if __name__ == '__main__':
    tf.test.main()