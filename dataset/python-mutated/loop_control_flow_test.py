"""Nested loops and loop control statements (e.g. break and continue).

Meant to verify that:
  * break/continue in the inner loop does not affect outer loop
  * break/continue inside nested conditionals still works
"""
import itertools
from absl.testing import parameterized
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def continue_in_single_for(l):
    if False:
        return 10
    s = 0
    for c in l:
        if c % 2 > 0:
            continue
        s += c
    return s

def continue_in_single_while(x):
    if False:
        for i in range(10):
            print('nop')
    s = 0
    while x > 0:
        x -= 1
        if x % 2 > 0:
            continue
        s += x
    return s

def continue_in_inner_for(m):
    if False:
        i = 10
        return i + 15
    s = 0
    for l in m:
        for c in l:
            if c % 2 > 0:
                continue
            s += c
    return s

def continue_in_inner_while(x, y):
    if False:
        return 10
    s = 0
    while x > 0:
        x -= 1
        while y > 0:
            y -= 1
            if (x + y) % 2 > 0:
                continue
            s += x + y
    return s

def break_in_single_for(l):
    if False:
        for i in range(10):
            print('nop')
    s = 0
    for c in l:
        if c % 2 > 0:
            break
        s += c
    return s

def unconditional_return_in_single_for(l):
    if False:
        for i in range(10):
            print('nop')
    s = 0
    for c in l:
        s += c
        return s
    return s

def effectively_unconditional_return_in_single_for(l):
    if False:
        while True:
            i = 10
    s = 0
    for c in l:
        s += c
        if c % 2 > 0:
            return s
        else:
            return -s
    return s

def unconditional_return_in_single_while(x):
    if False:
        i = 10
        return i + 15
    s = 0
    while x > 0:
        x -= 1
        s += x
        return s
    return s

def effectively_unconditional_return_in_single_while(x):
    if False:
        for i in range(10):
            print('nop')
    s = 0
    while x > 0:
        x -= 1
        s += x
        if x % 2 > 0:
            return s
        else:
            return -s
    return s

def break_in_single_while(x):
    if False:
        print('Hello World!')
    s = 0
    while x > 0:
        x -= 1
        if x % 2 > 0:
            break
        s += x
    return s

def break_in_inner_for(m):
    if False:
        while True:
            i = 10
    s = 0
    for l in m:
        for c in l:
            if c % 2 > 0:
                break
            s += c
    return s

def break_in_inner_while(x, y):
    if False:
        for i in range(10):
            print('nop')
    s = 0
    while x > 0:
        x -= 1
        while y > 0:
            y -= 1
            if (x + y) % 2 == 0:
                break
            s += x + y
    return s

def break_continue_in_inner_for(m):
    if False:
        while True:
            i = 10
    s = 0
    for l in m:
        for c in l:
            if c % 2 > 0:
                break
            else:
                continue
            s += c
    return s

def break_continue_in_inner_while(x, y):
    if False:
        print('Hello World!')
    s = 0
    while x > 0:
        x -= 1
        while y > 0:
            y -= 1
            if (x + y) % 2 > 0:
                break
            else:
                continue
            s += x + y
    return s

def break_followed_by_cond_in_single_for(x, y):
    if False:
        while True:
            i = 10
    for i in range(y):
        if i == 2:
            break
        if x > 0:
            x -= 1
    return x

def break_followed_by_cond_in_single_while(x):
    if False:
        return 10
    while x > 0:
        if x == 2:
            break
        if x > 0:
            x -= 1
    return x

def multiple_breaks_in_single_while(n):
    if False:
        i = 10
        return i + 15
    s = 1
    i = 0
    while i < n:
        i += 1
        if i > 10 * n:
            break
        if i == n:
            break
        s = s * 10 + i
    return (i, s)

def _int_tensor(x):
    if False:
        print('Hello World!')
    return tf.constant(x, dtype=tf.int32)

def _list_of_int_tensor(l):
    if False:
        i = 10
        return i + 15
    return [_int_tensor(x) for x in l]

def _int_dataset(l):
    if False:
        i = 10
        return i + 15
    return tf.data.Dataset.from_tensor_slices(tf.constant(l, dtype=tf.int32))

class LoopControlFlowTest(reference_test_base.TestCase, parameterized.TestCase):

    @parameterized.parameters(*itertools.product(([], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]), (list, _int_tensor, _int_dataset), (continue_in_single_for, break_in_single_for, unconditional_return_in_single_for, effectively_unconditional_return_in_single_for)))
    def test_single_for(self, l, type_, target):
        if False:
            i = 10
            return i + 15
        if type_ is _int_dataset and target in (unconditional_return_in_single_for, effectively_unconditional_return_in_single_for):
            self.skipTest('Creating symbols in dataset loops.')
        if not l and target in (unconditional_return_in_single_for, effectively_unconditional_return_in_single_for):
            self.skipTest('Undefined symbols require at least one iteration.')
        l = type_(l)
        self.assertFunctionMatchesEager(target, l)

    @parameterized.parameters(*itertools.product((0, 1, 2, 3, 4), (int, _int_tensor), (continue_in_single_while, break_in_single_while, multiple_breaks_in_single_while, break_followed_by_cond_in_single_while, unconditional_return_in_single_while, effectively_unconditional_return_in_single_while)))
    def test_single_while(self, n, type_, target):
        if False:
            while True:
                i = 10
        if not n and target in (unconditional_return_in_single_while, effectively_unconditional_return_in_single_while):
            self.skipTest('Undefined symbols require at least one iteration.')
        n = type_(n)
        self.assertFunctionMatchesEager(target, n)

    @parameterized.parameters((unconditional_return_in_single_for, _int_tensor, []), (effectively_unconditional_return_in_single_for, _int_tensor, []), (unconditional_return_in_single_while, _int_tensor, 0), (effectively_unconditional_return_in_single_while, _int_tensor, 0))
    def test_single_loop_illegal_return(self, target, type_, l):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'must iterate at least once to initialize'):
            tf.function(target)(type_(l))

    @parameterized.parameters(*itertools.product(([[], []], [[1], [2]], [[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6]]), (list, _int_tensor, _list_of_int_tensor, _int_dataset), (continue_in_inner_for, break_in_inner_for, break_continue_in_inner_for)))
    def test_nested_for(self, a, type_, target):
        if False:
            i = 10
            return i + 15
        a = type_(a)
        self.assertFunctionMatchesEager(target, a)

    @parameterized.parameters(*itertools.product((0, 1, 2, 3, 4), (0, 1, 2, 3, 4), (int, _int_tensor), (int, _int_tensor), (continue_in_inner_while, break_in_inner_while, break_continue_in_inner_while)))
    def test_nested_while(self, m, n, m_type, n_type, target):
        if False:
            print('Hello World!')
        m = m_type(m)
        n = m_type(n)
        self.assertFunctionMatchesEager(target, m, n)
if __name__ == '__main__':
    tf.test.main()