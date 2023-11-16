"""Simple call to construct a namedtuple."""
import collections
import tensorflow.compat.v1 as tf
from tensorflow.python.autograph.tests import reference_test_base

def inline_namedtuple(x):
    if False:
        while True:
            i = 10
    nt = collections.namedtuple('TestNamedTuple', ('a', 'b'))
    n = nt(a=1, b=x)
    return n

def external_namedtuple(x, nt):
    if False:
        while True:
            i = 10
    return nt(a=1, b=x)

class NamedTupleSubclass(collections.namedtuple('TestNamedTuple', ('a',))):

    def foo(self):
        if False:
            print('Hello World!')
        return self.a + 1

def namedtuple_subclass(x):
    if False:
        while True:
            i = 10
    nt = NamedTupleSubclass(x)
    return nt.foo()

class ReferenceTest(reference_test_base.TestCase):

    def test_inline(self):
        if False:
            return 10
        self.assertFunctionMatchesEager(inline_namedtuple, 1)
        self.assertFunctionMatchesEager(inline_namedtuple, tf.constant(1))

    def test_external(self):
        if False:
            return 10
        nt = collections.namedtuple('TestNamedTuple', ('a', 'b'))
        self.assertFunctionMatchesEager(external_namedtuple, 1, nt)
        self.assertFunctionMatchesEager(external_namedtuple, tf.constant(1), nt)

    def test_subclass(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFunctionMatchesEager(namedtuple_subclass, 1)
        self.assertFunctionMatchesEager(namedtuple_subclass, tf.constant(1))
if __name__ == '__main__':
    tf.test.main()