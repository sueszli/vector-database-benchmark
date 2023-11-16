"""Simple call to lambda functions."""
import tensorflow.compat.v1 as tf
from tensorflow.python.autograph.tests import reference_test_base

def inline_lambda(x):
    if False:
        while True:
            i = 10
    l = lambda x: x * x if x > 0 else -x
    return l(x)

def external_lambda(x, l):
    if False:
        return 10
    return l(x)

class ReferenceTest(reference_test_base.TestCase):

    def test_inline(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFunctionMatchesEager(inline_lambda, 1)
        self.assertFunctionMatchesEager(inline_lambda, tf.constant(1))

    def test_external(self):
        if False:
            return 10
        self.assertFunctionMatchesEager(external_lambda, 1, lambda x: x == 0)
        self.assertFunctionMatchesEager(external_lambda, tf.constant(1), lambda x: x == 0)
if __name__ == '__main__':
    tf.test.main()