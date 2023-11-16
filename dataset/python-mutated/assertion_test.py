"""Basic assertions."""
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def simple_assertion(x):
    if False:
        while True:
            i = 10
    assert x > 0
    return x

class ReferenceTest(reference_test_base.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(ReferenceTest, self).setUp()
        self.autograph_opts = tf.autograph.experimental.Feature.ASSERT_STATEMENTS

    def test_basic(self):
        if False:
            return 10
        self.assertFunctionMatchesEager(simple_assertion, 1)
        self.assertFunctionMatchesEager(simple_assertion, tf.constant(1))
        with self.assertRaises(AssertionError):
            self.function(simple_assertion)(0)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.function(simple_assertion)(tf.constant(0))
if __name__ == '__main__':
    tf.test.main()