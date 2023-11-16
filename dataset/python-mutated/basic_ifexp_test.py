"""Basic if conditional.

The loop is converted to tf.cond.
"""
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def consecutive_conds(x):
    if False:
        for i in range(10):
            print('nop')
    if x > 0:
        x = -x if x < 5 else x
    else:
        x = -2 * x if x < 5 else 1
    if x > 0:
        x = -x if x < 5 else x
    else:
        x = 3 * x if x < 5 else x
    return x

def cond_with_multiple_values(x):
    if False:
        i = 10
        return i + 15
    if x > 0:
        x = -x if x < 5 else x
        y = 2 * x if x < 5 else x
        z = -y if y < 5 else y
    else:
        x = 2 * x if x < 5 else x
        y = -x if x < 5 else x
        z = -y if y < 5 else y
    return (x, y, z)

class ReferenceTest(reference_test_base.TestCase):

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        for x in [-1, 1, 5, tf.constant(-1), tf.constant(1), tf.constant(5)]:
            self.assertFunctionMatchesEager(consecutive_conds, x)
            self.assertFunctionMatchesEager(cond_with_multiple_values, x)
if __name__ == '__main__':
    tf.test.main()