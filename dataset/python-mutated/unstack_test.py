"""Tests for tensorflow.compiler.tests.unstack."""
from absl.testing import parameterized
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.platform import test

class UnstackOpTest(xla_test.XLATestCase, parameterized.TestCase):

    def _test(self, size):
        if False:
            while True:
                i = 10
        with self.session() as sess:
            x_tf = array_ops.placeholder(np.float32, shape=[size, 512])
            with self.test_scope():
                ret = array_ops_stack.unstack(x_tf)
            ret_vals = sess.run([ret], feed_dict={x_tf: np.zeros([size, 512])})
            self.assertLen(ret_vals[0], size)
            for ret_val in ret_vals[0]:
                self.assertTrue(np.all(ret_val == 0.0))

    def testLarge2000(self):
        if False:
            for i in range(10):
                print('nop')
        self._test(2000)
if __name__ == '__main__':
    test.main()